import json
import re
import argparse
import tqdm
from openai import OpenAI
from typing import Dict, Any, List
import logging
import sys
import numpy as np
from collections import Counter
from config import Config

# Setup logging with more informative format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

evaluation_prompt = """Evaluate the conversation between the patient and doctor based on the provided patient character card and conversation history. Assess the patient's responses according to the following criteria: character consistency, character recall, conversational coherence, and realism. Provide a score for each criterion on a scale of 0 to 1, where 1 represents perfect performance. Take special importance into the coherence between the symptomps and dialogue.
**Character card:**
{script}
**Conversation history:**
{dialogue}

Respond only with the scores, no further explanation"""

evaluation_schema = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Judge Response",
    "type": "object",
    "properties": {
        "character_consistency": {
            "type": "object",
            "properties": {
                "score": {"type": "number", "minimum": 0, "maximum": 1}
            },
            "required": ["score"]
        },
        "character_recall": {
            "type": "object",
            "properties": {
                "score": {"type": "number", "minimum": 0, "maximum": 1}
            },
            "required": ["score"]
        },
        "conversational_coherence": {
            "type": "object",
            "properties": {
                "score": {"type": "number", "minimum": 0, "maximum": 1}
            },
            "required": ["score"]
        },
        "realism": {
            "type": "object",
            "properties": {
                "score": {"type": "number", "minimum": 0, "maximum": 1}
            },
            "required": ["score"]
        }
    },
    "required": ["character_consistency", "character_recall", "conversational_coherence", "realism"]
}

def evaluate(client: OpenAI, dialogue: Dict[str, Any], evaluation_prompt: str) -> Dict[str, Any]:
    """
    Evaluate a dialogue using the OpenAI API.
    
    Args:
        client: OpenAI client instance
        dialogue: Dictionary containing script and dialogue
        evaluation_prompt: Template for the evaluation prompt
    
    Returns:
        Dict containing evaluation scores
    """
    try:
        evaluation = client.chat.completions.create(
            messages=[
                {
                    'role': 'user',
                    'content': evaluation_prompt.format(
                        script=dialogue['script'], 
                        dialogue=dialogue['dialogue']
                    )
                }
            ],
            temperature=1.1,
            response_format={"type": "json"},
            model="gpt-4",  # You should replace this with your actual model
        )
        return json.loads(evaluation.choices[0].message.content)
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        return None

def process_dialogue_file(input_path: str, client: OpenAI, evaluation_prompt: str) -> None:
    """
    Process the dialogue file and evaluate each dialogue.
    
    Args:
        input_path: Path to the input JSONL file
        client: OpenAI client instance
        evaluation_prompt: Template for the evaluation prompt
    """
    logger.info(f"Starting evaluation of dialogues from: {input_path}")
    try:
        with open(input_path, 'r') as f:
            lines = [line for line in f]
            total_dialogues = len(lines)
            logger.info(f"Found {total_dialogues} dialogues to evaluate")
            
            f.seek(0)
            results = []
            pbar = tqdm.tqdm(f, total=total_dialogues, desc="Evaluating dialogues")
            
            evaluation_failures = 0
            for i, line in enumerate(pbar):
                try:
                    dialogue = json.loads(line)
                    result = evaluate(client, dialogue, evaluation_prompt)
                    if result:
                        results.append({
                            'dialogue_id': i,
                            'evaluation': result
                        })
                        
                        # Log meaningful evaluation metrics
                        scores = [
                            result["character_consistency"]["score"],
                            result["character_recall"]["score"],
                            result["conversational_coherence"]["score"],
                            result["realism"]["score"]
                        ]
                        avg_score = sum(scores) / len(scores)
                        
                        if i % 10 == 0:  # Log every 10 dialogues to avoid excessive logging
                            pbar.write(f"Dialogue {i} - Avg score: {avg_score:.2f} - Consistency: {scores[0]:.2f}, Recall: {scores[1]:.2f}, Coherence: {scores[2]:.2f}, Realism: {scores[3]:.2f}")
                    else:
                        evaluation_failures += 1
                        pbar.write(f"Failed to evaluate dialogue {i}")
                except json.JSONDecodeError:
                    logger.error(f"Error parsing dialogue {i} - Invalid JSON")
                except Exception as e:
                    logger.error(f"Error processing dialogue {i}: {str(e)}")
            
            # Save results
            output_path = input_path.replace('.jsonl', '_evaluated.jsonl')
            with open(output_path, 'w') as f:
                for result in results:
                    f.write(json.dumps(result) + '\n')
            
            success_rate = ((total_dialogues - evaluation_failures) / total_dialogues) * 100 if total_dialogues > 0 else 0
            logger.info(f"Evaluation complete. {len(results)} dialogues evaluated successfully ({success_rate:.1f}%).")
            logger.info(f"Results saved to {output_path}")
                    
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during evaluation: {str(e)}")
        sys.exit(1)

def evaluate_dialogue(dialogue: Dict) -> Dict[str, float]:
    """
    Evaluate dialogue quality based on multiple metrics.
    Returns a dictionary of scores.
    """
    metrics = {
        'turn_balance': evaluate_turn_balance(dialogue),
        'content_relevance': evaluate_content_relevance(dialogue),
        'language_consistency': evaluate_language_consistency(dialogue),
        'interaction_quality': evaluate_interaction_quality(dialogue)
    }
    
    # Compute overall score
    metrics['overall_score'] = np.mean(list(metrics.values()))
    return metrics

def evaluate_turn_balance(dialogue: Dict) -> float:
    """Evaluate the balance of turns between doctor and patient"""
    roles = [msg['role'] for msg in dialogue['dialogue']]
    role_counts = Counter(roles)
    
    # Perfect balance would be equal counts
    total = sum(role_counts.values())
    expected = total / len(role_counts)
    
    # Calculate deviation from perfect balance
    deviations = [abs(count - expected) for count in role_counts.values()]
    max_possible_deviation = total
    
    return 1 - (sum(deviations) / max_possible_deviation)

def evaluate_content_relevance(dialogue: Dict) -> float:
    """Evaluate how relevant the dialogue content is to the patient's script"""
    script = dialogue.get('script', '')
    dialogue_text = ' '.join(msg['content'] for msg in dialogue['dialogue'])
    
    # Extract key terms from script
    script_terms = set(re.findall(r'\b\w+\b', script.lower()))
    dialogue_terms = set(re.findall(r'\b\w+\b', dialogue_text.lower()))
    
    # Calculate Jaccard similarity
    intersection = len(script_terms & dialogue_terms)
    union = len(script_terms | dialogue_terms)
    
    return intersection / union if union > 0 else 0.0

def evaluate_language_consistency(dialogue: Dict) -> float:
    """Evaluate consistency of language use (Spanish)"""
    spanish_markers = [
        r'\b(?:el|la|los|las)\b',
        r'\b(?:que|porque|pero|si)\b',
        r'\b(?:está|estoy|estás)\b'
    ]
    
    text = ' '.join(msg['content'] for msg in dialogue['dialogue'])
    total_words = len(re.findall(r'\b\w+\b', text))
    spanish_words = sum(len(re.findall(pattern, text)) for pattern in spanish_markers)
    
    return min(1.0, spanish_words / (total_words * 0.1))  # Expect at least 10% Spanish markers

def evaluate_interaction_quality(dialogue: Dict) -> float:
    """Evaluate the quality of doctor-patient interaction"""
    interaction_markers = {
        'greeting': r'\b(?:hola|buenos días|buenas tardes)\b',
        'politeness': r'\b(?:por favor|gracias|perdón)\b',
        'questions': r'\?',
        'acknowledgment': r'\b(?:entiendo|comprendo|claro)\b'
    }
    
    text = ' '.join(msg['content'] for msg in dialogue['dialogue'])
    scores = []
    
    for marker_type, pattern in interaction_markers.items():
        matches = len(re.findall(pattern, text, re.IGNORECASE))
        scores.append(min(1.0, matches / 2))  # Expect at least 2 occurrences
        
    return np.mean(scores)

def batch_evaluate(dialogues: List[Dict]) -> List[Dict[str, float]]:
    """Evaluate a batch of dialogues"""
    return [evaluate_dialogue(d) for d in dialogues]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate doctor-patient dialogues")
    parser.add_argument("--dataset_input", type=str, default='dialogo_medico-paciente_es_Llama-3.1-8B-Q8.jsonl', help="Path to dialogue dataset to evaluate")
    parser.add_argument("--config", type=str, default='config.yaml', help="Path to config file")
    parser.add_argument("--log_level", type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
                       default='INFO', help="Set the logging level")
    args = parser.parse_args()

    # Set log level from command line
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    logger.info(f"Starting dialogue evaluation with log level: {args.log_level}")

    try:
        logger.info(f"Loading configuration from {args.config}")
        config = Config(args.config)
        
        logger.info("Initializing OpenAI client")
        client = OpenAI(
            base_url=config.get('llm.base_url'),
            api_key=config.get('llm.api_key'),
            timeout=config.get('llm.timeout', 70000)
        )
        process_dialogue_file(args.dataset_input, client, evaluation_prompt)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)