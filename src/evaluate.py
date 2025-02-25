import json
import re
import argparse
import tqdm
from openai import OpenAI

parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument("--dataset_input", type=str, default='dialogo_medico-paciente_es_Llama-3.1-8B-Q8.jsonl', help="Path to dialogue dataset to evaluate")
parser.add_argument("--base_url", type=str, default='http://localhost:7000/v1/', help="Base URL for API requests")

# Parse arguments
args = parser.parse_args()

evaluation_prompt = """Evaluate the conversation between the patient and doctor based on the provided patient character card and conversation history. Assess the patient's responses according to the following criteria: character consistency, character recall, conversational coherence, and realism. Provide a score for each criterion on a scale of 0 to 1, where 1 represents perfect performance. Take special importance into the coherence between the symptomps and dialogue.
**Character card:**
{script}
**Conversation history:**
{dialogue}

Respond only with the scores, no further explanation"""

evaluation_schema = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Judge Response",
    "type": "json_object",
    "properties": {
        "character_consistency": {
        "type": "json_object",
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

def evaluate(client, dialogue, evaluation_prompt):
    evaluation = client.chat.completions.create(
        messages=[
            {
                'role': 'user',
                'content': evaluation_prompt.format(script=dialogue['script'], dialogue=dialogue['dialogue'])
            }
        ],
        temperature=1.1,
        response_format=evaluation_schema,
        model=""
    )
    return evaluation.choices[0].message.content

if __name__ == "__main__":
    
    client = OpenAI(
        base_url=args.base_url,
        api_key='llamacpp',
        timeout=70000
    )

    with open(args.dataset_input, 'r') as f:
        lines = [line for line in f]
        f.seek(0)
        pbar = tqdm.tqdm(f, total=len(lines), desc="Evaluating dialogues")
        for i, line in enumerate(pbar):
            dialogue = json.loads(line)
            result = evaluate(client, dialogue, evaluation_prompt)
            pbar.write(f"Result: {result}")