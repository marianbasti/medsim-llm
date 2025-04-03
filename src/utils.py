import json
import logging
import tenacity
from typing import Dict, Any, Tuple, Union
from tqdm import tqdm
from schemas import SPANISH_CHARS, prompt_script_2, prompt_dialog
from json_schemas import dialogue_json_schema, patient_script_json_schema

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('dialogue_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add retry decorator for API calls
@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
    retry=tenacity.retry_if_exception_type((TimeoutError, ConnectionError)),
    before_sleep=lambda retry_state: logger.warning(f"API call failed, retrying in {retry_state.next_action.sleep} seconds...")
)
def api_call_with_retry(client, messages, temperature, response_format, model):
    """Make an API call with retry logic"""
    if not model:
        raise ValueError("Model ID is required")
    
    logger.debug(f"API call to model: {model}")
    logger.debug(f"Temperature: {temperature}")
    logger.debug(f"Response format schema: {json.dumps(response_format, indent=2)}")
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            extra_body={"guided_json": response_format}
        )
        logger.debug(f"API call successful. Response ID: {response.id}")
        return response
    except Exception as e:
        logger.error(f"API call failed: {str(e)}")
        # Log detailed error information
        if hasattr(e, 'response'):
            logger.error(f"Response status: {e.response.status_code}")
            logger.error(f"Response content: {e.response.text}")
        raise

def sanitize_spanish_text(text: Union[str, dict, list]) -> Union[str, dict, list]:
    """
    Recursively sanitize Spanish text by replacing Unicode escape sequences with proper characters.
    Can handle strings, dictionaries, and lists.
    """
    if isinstance(text, str):
        result = text
        for escaped, char in SPANISH_CHARS.items():
            result = result.replace(escaped, char)
        return result
    elif isinstance(text, dict):
        return {k: sanitize_spanish_text(v) for k, v in text.items()}
    elif isinstance(text, list):
        return [sanitize_spanish_text(item) for item in text]
    return text

# Transform the data to train with LMFlow
def lmflow_training_format(data):
    """
    Para cargar el jsonl
    with open('/path/to/file.jsonl') as f:
        data = [json.loads(line) for line in f]
    """
    transformed_data = [{"text": f"<bos>{entry['script']}\n{entry['dialogue']}"} for entry in data]
    # We add transformed data as the "instances key" to a json
    return {"type":"text_only","instances": transformed_data}

def dataset2sharegpt(input_file, output_file):
    all_conversations = {"conversations": []}
    
    logger.info(f"Converting data from {input_file} to ShareGPT format")
    with open(input_file, 'r', encoding='utf-8') as f_in:
        lines = [line for line in f_in]
        conversation_count = 0
        
        for line in tqdm(lines, desc="Converting to ShareGPT"):
            data = json.loads(line)
            
            # Create a new conversation with the 'script' as the first message from "system"
            new_conversation = [{
                "from": "system",
                "value": data['script']
            }]
            
            # Since 'dialogue' is a list, iterate over it
            dialogue = data['dialogue']
            
            # Enforce conversation alternation between roles
            filtered_dialogue = []
            last_role = None
            for message in dialogue:
                if message['role'] != last_role:  # Enforce alternating roles
                    filtered_dialogue.append(message)
                    last_role = message['role']
            # Remove last message if it's from "doctor"
            if filtered_dialogue and filtered_dialogue[-1]['role'] == "doctor":
                filtered_dialogue.pop()
            
            # Append the filtered dialogue items
            for message in filtered_dialogue:
                new_conversation.append({
                    "from": "user" if message['role'] == "doctor" else "assistant",
                    "value": message['content']
                })
            
            if len(new_conversation) > 1:
                # Add the transformed conversation to the list
                all_conversations['conversations'].append(
                    new_conversation
                )
                conversation_count += 1
        
    # Write the final aggregated data to the output JSON file
    with open(output_file, 'w', encoding='utf-8') as f_out:
        json.dump(all_conversations, f_out, ensure_ascii=False, indent=4)
    
    logger.info(f"Conversion complete. {conversation_count} conversations saved to {output_file}")

def synth_dialogue(client, prompt_dialog, script, model_id):
    """
    Synthesize a patient-doctor dialogue using vLLM structured output.
    Args:
        client: The OpenAI API client.
        prompt_dialog (str): The prompt template for generating a dialog.
        script (str): The patient script.
        model_id (str): The ID of the model to use.
    
    Returns:
        dict: The synthesized patient-doctor dialogue.
    """
    try:
        formatted_prompt = prompt_dialog.format(patient_script=script)
        logger.debug(f"Dialogue prompt length: {len(formatted_prompt)} characters")
        logger.debug(f"Using dialogue schema: {type(dialogue_json_schema).__name__}")
        
        # Log a sample of the schema for debugging
        logger.debug(f"Dialogue schema structure: {dialogue_json_schema.keys() if isinstance(dialogue_json_schema, dict) else 'Not a dict'}")
        
        message_dialog = api_call_with_retry(
            client,
            messages=[{'role': 'user', 'content': formatted_prompt}],
            temperature=1.0,
            response_format=dialogue_json_schema,
            model=model_id
        )
        
        # Get content from the response
        content = message_dialog.choices[0].message.content
        logger.debug(f"Received dialogue response type: {type(content).__name__}")
        
        # Check if content is already a dict or if it needs parsing
        if isinstance(content, str):
            try:
                # Try to parse it as JSON
                logger.debug("Parsing string response as JSON")
                parsed_content = json.loads(content)
                logger.debug(f"Successfully parsed JSON response with keys: {parsed_content.keys() if isinstance(parsed_content, dict) else 'Not a dict'}")
                return parsed_content
            except json.JSONDecodeError as e:
                # If it can't be parsed, wrap it in a proper structure
                logger.warning(f"Received non-JSON response from model: {str(e)}")
                logger.debug(f"Response sample: {content[:100]}...")  # Log first 100 chars
                return {"dialogue": [{"role": "doctor", "content": content}]}
        else:
            # Content is already a dictionary
            logger.debug(f"Received dictionary response with keys: {content.keys() if hasattr(content, 'keys') else 'No keys'}")
            return content
    except Exception as e:
        logger.error(f"Error generating dialogue: {str(e)}")
        raise

def synth_script(client, prompt_script, model_id):
    """
    Synthesize a dataset of patient scripts using vLLM structured output.
    Args:
        client: The OpenAI API client.
        prompt_script (str): The prompt template for creating a patient script.
        model_id (str): The ID of the model to use.
    
    Returns:
        str: The synthesized patient script.
    """
    try:
        logger.debug(f"Patient script prompt length: {len(prompt_script)} characters")
        logger.debug(f"Using patient schema: {type(patient_script_json_schema).__name__}")
        
        # Log a sample of the schema for debugging
        logger.debug(f"Patient schema structure: {patient_script_json_schema.keys() if isinstance(patient_script_json_schema, dict) else 'Not a dict'}")
        
        patient_script = api_call_with_retry(
            client,
            messages=[{'role': 'user', 'content': prompt_script}],
            temperature=1.1,
            response_format=patient_script_json_schema,
            model=model_id
        )
        
        content = patient_script.choices[0].message.content
        logger.debug(f"Received patient script response type: {type(content).__name__}")
        
        if isinstance(content, str):
            logger.debug(f"Patient script response sample: {content[:100]}...")  # Log first 100 chars
        else:
            logger.debug(f"Patient script response keys: {content.keys() if hasattr(content, 'keys') else 'No keys'}")
            
        return content
    except Exception as e:
        logger.error(f"Error generating patient script: {str(e)}")
        raise

def save_batch(batch, output_file):
    """Save a batch of dialogues to file with error handling"""
    try:
        with open(output_file, 'a', encoding='utf-8') as f:
            for item in batch:
                # Sanitize the item before saving
                sanitized_item = sanitize_spanish_text(item)
                f.write(json.dumps(sanitized_item, ensure_ascii=False) + '\n')
        logger.info(f"Saved batch of {len(batch)} samples")
    except Exception as e:
        logger.error(f"Error saving batch: {str(e)}")
        # Save to backup file
        backup_file = output_file + '.backup'
        logger.warning(f"Attempting to save to backup file: {backup_file}")
        with open(backup_file, 'a', encoding='utf-8') as f:
            for item in batch:
                # Sanitize the item before saving to backup
                sanitized_item = sanitize_spanish_text(item)
                f.write(json.dumps(sanitized_item, ensure_ascii=False) + '\n')
        logger.info(f"Batch saved to backup file: {backup_file}")

def generate_sample(client, name: str, illness: Tuple[str, str], model_id: str) -> Dict[str, Any]:
    """Generate a dialogue for a given patient case"""
    try:
        script = synth_script(client, prompt_script_2.format(
            name=name, 
            illness=illness[0],
            symptoms=illness[1]
        ), model_id)
        
        # Generate dialogue
        dialogue = synth_dialogue(client, prompt_dialog, script, model_id)
        
        return {"script": script, "dialogue": dialogue['dialogue']}
    except Exception as e:
        logger.error(f"Error generating sample for {name} with {illness[0]}: {str(e)}")
        return None
