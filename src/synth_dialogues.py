import json, random
import re
import argparse
import logging
from tqdm import tqdm
import tenacity
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Tuple
from openai import OpenAI
from config import Config
from pydantic import BaseModel, Field
from enum import Enum
from typing import List, Optional, Union

# Configure logging with more informative format
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
    return client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        extra_body={"guided_json": response_format}
    )

parser = argparse.ArgumentParser()
# Add arguments
parser.add_argument("--n_samples", type=int, default=20500, help="Number of dataset samples to generate")
parser.add_argument("--dataset_output", type=str, default='dialogo_medico-paciente_es.json', help="Path to JSON output")
parser.add_argument("--base_url", type=str, default='http://localhost:7000/v1/', help="Base URL for API requests")

# Parse arguments
args = parser.parse_args()

### DIALOG PROMPT AND SCHEMA
prompt_dialog = """Generate a natural conversation in Argentinian Spanish between a doctor and a patient based on the following patient script: {patient_script}. The patient should not reveal all the information at once, and the conversation should reflect the doctor's active listening, clarity, respect, reassurance, patient-centered care, clear communication, encouragement, confidentiality, and effective time management. The doctor should NOT perform any actions, just talk.
The doctor should NOT perform any actions, just talk.
Only provide the spoken dialogue, without any additional explanation, performance or translation.
"""

# Updated schema for vLLM - dialogue schema
class Role(str, Enum):
    doctor = "doctor"
    patient = "patient"

class DialogueMessage(BaseModel):
    role: Role
    content: str

class Dialogue(BaseModel):
    dialogue: List[DialogueMessage]

# Create schema for dialogue using Pydantic
dialogue_json_schema = Dialogue.model_json_schema()

### PATIENT CARD PROMPT AND JSON SCHEMA
prompt_script_2 = """Generate a patient card for {name} that has {illness}. Their symptoms are: {symptoms}.
The patient card must have with the following features: demographics, medical history, current symptoms, personal details, behavioral and cognitive factors, and healthcare utilization. Ensure that the patient card is realistic and diverse in terms of age, sex, occupation, and medical conditions. Use natural language. The features should be realistic and coherent. Let's have a complex and low-income social context, in a country with free healthcare. The patient's dialogue must be short and simple."""

# Pydantic models for patient script schema
class Demographics(BaseModel):
    age: int
    sex: str
    occupation: str
    education_level: str = Field(alias="education level")

class MedicalHistory(BaseModel):
    conditions: List[str]
    medications: List[str]
    allergies: List[str]
    surgical_history: List[str] = Field(alias="surgical history")

class CurrentSymptoms(BaseModel):
    chief_complaint: str = Field(alias="chief complaint")
    duration: str
    severity: str
    associated_symptoms: str = Field(alias="associated symptoms")

class PersonalDetails(BaseModel):
    lifestyle_habits: str = Field(alias="lifestyle habits")
    family_dynamics: str = Field(alias="family dynamics")
    work: str
    social_history: str = Field(alias="social history")
    mental_health_history: str = Field(alias="mental health history")

class BehavioralCognitiveFactors(BaseModel):
    personality_traits: str = Field(alias="personality traits")
    cognitive_function: str = Field(alias="cognitive function")
    behavioral_patterns: str = Field(alias="behavioral patterns")

class HealthcareUtilization(BaseModel):
    recent_hospitalizations: bool
    recent_hospitalizations_cause: Optional[str] = None
    emergency_room_visits: bool
    emergency_room_visits_cause: Optional[str] = None

class PatientScript(BaseModel):
    Name: str
    Demographics: Demographics
    Medical_History: MedicalHistory = Field(alias="Medical History")
    Current_Symptoms: CurrentSymptoms = Field(alias="Current Symptoms")
    Personal_Details: PersonalDetails = Field(alias="Personal Details")
    Behavioral_and_Cognitive_Factors: BehavioralCognitiveFactors = Field(alias="Behavioral and Cognitive Factors")
    Healthcare_Utilization: HealthcareUtilization = Field(alias="Healthcare Utilization")

# Create schema for patient script using Pydantic
patient_script_json_schema = PatientScript.model_json_schema()

argentinian_names = [
    "Juan Pérez",
    "María González",
    "Carlos Rodríguez",
    "Ana Fernández",
    "Martín López",
    "Laura Martínez",
    "Pedro García",
    "Sofía Ramírez",
    "Jorge Díaz",
    "Lucía Torres",
    "Miguel Sánchez",
    "Valentina Castro",
    "Fernando Romero",
    "Juliana Ríos",
    "Alejandro Silva",
    "Camila Gómez",
    "Ricardo Morales",
    "Florencia Herrera",
    "Emiliano Muñoz",
    "Daniela Ruiz",
    "Federico Sosa",
    "Carolina Ortiz",
    "Gabriel Pérez",
    "Verónica Álvarez",
    "Santiago Méndez",
    "Paula Jiménez",
    "Gustavo Navarro",
    "Marta Ponce",
    "Agustín Soto",
    "Bianca Vega",
    "Marcos Correa",
    "Rocío Campos",
    "Sebastián Gil",
    "Victoria Medina",
    "Hernán Delgado",
    "Milagros Núñez",
    "Ramiro Aguirre",
    "Natalia Acosta",
    "Diego Varela",
    "Andrea Villalba",
    "Nicolás Salinas",
    "Mariana Espíndola",
    "Ezequiel Ojeda",
    "Claudia Guzmán",
    "Francisco Castro",
    "Gabriela Salgado",
    "Gonzalo Benítez",
    "Micaela Maldonado",
    "Hugo Reyes",
    "Leandro Castro",
    "Romina López",
    "Matías Gómez",
    "Eliana Molina",
    "Esteban Romero",
    "Daniela Torres",
    "Iván Rojas",
    "Cecilia Cabrera",
    "Maximiliano Navarro",
    "Victoria Aguilar",
    "Tomás Mendoza",
    "Brenda Cabrera",
    "Pablo Luna",
    "Soledad Montoya",
    "Cristian Vallejos",
    "Belén Flores",
    "Mauricio Toledo",
    "Luciana Pérez",
    "Roberto Figueroa",
    "Silvia Romero",
    "Mariano Vargas",
    "Carina Ramos",
    "Enzo Villalobos",
    "Alejandra Duarte",
    "Sergio Pereyra",
    "Ailén Vázquez",
    "Damián Gómez",
    "Josefina Roldán",
    "Leandro Méndez",
    "Mónica Olivera",
    "Matías Soria",
]

illneses = [
    ["Common Cold", "Runny nose with clear discharge, scratchy throat, dry cough, slight congestion, feeling tired, temperature of 37.2°C"],
    ["Influenza (Flu)", "Sudden onset of 39.5°C fever, severe body aches especially in legs and back, extreme fatigue, pounding headache, sore throat, dry cough that worsens at night"],
    ["Strep Throat", "Severe pain when swallowing, 38.9°C fever, swollen tonsils with white patches, tender lymph nodes in neck, loss of appetite"],
    ["Urinary Tract Infection (UTI)", "Frequent urination every 30 minutes, intense burning sensation when urinating, cloudy urine with strong odor, dull ache in lower abdomen"],
    ["Migraine", "Throbbing pain on left side of head, nausea without vomiting, extreme sensitivity to office fluorescent lights, visual aura preceding headache"],
    ["Hypertension (High Blood Pressure)", "Blood pressure reading of 152/95 mmHg, mild headache at back of head, slightly blurred vision, feeling of general unease"],
    ["Type 2 Diabetes", "Excessive thirst leading to drinking 4 liters of water daily, urinating 7-8 times per day, blurred vision when reading, slow-healing paper cut on index finger, unusual fatigue after meals"],
    ["Asthma", "Wheezing especially when lying down at night, shortness of breath when climbing stairs, tight feeling in chest, dry cough that worsens with exercise"],
    ["Gastroesophageal Reflux Disease (GERD)", "Burning sensation in chest 30 minutes after eating, regurgitation of sour liquid when bending over, difficulty swallowing large pills, persistent cough especially at night"],
    ["Depression", "Feeling sad and empty for the past 6 weeks, lost interest in favorite hobby, sleeping 10-12 hours but still feeling tired, difficulty making decisions at work"],
    ["Anxiety Disorder", "Excessive worry about minor issues, restlessness and inability to sit still, racing thoughts that interfere with sleep, muscle tension in neck and shoulders"],
    ["Allergies", "Sneezing fits when exposed to pollen, itchy and watery eyes, stuffy nose, itchy rash on arms after petting a cat"],
    ["Pneumonia", "Productive cough with greenish phlegm, fever of 38.7°C, chills and sweating, shortness of breath when walking short distances, sharp chest pain when coughing"],
    ["Acute Bronchitis", "Persistent cough for 10 days producing yellow mucus, mild chest discomfort, fatigue, low-grade fever of 37.8°C"],
    ["Skin Infection (Cellulitis)", "Red, swollen, warm area on lower left leg, pain and tenderness to touch, fever of 38.3°C, small cut visible at center of affected area"],
    ["Osteoarthritis", "Pain and stiffness in right knee, especially in the morning and after sitting for long periods, creaking sound when moving the joint, difficulty kneeling"],
    ["Hyperthyroidism", "Unexplained weight loss of 5kg in 2 months, increased appetite, feeling jittery and anxious, sensitivity to heat, irregular heartbeat"],
    ["Iron-deficiency Anemia", "Extreme fatigue even after 9 hours of sleep, pale inner eyelids, shortness of breath when climbing one flight of stairs, craving for ice"],
    ["Chronic Obstructive Pulmonary Disease (COPD)", "Shortness of breath when doing light housework, chronic cough that's worse in the morning, wheezing when breathing out, having to sleep with head elevated"],
    ["Irritable Bowel Syndrome (IBS)", "Cramping abdominal pain that improves after bowel movement, alternating constipation and diarrhea, bloating that worsens throughout the day, mucus in stool"],
    ["Chronic Kidney Disease", "Swelling in legs and ankles, fatigue, shortness of breath, metallic taste in mouth, decreased urine output, high blood pressure"],
    ["Rheumatoid Arthritis", "Pain and swelling in multiple joints, especially in hands and feet, morning stiffness lasting more than 1 hour, fatigue, low-grade fever"],
    ["Mild Cognitive Impairment", "Forgetting recent events, difficulty finding words in conversation, trouble following recipe instructions, losing track of time, misplacing keys"],
    ["Hypothyroidism", "Unexplained weight gain of 4kg in 3 months, feeling cold all the time, dry skin, hair loss, constipation, fatigue"],
    ["Gout", "Sudden onset of severe pain, redness, and swelling in big toe, pain worsens at night, skin feels warm to touch, limited range of motion in joint"],
    ["Sinusitis", "Severe pressure behind eyes, thick yellow nasal discharge, reduced sense of smell, 37.8°C low-grade fever, fatigue worsening in the afternoon"],
    ["Mononucleosis", "Extreme fatigue for 3 weeks, sore throat with swollen tonsils, 38.3°C fever spiking in evenings, swollen lymph nodes in neck, loss of appetite"],
    ["Appendicitis", "Sharp pain starting near navel then moving to lower right abdomen, nausea without vomiting, low-grade fever of 37.5°C, constipation, loss of appetite"],
    ["Plantar Fasciitis", "Stabbing pain in right heel, especially with first steps in the morning, pain decreases with activity but returns after long periods of standing"],
    ["Vertigo", "Sudden spinning sensation lasting 2-3 minutes, nausea, balance problems, feeling of fullness in left ear, symptoms worsen with head movements"],
    ["Gallstones", "Intense pain in upper right abdomen lasting 2-3 hours, pain radiating to right shoulder blade, nausea, pain triggered 1 hour after eating fatty meals"],
    ["Psoriasis", "Red, scaly patches on elbows and knees, silvery scale on patches, itching and burning sensation, patches worsening with stress and dry weather"],
    ["Glaucoma", "Gradual loss of peripheral vision in left eye, seeing halos around lights at night, occasional eye pain, slight redness in eye"],
    ["Gout", "Sudden, severe pain in left big toe joint, swelling and redness of the joint, joint feels warm to touch, pain is worse at night, inability to wear normal shoes"],
    ["Ulcerative Colitis", "Bloody diarrhea 4-5 times a day, abdominal cramping on left side, urgency to defecate, 3kg weight loss in past month, fatigue, loss of appetite"],
    ["Carpal Tunnel Syndrome", "Numbness and tingling in thumb, index, and middle fingers of right hand, symptoms worse at night, dropping objects occasionally, weak grip strength"],
    ["Hypothyroidism", "Unexplained weight gain of 4kg in 2 months, feeling cold all the time, constipation, dry skin, fatigue, depression"],
    ["Meniere's Disease", "Episodes of vertigo lasting 20 minutes, ringing in left ear, feeling of fullness in ear, fluctuating hearing loss in left ear"],
    ["Shingles", "Painful, blistering rash on left side of torso, pain described as burning or stabbing, sensitivity to touch in affected area, 37.6°C low-grade fever, fatigue"],
    ["Celiac Disease", "Chronic diarrhea, abdominal bloating and pain, fatigue, 5kg weight loss despite normal appetite, anemia, mouth ulcers"],
    ["Rheumatoid Arthritis", "Symmetric joint pain and swelling in hands and wrists, morning stiffness lasting more than an hour, fatigue, low-grade fever of 37.4°C, loss of appetite"],
    ["Sleep Apnea", "Loud snoring reported by partner, waking up gasping for air, excessive daytime sleepiness, morning headaches, difficulty concentrating at work"],
    ["Fibromyalgia", "Widespread muscle pain and tenderness, fatigue despite sleeping 8-9 hours, difficulty concentrating ('fibro fog'), heightened sensitivity to touch, headaches"],
    ["Rotator Cuff Tendinitis", "Pain in right shoulder, especially when reaching overhead or behind back, weakness when lifting arm, pain worsens at night, clicking sound with movement"],
    ["Peptic Ulcer", "Burning pain in upper abdomen, pain improves briefly when eating then worsens, nausea, feeling uncomfortably full after eating small meals, unexplained weight loss of 3kg"]
]

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
        message_dialog = api_call_with_retry(
            client,
            messages=[{'role': 'user', 'content': prompt_dialog.format(patient_script=script)}],
            temperature=1.0,
            response_format=dialogue_json_schema,
            model=model_id
        )
        
        # Get content from the response
        content = message_dialog.choices[0].message.content
        
        # Check if content is already a dict or if it needs parsing
        if isinstance(content, str):
            try:
                # Try to parse it as JSON
                parsed_content = json.loads(content)
                return parsed_content
            except json.JSONDecodeError:
                # If it can't be parsed, wrap it in a proper structure
                logger.warning("Received non-JSON response from model, attempting to format it")
                return {"dialogue": [{"role": "doctor", "content": content}]}
        else:
            # Content is already a dictionary
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
        patient_script = api_call_with_retry(
            client,
            messages=[{'role': 'user', 'content': prompt_script}],
            temperature=1.1,
            response_format=patient_script_json_schema,
            model=model_id
        )
        # With vLLM, the content is already parsed as JSON
        return patient_script.choices[0].message.content
    except Exception as e:
        logger.error(f"Error generating patient script: {str(e)}")
        raise

def save_batch(batch, output_file):
    """Save a batch of dialogues to file with error handling"""
    try:
        with open(output_file, 'a') as f:
            for item in batch:
                f.write(json.dumps(item) + '\n')
        logger.info(f"Saved batch of {len(batch)} samples")
    except Exception as e:
        logger.error(f"Error saving batch: {str(e)}")
        # Save to backup file
        backup_file = output_file + '.backup'
        logger.warning(f"Attempting to save to backup file: {backup_file}")
        with open(backup_file, 'a') as f:
            for item in batch:
                f.write(json.dumps(item) + '\n')
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=20500, help="Number of dataset samples to generate")
    parser.add_argument("--dataset_output", type=str, default='dialogo_medico-paciente_es.jsonl', help="Path to JSON output")
    parser.add_argument("--config", type=str, default='config.yaml', help="Path to config file")
    parser.add_argument("--log_level", type=str, default="INFO", 
                      choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                      help="Set the logging level")
    
    args = parser.parse_args()
    
    # Set logging level from command line
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Load config
    config = Config(args.config)
    logger.info(f"Loaded configuration from: {args.config}")
    
    client = OpenAI(
        base_url=config.get('llm.base_url'),
        api_key=config.get('llm.api_key'),
        timeout=config.get('llm.timeout', 70000)
    )
    
    # Get available model from the API
    try:
        models_response = client.models.list()
        model_id = models_response.data[0].id
        logger.info(f"Using model: {model_id}")
    except Exception as e:
        logger.error(f"Error getting models: {str(e)}")
        raise SystemExit("Failed to get model ID from API")
        
    # Create output directory if needed
    output_path = Path(args.dataset_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate sample cases
    logger.info(f"Generating {args.n_samples} dialogue samples")
    cases = [(random.choice(argentinian_names), random.choice(illneses)) 
            for _ in range(args.n_samples)]
    
    current_batch = []
    batch_size = config.get('generation.batch_size', 10)
    pbar = tqdm(total=args.n_samples, desc="Generating samples")
    
    try:
        num_workers = config.get('generation.num_workers', mp.cpu_count())
        logger.info(f"Using {num_workers} workers for parallel generation")
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for sample in executor.map(lambda x: generate_sample(client, x[0], x[1], model_id), cases):
                if sample:
                    current_batch.append(sample)
                    
                    if len(current_batch) >= batch_size:
                        save_batch(current_batch, args.dataset_output)
                        current_batch = []
                        
                    pbar.update(1)
                    
    except KeyboardInterrupt:
        logger.warning("Generation interrupted by user")
    finally:
        if current_batch:
            save_batch(current_batch, args.dataset_output)
        pbar.close()
        logger.info(f"Generation complete. Results saved to {args.dataset_output}")