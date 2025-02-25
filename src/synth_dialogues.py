import json, random
import re
import argparse

parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument("--n_samples", type=int, default=20500, help="Number of dataset samples to generate")
parser.add_argument("--dataset_output", type=str, default='dialogo_medico-paciente_es_Llama-3.1-8B-Q8.json', help="Path to JSON output")
parser.add_argument("--base_url", type=str, default='http://localhost:7000/v1/', help="Base URL for API requests")

# Parse arguments
args = parser.parse_args()


### DIALOG PROMPT AND SCHEMA
prompt_dialog_correct = """Generate a natural conversation in Argentinian Spanish between a doctor and a patient based on the following patient script: {patient_script}. The patient should not reveal all the information at once, and the conversation should reflect the doctor's active listening, clarity, respect, reassurance, patient-centered care, clear communication, encouragement, confidentiality, and effective time management. The doctor should NOT perform any actions, just talk.
Only provide the spoken dialogue, without any additional explanation, performance or translation.
"""

prompt_dialog_incorrect = """Generate a natural conversation in Argentinian Spanish between a doctor and a patient based on the following patient script: {patient_script}. The patient should not reveal all the information at once. The conversation should reflect the doctor's ineptitude in communication, including some of these behaviors:
Using excessive medical jargon without explanation, showing impatience or rushing the conversation, failing to address the patient's concerns, minimizing the patient's symptoms, lacking empathy or emotional support, jumping to conclusions without gathering all necessary information

The doctor should NOT perform any actions, just talk.
Only provide the spoken dialogue, without any additional explanation, performance or translation.

"""

dialogue_json_schema = {
    "type": "json_object",
    "schema": {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "dialogue": {
                "type": "array",
                "items": {
                    "$ref": "#/definitions/message"
                }
            }
        },
        "definitions": {
            "message": {
                "type": "object",
                "properties": {
                    "role": {
                        "type": "string",
                        "enum": ["doctor", "patient"]
                    },
                    "content": {"type": "string"}
                },
                "required": ["role", "content"]
            }
        },
        "required": ["dialogue"]
    }
}

### PATIENT CARD PROMPT AND JSON SCHEMA
prompt_script_2 = """Generate a patient card for {name} that has {illness}. Their symptoms are: {symptoms}.
The patient card must have with the following features: demographics, medical history, current symptoms, personal details, behavioral and cognitive factors, and healthcare utilization. Ensure that the patient card is realistic and diverse in terms of age, sex, occupation, and medical conditions. Use the following JSON format:

{{
"Name": {name},
"Demographics": {{"age", "sex", "occupation", "education level"}},
"Medical History": {{"conditions":["list of previous medical conditions"], "medications":["list of medications"], "allergies":["list of allergies"], "surgical history":["list of surgical procedures"]}},
"Current Symptoms": "chief complaint, duration, severity, associated symptoms",
"Personal Details": "lifestyle habits, family dynamics, work, social history, mental health history",
"Behavioral and Cognitive Factors": "personality traits, cognitive function, behavioral patterns",
"Healthcare Utilization": "recent hospitalizations or emergency room visits"
}}

Only provide the card for {name} in your response. Use natural language. The features should be realistic and coherent. Let's have a complex and low-income social context, in a country with free healthcare. The patient's dialogue must be short and simple."""

patient_card_json_schema = {
    "type": "json_object",
    "properties": {
        "Name": {
        "type": "string"
        },
        "Demographics": {
        "type": "object",
        "properties": {
            "age": { "type": "integer" },
            "sex": { "type": "string" },
            "occupation": { "type": "string" },
            "education level": { "type": "string" }
        },
        "required": ["age", "sex", "occupation", "education level"]
        },
        "Medical History": {
        "type": "object",
        "properties": {
            "conditions": {
            "type": "array",
            "items": { "type": "string" }
            },
            "medications": {
            "type": "array",
            "items": { "type": "string" }
            },
            "allergies": {
            "type": "array",
            "items": { "type": "string" }
            },
            "surgical history": {
            "type": "array",
            "items": { "type": "string" }
            }
        },
        "required": ["conditions", "medications", "surgical history"]
        },
        "Current Symptoms": {
        "type": "object",
        "properties": {
            "chief complaint": { "type": "string" },
            "duration": { "type": "string" },
            "severity": { "type": "string" },
            "associated symptoms": { "type": "string" }
        },
        "required": ["chief complaint", "duration", "severity", "associated symptoms"]
        },
        "Personal Details": {
        "type": "object",
        "properties": {
            "lifestyle habits": { "type": "string" },
            "family dynamics": { "type": "string" },
            "work": { "type": "string" },
            "social history": { "type": "string" },
            "mental health history": { "type": "string" }
        },
        "required": ["lifestyle habits", "family dynamics", "work", "social history", "mental health history"]
        },
        "Behavioral and Cognitive Factors": {
        "type": "object",
        "properties": {
            "personality traits": { "type": "string" },
            "cognitive function": { "type": "string" },
            "behavioral patterns": { "type": "string" }
        },
        "required": ["personality traits", "cognitive function", "behavioral patterns"]
        },
        "Healthcare Utilization": {
        "type": "object",
        "properties": {
            "recent hospitalizations": { "type": "string" },
            "emergency room visits": { "type": "string" }
        },
        "required": ["recent hospitalizations", "emergency room visits"]
        }
    },
    "required": [
        "Name", 
        "Demographics", 
        "Medical History", 
        "Current Symptoms", 
        "Personal Details", 
        "Behavioral and Cognitive Factors", 
        "Healthcare Utilization"
    ]
}

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

# Transform the data to train with MS-Swift
# LEGACY
#def sharegpt_training_format(input_file, output_file):
#    result = []#
#
#    with open(input_file, 'r', encoding='utf-8') as f:
#        for line_number, line in enumerate(f, 1):
#            conversation = json.loads(line)
#
#            formatted_conversation = [
#                {
#                    "from": "system",
#                    "value": conversation['script']
#                }
#            ]
#            
#            dialogue = conversation['dialogue'].replace("**", "")
#            dialogue = re.sub(r'\b(doctor|Doctor)\b\n', r'user\n', dialogue)
#            dialogue = re.sub(r'\b(patient|paciente|Patient|Paciente)\b\n', r'assistant\n', dialogue)
#            turns = re.split(r'<start_of_turn>|<end_of_turn>', dialogue)
#            
#            last_speaker = None
#            for turn in turns:
#                turn = turn.strip()
#                if turn:
#                    parts = turn.split('\n', 1)
#                    if len(parts) == 2:
#                        speaker, message = parts
#                    else:
#                        speaker = formatted_conversation[-1]['from'] if formatted_conversation else 'unknown'
#                        message = parts[0]
#                    
#                    speaker = speaker.strip()
#                    message = message.strip()
#                    
#                    if formatted_conversation and formatted_conversation[-1]['from'] == speaker:
#                        formatted_conversation[-1]['value'] += " " + message
#                    else:
#                        formatted_conversation.append({
#                            "from": speaker,
#                           "value": message
#                        })
#            
#            # Remove the last message if it's from the user
#            if formatted_conversation[-1]['from'].lower() == 'user':
#                formatted_conversation.pop()
#           
#            # Ensure the last message is from the assistant
#            if formatted_conversation[-1]['from'].lower() != 'assistant':
#                formatted_conversation.pop()
#            
#            # Only add the conversation if it has at least one message after the system message
#            if len(formatted_conversation) > 1:
#                # Generate a unique ID for this conversation
#                unique_id = f"conv_{line_number}"
#                
#                # Add this conversation to the result with its unique ID
#                result.append({
#                    "id": unique_id,
#                    "conversations": formatted_conversation
#                })
#    
#    with open(output_file, 'w', encoding='utf-8') as f:
#       json.dump(result, f, ensure_ascii=False, indent=2)
#
#    print(f"Conversion complete. Output saved to {output_file}")

def dataset2sharegpt(input_file, output_file):
    all_conversations = {"conversations": []}
    
    with open(input_file, 'r', encoding='utf-8') as f_in:
        for line in f_in:
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
        
    # Write the final aggregated data to the output JSON file
    with open(output_file, 'w', encoding='utf-8') as f_out:
        json.dump(all_conversations, f_out, ensure_ascii=False, indent=4)

def synth_dialogue(client, prompt_dialog, script):
    """
    Synthesize a patient-doctor dialogue.

    Args:
        client: The OpenAI API client.
        prompt_dialog (str): The prompt template to use for generating a dialog.
        prompt_script (str): The prompt template to use for creating a patient script.
    
    Returns:
        dict: The synthesized patient-doctor dialogue.
    """
    message_dialog = client.chat.completions.create(
        messages=[
            {
                'role': 'user',
                'content': prompt_dialog.format(patient_script=script)
            }
        ],
        temperature=1.0,
        response_format = dialogue_json_schema,
        model=""
    )

    generated_dialog=message_dialog.choices[0].message.content

    return generated_dialog

def synth_script(client, prompt_script):
    """
    Synthesize a dataset of patient scripts.

    Args:
        client: The OpenAI API client.
        prompt_script (str): The prompt template to use for creating a patient script.
    
    Returns:
        str: The synthesized patient script.
    """
    patient_script = client.chat.completions.create(
        messages=[
            {
                'role': 'user',
                'content': prompt_script
            }
        ],
        temperature=1.1,
        response_format=prompt_script,
        model=""
    )

    return patient_script.choices[0].message.content

if __name__ == "__main__":
    from openai import OpenAI
    
    client = OpenAI(
        base_url=args.base_url,
        api_key='llamacpp',
        timeout=70000
    )
    
    for i in range(0, args.n_samples):
        index = i % len(argentinian_names)
        name = random.choice(argentinian_names)
        illness = random.choice(illneses)
        print(f"Generating script for {name} with {illness[0]}")
        script = synth_script(client, prompt_script_2.format(name=name, illness=illness[0],symptoms=illness[1]))
        print(script)
        print(f"Generating good dialogue...")
        good_dialogue = synth_dialogue(client, prompt_dialog_correct, script)
        print(f"Generating inept dialogue...")
        inept_dialogue = synth_dialogue(client, prompt_dialog_incorrect, script)
        print(good_dialogue)
        print(inept_dialogue)

        with open(args.dataset_output, 'a') as f:
            f.write(json.dumps({"script": script, "dialogue": json.loads(good_dialogue)['dialogue']}))
            f.write('\n')
            f.write(json.dumps({"script": script, "dialogue": json.loads(inept_dialogue)['dialogue']}))
            f.write('\n')