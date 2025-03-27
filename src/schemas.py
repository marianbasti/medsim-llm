from pydantic import BaseModel, Field
from enum import Enum
from typing import List, Optional

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

# Spanish character mappings
SPANISH_CHARS = {
    '\\u00e1': 'á', 
    '\\u00e9': 'é', 
    '\\u00ed': 'í', 
    '\\u00f3': 'ó', 
    '\\u00fa': 'ú', 
    '\\u00f1': 'ñ', 
    '\\u00c1': 'Á', 
    '\\u00c9': 'É', 
    '\\u00cd': 'Í', 
    '\\u00d3': 'Ó', 
    '\\u00da': 'Ú', 
    '\\u00d1': 'Ñ', 
    '\\u00fc': 'ü', 
    '\\u00dc': 'Ü', 
    '\\u00bf': '¿', 
    '\\u00a1': '¡'  
}
