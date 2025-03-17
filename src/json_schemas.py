from typing import List, Optional
from pydantic import BaseModel, Field
from enum import Enum

# Pydantic models for dialogue JSON schema
class RoleType(str, Enum):
    doctor = "doctor"
    patient = "patient"

class DialogMessage(BaseModel):
    role: RoleType
    content: str

class Dialogue(BaseModel):
    dialogue: List[DialogMessage]

# Create JSON schema from the Pydantic model
dialogue_json_schema = Dialogue.model_json_schema()

# Patient script schema as a dictionary since it's more complex
# with conditional validation that's harder to express in Pydantic
patient_script_json_schema = {
    "type": "object",
    "properties": {
        "Name": {"type": "string"},
        "Demographics": {
            "type": "object",
            "properties": {
                "age": {"type": "integer"},
                "sex": {"type": "string"},
                "occupation": {"type": "string"},
                "education level": {"type": "string"}
            },
            "required": ["age", "sex", "occupation", "education level"]
        },
        "Medical History": {
            "type": "object",
            "properties": {
                "conditions": {"type": "array", "items": {"type": "string"}},
                "medications": {"type": "array", "items": {"type": "string"}},
                "allergies": {"type": "array", "items": {"type": "string"}},
                "surgical history": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["conditions", "medications", "allergies", "surgical history"]
        },
        "Current Symptoms": {
            "type": "object",
            "properties": {
                "chief complaint": {"type": "string"},
                "duration": {"type": "string"},
                "severity": {"type": "string"},
                "associated symptoms": {"type": "string"}
            },
            "required": ["chief complaint", "duration", "severity", "associated symptoms"]
        },
        "Personal Details": {
            "type": "object",
            "properties": {
                "lifestyle habits": {"type": "string"},
                "family dynamics": {"type": "string"},
                "work": {"type": "string"},
                "social history": {"type": "string"},
                "mental health history": {"type": "string"}
            },
            "required": ["lifestyle habits", "family dynamics", "work", "social history", "mental health history"]
        },
        "Behavioral and Cognitive Factors": {
            "type": "object",
            "properties": {
                "personality traits": {"type": "string"},
                "cognitive function": {"type": "string"},
                "behavioral patterns": {"type": "string"}
            },
            "required": ["personality traits", "cognitive function", "behavioral patterns"]
        },
        "Healthcare Utilization": {
            "type": "object",
            "properties": {
                "recent_hospitalizations": {"type": "boolean"},
                "recent_hospitalizations_cause": {"type": "string"},
                "emergency_room_visits": {"type": "boolean"},
                "emergency_room_visits_cause": {"type": "string"}
            },
            "required": ["recent_hospitalizations", "emergency_room_visits"]
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