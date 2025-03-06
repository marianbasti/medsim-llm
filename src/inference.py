#!/usr/bin/env python
"""
Inference script for using the fine-tuned Llama-3.2-1B model to roleplay as a patient.
Provide a patient script and doctor's questions to get patient responses.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import time
from dataclasses import dataclass

import torch
import yaml
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)
from peft import PeftModel

from config import load_config
from validation import PatientResponseValidator

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class InferenceArguments:
    """Arguments for model inference."""
    model_path: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_4bit: bool = True
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    use_validation: bool = True


class PatientRoleplayModel:
    """Model for roleplaying as a patient based on a patient script."""
    
    def __init__(self, args: InferenceArguments):
        """Initialize the model."""
        self.args = args
        self.config = load_config()
        
        # Load tokenizer
        logger.info(f"Loading tokenizer from {args.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            padding_side="right",
            trust_remote_code=True,
        )
        
        if not self.tokenizer.pad_token_id:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load model with quantization if specified
        if args.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            bnb_config = None
        
        logger.info(f"Loading model from {args.model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        
        # Set up generation config
        self.generation_config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            do_sample=args.do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        
        # Initialize validator if validation is enabled
        if args.use_validation:
            self.validator = PatientResponseValidator()
        else:
            self.validator = None
    
    def generate_response(self, 
                         patient_script: Union[str, Dict], 
                         doctor_question: str, 
                         conversation_history: Optional[List[Dict]] = None) -> str:
        """
        Generate a patient response to a doctor's question.
        
        Args:
            patient_script: Patient script with demographics and medical history
            doctor_question: Question from the doctor
            conversation_history: Optional previous conversation history
            
        Returns:
            Patient response string
        """
        # Format patient script if needed
        if isinstance(patient_script, dict):
            script_str = json.dumps(patient_script, ensure_ascii=False)
        else:
            script_str = patient_script
        
        if conversation_history is None or len(conversation_history) == 0:
            # First turn, use the script to build the prompt
            prompt = f"You are roleplaying as a patient visiting a doctor. Here are your details:\n{script_str}\n\nYou must stay in character as this patient and respond to the doctor's questions in a realistic way based on your patient details.\n\nDoctor: {doctor_question}\nPatient:"
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.args.device)
        else:
            # Use conversation history with system prompt
            messages = [
                {
                    "role": "system",
                    "content": f"You are roleplaying as a patient visiting a doctor. Here are your details:\n{script_str}\n\nYou must stay in character as this patient and respond to the doctor's questions in a realistic way based on your patient details."
                }
            ]
            
            # Add conversation history
            for turn in conversation_history:
                role = turn["role"].lower()
                content = turn["content"]
                
                if role == "doctor":
                    messages.append({
                        "role": "user",
                        "content": f"Doctor: {content}"
                    })
                elif role == "patient":
                    messages.append({
                        "role": "assistant",
                        "content": f"Patient: {content}"
                    })
            
            # Add current doctor question
            messages.append({
                "role": "user",
                "content": f"Doctor: {doctor_question}"
            })
            
            # Format as chat
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.args.device)
        
        # Generate response
        start_time = time.time()
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                generation_config=self.generation_config,
            )
        end_time = time.time()
        
        # Decode the generated response
        generated_text = self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        # Clean up response if needed
        generated_text = generated_text.strip()
        if generated_text.startswith("Patient:"):
            generated_text = generated_text[8:].strip()
        
        # Validate response if validator is available
        if self.validator:
            validation_result = self.validator.validate_response(
                response=generated_text,
                doctor_question=doctor_question,
                script=patient_script,
                dialogue_history=conversation_history
            )
            
            logger.info(f"Response validation: {validation_result}")
        
        logger.info(f"Generated response in {end_time - start_time:.2f}s")
        return generated_text
    
    def interactive_session(self, patient_script: Union[str, Dict]):
        """
        Start an interactive doctor-patient session.
        
        Args:
            patient_script: Patient script with demographics and medical history
        """
        print("Starting interactive patient simulation. Type 'exit' to end the session.")
        print("You are the doctor, the model will roleplay as the patient.")
        print("\nPatient Script:")
        if isinstance(patient_script, dict):
            print(json.dumps(patient_script, indent=2, ensure_ascii=False))
        else:
            print(patient_script)
        
        conversation_history = []
        
        while True:
            # Get doctor input
            doctor_input = input("\nDoctor: ")
            if doctor_input.lower() in ["exit", "quit", "bye"]:
                print("\nEnding session. Goodbye!")
                break
            
            # Generate patient response
            patient_response = self.generate_response(
                patient_script=patient_script,
                doctor_question=doctor_input,
                conversation_history=conversation_history
            )
            
            # Add to conversation history
            conversation_history.append({"role": "doctor", "content": doctor_input})
            conversation_history.append({"role": "patient", "content": patient_response})
            
            # Show patient response
            print(f"\nPatient: {patient_response}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Generate patient responses')
    parser.add_argument('--model_path', type=str, required=True, help='Path to fine-tuned model')
    parser.add_argument('--script_file', type=str, help='Path to patient script file')
    parser.add_argument('--script_json', type=str, help='Patient script as JSON string')
    parser.add_argument('--question', type=str, help='Doctor question')
    parser.add_argument('--interactive', action='store_true', help='Start interactive session')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    parser.add_argument('--max_new_tokens', type=int, default=256, help='Maximum new tokens')
    parser.add_argument('--use_4bit', action='store_true', help='Use 4-bit quantization')
    parser.add_argument('--no_validation', action='store_true', help='Disable response validation')
    args = parser.parse_args()
    
    # Convert arguments to InferenceArguments
    inference_args = InferenceArguments(
        model_path=args.model_path,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        use_4bit=args.use_4bit,
        use_validation=not args.no_validation,
    )
    
    # Initialize model
    model = PatientRoleplayModel(inference_args)
    
    # Get patient script
    if args.script_json:
        patient_script = json.loads(args.script_json)
    elif args.script_file:
        with open(args.script_file, 'r', encoding='utf-8') as f:
            if args.script_file.endswith('.json'):
                patient_script = json.load(f)
            else:
                patient_script = f.read().strip()
    else:
        # Use a default script for testing
        patient_script = {
            "Name": "Juan Pérez",
            "Demographics": {
                "age": 38,
                "sex": "Male",
                "occupation": "Construction Worker",
                "education level": "High School Diploma"
            },
            "Medical History": {
                "conditions": ["Hypertension", "Asthma (mild)", "Gastritis"],
                "medications": ["Enalapril (20mg)", "Salbutamol (inhaler)"],
                "allergies": ["Penicillin"],
                "surgical history": ["Appendectomy (2005)"]
            },
            "Current Symptoms": {
                "chief complaint": "Common Cold",
                "duration": "3 days",
                "severity": "Mild",
                "associated symptoms": "Runny nose with clear discharge, scratchy throat, dry cough, slight congestion"
            }
        }
    
    # Interactive or single question mode
    if args.interactive:
        model.interactive_session(patient_script)
    elif args.question:
        response = model.generate_response(
            patient_script=patient_script,
            doctor_question=args.question
        )
        print(f"Doctor: {args.question}")
        print(f"Patient: {response}")
    else:
        print("Either --interactive or --question must be specified.")


if __name__ == "__main__":
    main()