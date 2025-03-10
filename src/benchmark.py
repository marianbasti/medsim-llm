#!/usr/bin/env python
"""
Benchmark script for evaluating the fine-tuned Llama-3.2-1B model on patient roleplay.
This script measures how well the model can roleplay as a patient given a script and doctor questions.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
import random
import time

import torch
import numpy as np
from datasets import Dataset
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    GenerationConfig,
    set_seed,
)
from peft import PeftModel
from tqdm import tqdm
import evaluate
import yaml
from sklearn.metrics.pairwise import cosine_similarity

from config import load_config

# Configure logging
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class EvalArguments:
    """Arguments for evaluation."""
    model_path: str = None
    base_model: str = "meta-llama/Llama-3.2-1B"
    test_file: str = None
    output_dir: str = "evaluation_results"
    batch_size: int = 4
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    num_beams: int = 1
    do_sample: bool = True
    use_4bit: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    num_samples: int = -1  # Use all if -1


class PatientRoleplayEvaluator:
    """Evaluates a model on patient roleplay capabilities."""
    
    def __init__(self, args: EvalArguments):
        self.args = args
        self.config = load_config()
        
        # Set seed for reproducibility
        set_seed(args.seed)
        
        # Load tokenizer
        logger.info(f"Loading tokenizer from {args.model_path or args.base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_path or args.base_model,
            padding_side="right",
        )
        
        if not self.tokenizer.pad_token_id:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load model
        model_source = args.model_path or args.base_model
        logger.info(f"Loading model from {model_source}")
        
        # Use quantization if specified
        if args.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            logger.info("Using 4-bit quantization for efficient inference")
        else:
            bnb_config = None
        
        # Load the base or fine-tuned model
        start_time = time.time()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_source,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        logger.info(f"Model loaded successfully in {time.time() - start_time:.2f} seconds")
        
        # Set up generation config
        self.generation_config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            num_beams=args.num_beams,
            do_sample=args.do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        
        # Load metrics
        logger.info("Loading evaluation metrics")
        self.rouge = evaluate.load("rouge")
        self.bertscore = evaluate.load("bertscore")
        self.sentence_transformer = transformers.AutoModel.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2", 
            trust_remote_code=True
        ).to(args.device)
    
    def load_test_data(self):
        """Load and process test data."""
        logger.info(f"Loading test data from {self.args.test_file}")
        
        with open(self.args.test_file, 'r', encoding='utf-8') as f:
            if self.args.test_file.endswith('.jsonl'):
                data = [json.loads(line) for line in f]
            elif self.args.test_file.endswith('.json'):
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {self.args.test_file}")
        
        # Sample a subset if required
        if self.args.num_samples > 0 and self.args.num_samples < len(data):
            data = random.sample(data, self.args.num_samples)
            logger.info(f"Sampled {len(data)} examples for evaluation")
        else:
            logger.info(f"Using all {len(data)} examples for evaluation")
        
        # Process the data
        test_samples = []
        for sample in data:
            script = sample["script"]
            dialogue = sample["dialogue"]
            
            # Extract doctor questions and patient responses
            doctor_turns = [turn["content"] for turn in dialogue if turn["role"] == "doctor"]
            patient_turns = [turn["content"] for turn in dialogue if turn["role"] == "patient"]
            
            if not doctor_turns or not patient_turns:
                continue
            
            # For each doctor-patient exchange, create a sample
            for i, (doc_turn, patient_turn) in enumerate(zip(doctor_turns, patient_turns)):
                if i == 0:
                    # First turn: Just use the prompt + doctor's first question
                    prompt = f"You are roleplaying as a patient visiting a doctor. Here are your details:\n{script}\n\nYou must stay in character as this patient and respond to the doctor's questions in a realistic way based on your patient details.\n\nDoctor: {doc_turn}\nPatient:"
                    history = None
                else:
                    # Use conversation history up to this point
                    history = []
                    for j in range(i):
                        history.append({
                            "role": "user",
                            "content": f"Doctor: {doctor_turns[j]}"
                        })
                        history.append({
                            "role": "assistant",
                            "content": f"Patient: {patient_turns[j]}"
                        })
                    
                    prompt = f"Doctor: {doc_turn}\nPatient:"
                
                test_samples.append({
                    "id": f"{len(test_samples)}",
                    "script": script,
                    "prompt": prompt,
                    "history": history,
                    "reference": patient_turn,
                    "doctor_turn": doc_turn,
                })
        
        logger.info(f"Created {len(test_samples)} test samples for evaluation")
        return test_samples
    
    def generate_responses(self, test_samples):
        """Generate responses for test samples."""
        logger.info(f"Generating responses for {len(test_samples)} test samples")
        
        results = []
        
        for i, sample in enumerate(tqdm(test_samples, desc="Generating responses")):
            # Prepare input
            if sample["history"] is None:
                # No history, just use the prompt
                inputs = self.tokenizer(sample["prompt"], return_tensors="pt").to(self.args.device)
            else:
                # Use history with prompt format
                messages = sample["history"] + [{
                    "role": "user", 
                    "content": f"Doctor: {sample['doctor_turn']}"
                }]
                
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.args.device)
            
            # Generate response
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    generation_config=self.generation_config,
                )
            
            # Decode the generated response
            generated_text = self.tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            
            # Save result
            results.append({
                "id": sample["id"],
                "script": sample["script"],
                "doctor_turn": sample["doctor_turn"],
                "generated": generated_text,
                "reference": sample["reference"],
                "history": sample["history"],
            })
            
            # Log progress occasionally
            if (i+1) % 10 == 0:
                logger.debug(f"Generated {i+1}/{len(test_samples)} responses")
        
        return results
    
    def evaluate_responses(self, results):
        """Evaluate the generated responses."""
        logger.info("Evaluating model responses")
        
        # Prepare for metrics calculation
        generated_texts = [result["generated"] for result in results]
        reference_texts = [result["reference"] for result in results]
        
        # Calculate ROUGE scores
        logger.info("Calculating ROUGE scores")
        rouge_scores = self.rouge.compute(
            predictions=generated_texts,
            references=reference_texts,
            use_aggregator=True,
        )
        
        # Calculate BERTScore (semantic similarity)
        logger.info("Calculating BERTScore (semantic similarity)")
        bert_scores = self.bertscore.compute(
            predictions=generated_texts,
            references=reference_texts,
            lang="es",  # Most dialogues appear to be in Spanish
            model_type="distilbert-base-multilingual-cased",
        )
        
        # Calculate average BERTScore
        avg_bert_precision = sum(bert_scores["precision"]) / len(bert_scores["precision"])
        avg_bert_recall = sum(bert_scores["recall"]) / len(bert_scores["recall"])
        avg_bert_f1 = sum(bert_scores["f1"]) / len(bert_scores["f1"])
        
        # Aggregate metrics
        metrics = {
            "rouge1": rouge_scores["rouge1"],
            "rouge2": rouge_scores["rouge2"],
            "rougeL": rouge_scores["rougeL"],
            "rougeLsum": rouge_scores["rougeLsum"],
            "bert_precision": avg_bert_precision,
            "bert_recall": avg_bert_recall,
            "bert_f1": avg_bert_f1,
        }
        
        logger.info(f"ROUGE-1: {metrics['rouge1']:.4f}, ROUGE-L: {metrics['rougeL']:.4f}, BERTScore F1: {avg_bert_f1:.4f}")
        
        return metrics, results
    
    def calculate_role_consistency(self, results):
        """Calculate how consistently the model stays in its patient role."""
        logger.info("Calculating role consistency")
        
        consistency_scores = []
        
        for result in results:
            # Check if the output is in proper format (patient response)
            is_proper_patient = 1.0  # Assume proper format by default
            
            # Check if the model started speaking as the doctor or gave instructions
            lower_text = result["generated"].lower()
            if "doctor:" in lower_text or "doctor's" in lower_text or "as the doctor" in lower_text:
                is_proper_patient = 0.0
            
            # Check for other role violations or formatting issues
            if "you should" in lower_text or "you need to" in lower_text or "you must" in lower_text:
                is_proper_patient = 0.0
            
            consistency_scores.append(is_proper_patient)
        
        avg_consistency = sum(consistency_scores) / len(consistency_scores)
        logger.info(f"Role consistency score: {avg_consistency:.4f}")
        
        # Count examples that maintained proper role
        consistent_count = sum(1 for score in consistency_scores if score > 0.5)
        total_count = len(consistency_scores)
        logger.info(f"{consistent_count} out of {total_count} examples maintained proper patient role ({(consistent_count/total_count)*100:.1f}%)")
        
        return avg_consistency, consistency_scores
    
    def save_results(self, metrics, results, role_consistency_score, consistency_scores):
        """Save the evaluation results."""
        
        # Create output directory
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        metrics["role_consistency"] = role_consistency_score
        
        with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        
        # Add consistency scores to results
        for result, score in zip(results, consistency_scores):
            result["role_consistency_score"] = score
        
        # Save detailed results
        with open(output_dir / "generation_results.jsonl", "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        
        # Save some examples
        with open(output_dir / "examples.txt", "w", encoding="utf-8") as f:
            for i, result in enumerate(results[:10]):  # Just first 10 examples
                f.write(f"Example {i+1}:\n")
                f.write(f"Doctor: {result['doctor_turn']}\n")
                f.write(f"Reference: {result['reference']}\n")
                f.write(f"Generated: {result['generated']}\n")
                f.write(f"Role Consistency: {result['role_consistency_score']}\n")
                f.write("-" * 80 + "\n\n")
        
        logger.info(f"Results saved to {output_dir}")
    
    def run_evaluation(self):
        """Run the full evaluation pipeline."""
        start_time = time.time()
        
        # Load test data
        test_samples = self.load_test_data()
        
        # Generate responses
        results = self.generate_responses(test_samples)
        
        # Evaluate responses
        metrics, results = self.evaluate_responses(results)
        
        # Calculate role consistency
        role_consistency_score, consistency_scores = self.calculate_role_consistency(results)
        
        # Save results
        self.save_results(metrics, results, role_consistency_score, consistency_scores)
        
        # Log summary metrics
        logger.info(f"Evaluation completed in {time.time() - start_time:.2f} seconds")
        logger.info(f"Role consistency: {role_consistency_score:.4f}")
        logger.info(f"ROUGE-1: {metrics['rouge1']:.4f}")
        logger.info(f"ROUGE-2: {metrics['rouge2']:.4f}")
        logger.info(f"ROUGE-L: {metrics['rougeL']:.4f}")
        logger.info(f"BERTScore F1: {metrics['bert_f1']:.4f}")


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Evaluate patient roleplay model")
    parser.add_argument("--model_path", type=str, help="Path to the fine-tuned model")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.2-1B", help="Base model name")
    parser.add_argument("--test_file", type=str, required=True, help="Path to test data file")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for generation")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum number of new tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling top-p")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--num_beams", type=int, default=1, help="Beam search size")
    parser.add_argument("--do_sample", action="store_true", help="Use sampling for generation")
    parser.add_argument("--use_4bit", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_samples", type=int, default=-1, help="Number of samples to evaluate (-1 for all)")
    parser.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="Set logging level")
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    logger.info(f"Starting evaluation with log level: {args.log_level}")
    
    # Convert namespace to EvalArguments
    eval_args = EvalArguments(**vars(args))
    
    # Run evaluation
    evaluator = PatientRoleplayEvaluator(eval_args)
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()