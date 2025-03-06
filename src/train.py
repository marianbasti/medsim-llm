#!/usr/bin/env python
"""
Training script for fine-tuning Llama-3.2-1B model on doctor-patient dialogues.
The goal is to train the model to roleplay as a patient given a patient script.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass

import torch
import numpy as np
from datasets import Dataset, load_dataset
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import wandb
import yaml

from config import load_config

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune."""
    
    model_name_or_path: str = "meta-llama/Llama-3.2-1B"
    use_flash_attention: bool = True
    use_4bit: bool = True
    use_nested_quant: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    trust_remote_code: bool = False


@dataclass
class DataArguments:
    """Arguments pertaining to what data we are going to input our model for training."""
    
    train_file: str = None
    validation_file: Optional[str] = None
    max_seq_length: int = 4096
    validation_split_percentage: int = 10
    preprocessing_num_workers: Optional[int] = None


@dataclass
class PeftArguments:
    """Arguments for PEFT (Parameter-Efficient Fine-Tuning)."""
    
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_r: int = 64
    target_modules: str = "all-linear"
    bias: str = "none"
    use_peft: bool = True


def create_patient_dataset(
    data_path: str, 
    tokenizer,
    max_seq_length: int = 4096,
    validation_split: float = 0.1,
    seed: int = 42
) -> Tuple[Dataset, Optional[Dataset]]:
    """
    Load and process the doctor-patient dialogue dataset.
    
    Returns:
        Tuple containing the training dataset and optionally validation dataset
    """
    logger.info(f"Loading data from {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        if data_path.endswith('.jsonl'):
            data = [json.loads(line) for line in f]
        elif data_path.endswith('.json'):
            data = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
    
    # Prepare the conversation format for training
    processed_samples = []
    
    for sample in data:
        script = sample["script"]
        dialogue = sample["dialogue"]
        
        # Extract only the patient responses
        patient_turns = [turn["content"] for turn in dialogue if turn["role"] == "patient"]
        
        if len(patient_turns) == 0:
            continue
        
        # Create patient roleplay prompt from script
        prompt = f"You are roleplaying as a patient visiting a doctor. Here are your details:\n{script}\n\nYou must stay in character as this patient and respond to the doctor's questions in a realistic way based on your patient details.\n\nDoctor: "
        
        for i, (doc_turn, patient_turn) in enumerate(zip(
            [turn["content"] for turn in dialogue if turn["role"] == "doctor"],
            patient_turns
        )):
            if i == 0:
                # First turn: use the prompt + doctor's question
                input_text = f"{prompt}{doc_turn}\nPatient: "
            else:
                # Subsequent turns: use conversation history
                history = ""
                for j in range(i):
                    history += f"\nDoctor: {dialogue[j*2]['content']}\nPatient: {dialogue[j*2+1]['content']}"
                input_text = f"{prompt}{history}\nDoctor: {doc_turn}\nPatient: "
            
            # Create sample with input_text and target_text
            processed_samples.append({
                "input": input_text,
                "output": patient_turn,
                "instruction": f"Respond as the patient described in the profile to this doctor's question: {doc_turn}"
            })
    
    # Convert to Dataset
    dataset = Dataset.from_list(processed_samples)
    
    # Tokenize function
    def tokenize_function(examples):
        # Tokenize inputs
        tokenized_inputs = tokenizer(
            examples["input"],
            truncation=True,
            max_length=max_seq_length - 512,  # Reserve space for output
            padding=False,
            return_tensors=None,
        )
        
        # Tokenize outputs
        tokenized_outputs = tokenizer(
            examples["output"],
            truncation=True,
            max_length=512,  # Max output length
            padding=False,
            return_tensors=None,
        )
        
        # Create labels, copying input_ids but setting them to -100 (ignored in loss)
        input_ids = tokenized_inputs["input_ids"]
        output_ids = tokenized_outputs["input_ids"]
        
        # Combine input and output, with labels being -100 for input
        combined_input_ids = []
        combined_labels = []
        
        for input_seq, output_seq in zip(input_ids, output_ids):
            combined_seq = input_seq + output_seq
            # Labels: -100 for input, actual IDs for output
            labels = [-100] * len(input_seq) + output_seq
            
            # Truncate if needed
            if len(combined_seq) > max_seq_length:
                combined_seq = combined_seq[:max_seq_length]
                labels = labels[:max_seq_length]
                
            combined_input_ids.append(combined_seq)
            combined_labels.append(labels)
        
        return {
            "input_ids": combined_input_ids,
            "labels": combined_labels,
            "attention_mask": [[1] * len(seq) for seq in combined_input_ids],
        }
    
    # Tokenize the dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset",
    )
    
    # Split into train and validation datasets
    if validation_split > 0:
        tokenized_dataset = tokenized_dataset.train_test_split(
            test_size=validation_split, seed=seed
        )
        train_dataset = tokenized_dataset["train"]
        eval_dataset = tokenized_dataset["test"]
    else:
        train_dataset = tokenized_dataset
        eval_dataset = None
    
    logger.info(f"Created train dataset with {len(train_dataset)} samples")
    if eval_dataset:
        logger.info(f"Created validation dataset with {len(eval_dataset)} samples")
    
    return train_dataset, eval_dataset


def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, PeftArguments, TrainingArguments))
    model_args, data_args, peft_args, training_args = parser.parse_args_into_dataclasses()
    
    # Load config
    config = load_config()
    
    # Set seed
    set_seed(training_args.seed)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="right",  # For generation tasks, pad on right
        trust_remote_code=model_args.trust_remote_code,
    )
    
    # Ensure we have EOS token
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model with quantization if specified
    if model_args.use_4bit:
        compute_dtype = getattr(torch, peft_args.bnb_4bit_compute_dtype)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=model_args.use_4bit,
            bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=model_args.use_nested_quant,
        )
    else:
        bnb_config = None
    
    # Additional keyword arguments
    kwargs = {
        "device_map": "auto",
    }
    
    if model_args.use_flash_attention:
        kwargs["attn_implementation"] = "flash_attention_2"
    
    # Load base model
    logger.info(f"Loading base model {model_args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=bnb_config,
        trust_remote_code=model_args.trust_remote_code,
        **kwargs
    )
    
    # Prepare the model for training with PEFT if specified
    if peft_args.use_peft:
        logger.info("Preparing model for PEFT training")
        model = prepare_model_for_kbit_training(model)
        
        # Parse target modules
        if peft_args.target_modules == "all-linear":
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        else:
            target_modules = peft_args.target_modules.split(",") if peft_args.target_modules else None
        
        # Configure PEFT (LoRA)
        logger.info(f"Setting up LoRA with r={peft_args.lora_r}, alpha={peft_args.lora_alpha}")
        peft_config = LoraConfig(
            r=peft_args.lora_r,
            lora_alpha=peft_args.lora_alpha,
            lora_dropout=peft_args.lora_dropout,
            target_modules=target_modules,
            bias=peft_args.bias,
            task_type="CAUSAL_LM",
        )
        
        # Get PEFT model
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    # Create the dataset
    logger.info("Creating dataset")
    train_dataset, eval_dataset = create_patient_dataset(
        data_args.train_file, 
        tokenizer=tokenizer,
        max_seq_length=data_args.max_seq_length,
        validation_split=data_args.validation_split_percentage / 100,
        seed=training_args.seed
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    # Start training
    logger.info("Starting training")
    train_result = trainer.train()
    
    # Save model
    logger.info(f"Saving model to {training_args.output_dir}")
    trainer.save_model()
    
    # Save tokenizer
    tokenizer.save_pretrained(training_args.output_dir)
    
    # Log metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # Run evaluation if validation dataset exists
    if eval_dataset:
        logger.info("Running evaluation")
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()