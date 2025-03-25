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
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import wandb
import yaml

from config import load_config

# Configure logging with more informative format
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
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
    
    raw_data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        if data_path.endswith('.jsonl'):
            for line in f:
                try:
                    raw_data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Error decoding JSON line: {e}")
                    continue
        elif data_path.endswith('.json'):
            raw_data = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
    
    # Prepare the conversation format for training
    processed_samples = []
    
    logger.info(f"Processing {len(raw_data)} dialogue samples")
    for sample in raw_data:
        if 'script' not in sample or 'dialogue' not in sample:
            logger.warning("Sample is missing required fields 'script' or 'dialogue'. Skipping.")
            continue
            
        script = sample["script"]
        dialogue = sample.get("dialogue", [])
        
        # Check if the dialogue contains doctor-patient conversation
        doctor_turns = [turn for turn in dialogue if turn.get("role") == "doctor"]
        if not doctor_turns:
            logger.warning("No doctor turns found in dialogue. Skipping sample.")
            continue
            
        # First doctor's message to start the conversation
        doctor_message = doctor_turns[0]["content"]
        
        # Create a prompt with instructions
        prompt = f"You are roleplaying as a patient visiting a doctor. Follow the instructions below to stay in character.\n\n"
        prompt += f"Patient Profile:\n{script}\n\n"
        prompt += f"Respond to the doctor's question as the patient described above. Be concise and realistic.\n\n"
        prompt += f"Doctor: {doctor_message}\n\nPatient:"
        
        # Get the patient's response (if any)
        patient_response = ""
        for i, turn in enumerate(dialogue):
            if i > 0 and turn.get("role") == "patient":
                patient_response = turn["content"]
                break
        
        if not patient_response:
            logger.warning("No patient response found. Skipping sample.")
            continue
            
        # Create the sample
        processed_samples.append({
            "prompt": prompt,
            "response": patient_response
        })
    
    if not processed_samples:
        raise ValueError("No valid samples found in the dataset after processing!")
    
    # Convert to Dataset
    dataset = Dataset.from_list(processed_samples)
    logger.info(f"Created dataset with {len(dataset)} training examples")
    
    # Tokenize function
    def tokenize_function(examples):
        # Tokenize prompts and responses
        prompts = examples["prompt"]
        responses = examples["response"]
        
        # Join them for the complete sequence
        texts = [prompt + response for prompt, response in zip(prompts, responses)]
        
        # Calculate prompt lengths for later use
        prompt_lengths = [len(tokenizer(prompt, add_special_tokens=False).input_ids) for prompt in prompts]
        
        # Tokenize the full sequences with padding and truncation
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",  # Changed to max_length to ensure all examples have the same length
            return_tensors=None,
        )
        
        # Create label mask: -100 for prompt tokens, actual ids for response tokens
        labels = []
        for i, input_ids in enumerate(tokenized["input_ids"]):
            # Set prompt part to -100 (ignored in loss)
            prompt_len = min(prompt_lengths[i], max_seq_length)
            label = [-100] * prompt_len + input_ids[prompt_len:]
            # Ensure the same length as input_ids by padding with -100
            if len(label) < max_seq_length:
                label.extend([-100] * (max_seq_length - len(label)))
            # Truncate if too long
            label = label[:max_seq_length]
            labels.append(label)
        
        tokenized["labels"] = labels
        return tokenized
    
    # Tokenize the dataset
    logger.info("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset",
        num_proc=os.cpu_count() // 2,  # Use half the available CPUs
    )
    
    # Split into train and validation datasets
    if validation_split > 0:
        logger.info(f"Splitting dataset with validation ratio: {validation_split}")
        tokenized_dataset = tokenized_dataset.train_test_split(
            test_size=validation_split, seed=seed
        )
        train_dataset = tokenized_dataset["train"]
        eval_dataset = tokenized_dataset["test"]
        logger.info(f"Split complete - Training: {len(train_dataset)} examples, Validation: {len(eval_dataset)} examples")
    else:
        train_dataset = tokenized_dataset
        eval_dataset = None
        logger.info(f"No validation split - Using all {len(train_dataset)} examples for training")
    
    return train_dataset, eval_dataset


def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, PeftArguments, TrainingArguments))
    model_args, data_args, peft_args, training_args = parser.parse_args_into_dataclasses()
    
    # Load config
    config = load_config()

    # Override model arguments with config values if available
    model_args.model_name_or_path = config.get('fine_tuning', {}).get('model', {}).get('name', model_args.model_name_or_path)
    model_args.use_4bit = config.get('fine_tuning', {}).get('model', {}).get('use_4bit', model_args.use_4bit)
    model_args.use_flash_attention = config.get('fine_tuning', {}).get('model', {}).get('use_flash_attention', model_args.use_flash_attention)

    # Override training arguments with config values if available
    training_args.per_device_train_batch_size = config.get('fine_tuning', {}).get('training', {}).get('batch_size', training_args.per_device_train_batch_size)
    training_args.gradient_accumulation_steps = config.get('fine_tuning', {}).get('training', {}).get('gradient_accumulation_steps', training_args.gradient_accumulation_steps)
    training_args.learning_rate = config.get('fine_tuning', {}).get('training', {}).get('learning_rate', training_args.learning_rate)
    training_args.weight_decay = config.get('fine_tuning', {}).get('training', {}).get('weight_decay', training_args.weight_decay)
    training_args.max_steps = config.get('fine_tuning', {}).get('training', {}).get('max_steps', training_args.max_steps)
    training_args.warmup_ratio = config.get('fine_tuning', {}).get('training', {}).get('warmup_ratio', training_args.warmup_ratio)
    training_args.save_steps = config.get('fine_tuning', {}).get('training', {}).get('save_steps', training_args.save_steps)
    training_args.eval_steps = config.get('fine_tuning', {}).get('training', {}).get('eval_steps', training_args.eval_steps)

    # Override data arguments with config values if available
    data_args.train_file = config.get('fine_tuning', {}).get('data', {}).get('train_file', data_args.train_file)
    data_args.validation_file = config.get('fine_tuning', {}).get('data', {}).get('validation_file', data_args.validation_file)
    data_args.validation_split_percentage = int(config.get('fine_tuning', {}).get('data', {}).get('validation_split', data_args.validation_split_percentage) * 100)

    # Override output directory
    training_args.output_dir = config.get('fine_tuning', {}).get('output', {}).get('dir', training_args.output_dir)

    # Setup logging for the transformers library
    transformers.utils.logging.set_verbosity_info()
    transformers.utils.logging.enable_default_handler()
    
    # Set seed
    set_seed(training_args.seed)
    logger.info(f"Training with random seed: {training_args.seed}")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {model_args.model_name_or_path}")
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
        logger.info("Using 4-bit quantization for training")
        compute_dtype = getattr(torch, model_args.bnb_4bit_compute_dtype)
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
        # "device_map": "auto",
    }
    
    if model_args.use_flash_attention:
        logger.info("Using Flash Attention 2.0")
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
        logger.info("Setting up Parameter-Efficient Fine-Tuning (PEFT)")
        model = prepare_model_for_kbit_training(model)
        
        # Parse target modules
        if peft_args.target_modules == "all-linear":
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        else:
            target_modules = peft_args.target_modules.split(",") if peft_args.target_modules else None
        
        # Configure PEFT (LoRA)
        logger.info(f"LoRA configuration: r={peft_args.lora_r}, alpha={peft_args.lora_alpha}, dropout={peft_args.lora_dropout}")
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
        
        # Log trainable parameters
        trainable_params, all_params = model.get_nb_trainable_parameters()
        logger.info(f"Trainable parameters: {trainable_params:,d} ({trainable_params / all_params:.2%} of {all_params:,d} total parameters)")
    
    # Create the dataset
    train_dataset, eval_dataset = create_patient_dataset(
        data_args.train_file, 
        tokenizer=tokenizer,
        max_seq_length=data_args.max_seq_length,
        validation_split=data_args.validation_split_percentage / 100,
        seed=training_args.seed
    )
    
    # Set remove_unused_columns=False in the training arguments
    if not hasattr(training_args, "remove_unused_columns") or training_args.remove_unused_columns:
        logger.info("Setting remove_unused_columns=False to prevent column mismatch errors")
        training_args.remove_unused_columns = False
    
    # Create a custom data collator that properly handles padding
    logger.info("Creating data collator with padding and truncation")
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )
    
    # Initialize trainer
    logger.info("Initializing Trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,  # Using the custom data collator
    )
    
    # Start training
    logger.info(f"Starting training with batch size: {training_args.per_device_train_batch_size}, gradient accumulation: {training_args.gradient_accumulation_steps}")
    logger.info(f"Using learning rate: {training_args.learning_rate}, weight decay: {training_args.weight_decay}")
    logger.info(f"Training will save checkpoints to: {training_args.output_dir}")
    
    train_result = trainer.train()
    
    # Save model
    logger.info(f"Training complete! Saving model to {training_args.output_dir}")
    trainer.save_model()
    
    # Save tokenizer
    tokenizer.save_pretrained(training_args.output_dir)
    
    # Log metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # Run evaluation if validation dataset exists
    if eval_dataset:
        logger.info("Running final evaluation")
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
        logger.info(f"Evaluation results: {eval_metrics}")
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()