#!/usr/bin/env python
"""
Training script for fine-tuning HuggingFaceTB/SmolLM2-360M model on doctor-patient dialogues.
The goal is to train the model to roleplay as a patient given a patient script.
Simplified for single-GPU training.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field

import torch
from datasets import Dataset
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

# Configure logging
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_config():
    """Load configuration from YAML file"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Successfully loaded config from {config_path}")
        return config
    except Exception as e:
        logger.warning(f"Could not load config file: {e}")
        return {}


@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune."""
    model_name_or_path: str = field(
        default="HuggingFaceTB/SmolLM2-360M",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    use_flash_attention: bool = field(
        default=True,
        metadata={"help": "Whether to use flash attention for faster training"}
    )
    use_4bit: bool = field(
        default=True,
        metadata={"help": "Whether to use 4-bit quantization"}
    )
    bnb_4bit_compute_dtype: str = field(
        default="float16",
        metadata={"help": "Compute dtype for 4-bit quantization"}
    )


@dataclass
class DataArguments:
    """Arguments pertaining to what data we are going to input our model for training."""
    train_file: str = field(
        default=None,
        metadata={"help": "The input training data file (a JSON file)"}
    )
    max_seq_length: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length for training"}
    )
    validation_split: float = field(
        default=0.1,
        metadata={"help": "Proportion of training data to use for validation"}
    )


@dataclass
class TrainingParams:
    """Parameters for training configuration."""
    lora_r: int = field(
        default=64,
        metadata={"help": "Lora attention dimension"}
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "Lora alpha parameter"}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout probability for Lora layers"}
    )


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
    if data_path is None:
        raise ValueError("No training file provided. Please specify a train_file in config.yaml or via command line.")
    
    # Check if the path is relative and convert to absolute path relative to the project root
    if not os.path.isabs(data_path):
        # Try different possible locations
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        possible_paths = [
            os.path.join(project_root, data_path),
            os.path.join(project_root, "data", data_path),
            os.path.join(project_root, "datasets", data_path)
        ]
        
        found = False
        for path in possible_paths:
            if os.path.exists(path):
                data_path = path
                found = True
                break
        
        if not found:
            raise FileNotFoundError(f"Could not find training file. Tried: {possible_paths}")
    
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
                
        # Combine all parts into a well-structured prompt with special tokens
        prompt = f"<|script|>\n{script}\n</|script|>\n\n"
        prompt += f"<|doctor|>\n{doctor_message}\n</|doctor|>\n\n"
        prompt += f"<|patient|>\n"
        
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
        prompts = examples["prompt"]
        responses = examples["response"]
        
        # Join prompts and responses
        texts = [prompt + response for prompt, response in zip(prompts, responses)]
        
        # Calculate prompt lengths for labels
        prompt_lengths = [len(tokenizer(prompt, add_special_tokens=False).input_ids) for prompt in prompts]
        
        # Tokenize with padding and truncation
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
            return_tensors=None,
        )
        
        # Create label mask: -100 for prompt tokens, actual ids for response tokens
        labels = []
        for i, input_ids in enumerate(tokenized["input_ids"]):
            prompt_len = min(prompt_lengths[i], max_seq_length)
            label = [-100] * prompt_len + input_ids[prompt_len:]
            
            # Ensure proper padding with -100
            if len(label) < max_seq_length:
                label.extend([-100] * (max_seq_length - len(label)))
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
        num_proc=4
    )
    
    # Split into train and validation datasets
    if validation_split > 0:
        logger.info(f"Splitting dataset with validation ratio: {validation_split}")
        split_dataset = tokenized_dataset.train_test_split(
            test_size=validation_split, seed=seed
        )
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
        logger.info(f"Split complete - Training: {len(train_dataset)} examples, Validation: {len(eval_dataset)} examples")
    else:
        train_dataset = tokenized_dataset
        eval_dataset = None
        logger.info(f"No validation split - Using all {len(train_dataset)} examples for training")
    
    return train_dataset, eval_dataset


def main():
    # Load config first - before argument parsing
    config = load_config()
    
    # Parse command line arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingParams, TrainingArguments))
    
    # Explicitly set the default arguments using config values
    default_model_name = config.get('fine_tuning', {}).get('model', {}).get('name', "HuggingFaceTB/SmolLM2-360M")
    default_train_file = config.get('fine_tuning', {}).get('data', {}).get('train_file')
    default_output_dir = config.get('fine_tuning', {}).get('output', {}).get('dir', "./output")
    
    logger.info(f"From config - Model: {default_model_name}, Train file: {default_train_file}, Output dir: {default_output_dir}")
    
    # Set default values including those from config
    if len(sys.argv) == 1:
        # Create explicit command line args with sensible defaults
        default_args = []
        
        # Model arguments
        default_args.extend(["--model_name_or_path", default_model_name])
        default_args.extend(["--use_4bit", str(config.get('fine_tuning', {}).get('model', {}).get('use_4bit', True))])
        default_args.extend(["--use_flash_attention", str(config.get('fine_tuning', {}).get('model', {}).get('use_flash_attention', True))])
        
        # Data arguments - MOST IMPORTANT: train_file
        if default_train_file:
            default_args.extend(["--train_file", default_train_file])
            
        default_args.extend(["--max_seq_length", str(config.get('fine_tuning', {}).get('training', {}).get('max_seq_length', 2048))])
        default_args.extend(["--validation_split", str(config.get('fine_tuning', {}).get('data', {}).get('validation_split', 0.1))])
        
        # Training params
        default_args.extend(["--lora_r", str(config.get('fine_tuning', {}).get('training', {}).get('lora_r', 32))])
        default_args.extend(["--lora_alpha", str(config.get('fine_tuning', {}).get('training', {}).get('lora_alpha', 16))])
        default_args.extend(["--lora_dropout", str(config.get('fine_tuning', {}).get('training', {}).get('lora_dropout', 0.1))])
        
        # Training arguments
        default_args.extend(["--output_dir", default_output_dir])
        default_args.extend(["--per_device_train_batch_size", str(config.get('fine_tuning', {}).get('training', {}).get('batch_size', 4))])
        default_args.extend(["--gradient_accumulation_steps", str(config.get('fine_tuning', {}).get('training', {}).get('gradient_accumulation_steps', 8))])
        default_args.extend(["--learning_rate", str(config.get('fine_tuning', {}).get('training', {}).get('learning_rate', 2e-5))])
        default_args.extend(["--weight_decay", str(config.get('fine_tuning', {}).get('training', {}).get('weight_decay', 0.01))])
        default_args.extend(["--warmup_ratio", str(config.get('fine_tuning', {}).get('training', {}).get('warmup_ratio', 0.03))])
        default_args.extend(["--num_train_epochs", str(config.get('fine_tuning', {}).get('training', {}).get('epochs', 3))])
        default_args.extend(["--logging_steps", "10"])
        default_args.extend(["--eval_steps", str(config.get('fine_tuning', {}).get('training', {}).get('eval_steps', 100))])
        default_args.extend(["--save_steps", str(config.get('fine_tuning', {}).get('training', {}).get('save_steps', 100))])
        default_args.extend(["--fp16", "True"])
        default_args.extend(["--save_total_limit", "3"])
        default_args.extend(["--report_to", "none"])
        
        # Parse the default arguments
        logger.info(f"Using default arguments: {default_args}")
        model_args, data_args, training_params, training_args = parser.parse_args_into_dataclasses(default_args)
    else:
        # Parse user-provided command line args
        logger.info("Parsing user-provided command line arguments")
        model_args, data_args, training_params, training_args = parser.parse_args_into_dataclasses()
    
    # Check if train_file is set from command line, if not use from config
    if data_args.train_file is None and default_train_file:
        logger.info(f"Setting train_file from config: {default_train_file}")
        data_args.train_file = default_train_file
    
    # Now validate the train_file is set
    if data_args.train_file is None:
        raise ValueError("No training file provided. Please specify a train_file in config.yaml or via command line.")
    else:
        logger.info(f"Using train_file: {data_args.train_file}")
    
    # Ensure output directory exists
    os.makedirs(training_args.output_dir, exist_ok=True)
    
    # Set seed
    set_seed(training_args.seed)
    logger.info(f"Training with seed: {training_args.seed}")
    
    # Verify GPU availability
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        logger.info(f"Using GPU: cuda:{device_id}")
        # Limit to a single GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    else:
        logger.warning("No GPU found, training will be slow on CPU!")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {model_args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="right",
        trust_remote_code=True,
    )
    
    # Add special tokens for medical dialogue
    special_tokens = {
        "additional_special_tokens": [
            "<|doctor|>", "</|doctor|>",
            "<|patient|>", "</|patient|>",
            "<|script|>", "</|script|>"
        ]
    }
    
    # Add tokens to tokenizer
    num_added_tokens = tokenizer.add_special_tokens(special_tokens)
    logger.info(f"Added {num_added_tokens} special tokens to the tokenizer")
    
    # Ensure pad token exists
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info("Using EOS token as padding token")
    
    # Configure 4-bit quantization if specified
    compute_dtype = getattr(torch, model_args.bnb_4bit_compute_dtype)
    quantization_config = None
    if model_args.use_4bit:
        logger.info("Using 4-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    
    # Load model for single GPU
    model_kwargs = {}
    if model_args.use_flash_attention:
        logger.info("Using Flash Attention 2.0")
        model_kwargs["attn_implementation"] = "flash_attention_2"
        
    # Set device map for a single GPU
    #if torch.cuda.is_available():
    #    model_kwargs["device_map"] = "cuda:0"
    #else:
    #    model_kwargs["device_map"] = "cpu"
    
    # Load base model
    logger.info(f"Loading model {model_args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=quantization_config,
        trust_remote_code=True,
        torch_dtype=compute_dtype,
        **model_kwargs
    )
    
    # Resize token embeddings for new tokens
    model.resize_token_embeddings(len(tokenizer))
    logger.info(f"Model vocab size after resizing: {len(tokenizer)}")
    
    # Prepare model for PEFT training
    logger.info("Preparing model for PEFT/LoRA training")
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    peft_config = LoraConfig(
        r=training_params.lora_r,
        lora_alpha=training_params.lora_alpha,
        lora_dropout=training_params.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Apply LoRA
    model = get_peft_model(model, peft_config)
    
    # Log trainable parameters
    trainable_params, all_params = model.get_nb_trainable_parameters()
    logger.info(f"Trainable parameters: {trainable_params:,d} ({trainable_params/all_params:.2%} of {all_params:,d})")
    
    # Create dataset
    train_dataset, eval_dataset = create_patient_dataset(
        data_args.train_file,
        tokenizer=tokenizer,
        max_seq_length=data_args.max_seq_length,
        validation_split=data_args.validation_split,
        seed=training_args.seed
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True, 
        return_tensors="pt",
        label_pad_token_id=-100,
    )
    
    # Create trainer - ensure no DataParallel
    training_args.remove_unused_columns = False
    
    # Create Trainer for single GPU
    logger.info("Creating Trainer for single GPU training")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Start training
    logger.info(f"Starting training with batch size: {training_args.per_device_train_batch_size}, "
                f"gradient accumulation: {training_args.gradient_accumulation_steps}")
    
    train_result = trainer.train()
    
    # Save final model
    logger.info(f"Saving model to {training_args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    
    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # Run final evaluation
    if eval_dataset:
        logger.info("Running final evaluation")
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()