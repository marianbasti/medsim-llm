# MedSim-LLM: Patient Roleplay Fine-tuning

This repository contains tools for fine-tuning small language models to roleplay as patients in doctor-patient interactions. The current implementation uses Llama-3.2-1B as the base model and fine-tunes it on a dataset of patient scripts and doctor-patient dialogues.

## Project Overview

The goal of this project is to create a lightweight language model that can:
1. Take a patient script (with demographics, medical history, and symptoms)
2. Roleplay as that patient in a realistic way during a doctor-patient conversation

This can be used for medical training simulations, doctor communication practice, and other healthcare education applications.

## Repository Structure

```
├── config.yaml          # Configuration file for the entire pipeline
├── README.md            # This file
├── requirements.txt     # Python dependencies
├── data/
│   ├── raw/             # Raw dialogue datasets
│   └── processed/       # Processed datasets (created during training)
├── models/              # Directory to store fine-tuned models
├── src/
│   ├── benchmark.py     # Benchmarking script
│   ├── config.py        # Configuration module
│   ├── evaluate.py      # Evaluation utilities
│   ├── inference.py     # Inference script for using the model
│   ├── synth_dialogues.py # Script for generating synthetic dialogues
│   ├── train.py         # Fine-tuning script
│   └── validation.py    # Validation utilities
└── tests/               # Unit tests
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/medsim-llm.git
cd medsim-llm
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Training Pipeline

### 1. Data Preparation

The training data consists of patient scripts (containing demographics, medical history, symptoms, etc.) and patient-doctor dialogues. These are used to train the model to respond as a patient.

The data should be in JSONL format, with each line containing:
```json
{
  "script": "patient script as JSON or string",
  "dialogue": [
    {"role": "doctor", "content": "doctor's question"},
    {"role": "patient", "content": "patient's response"},
    ...
  ]
}
```

### 2. Fine-tuning

To fine-tune the model, run:

```bash
python src/train.py \
  --model_name_or_path meta-llama/Llama-3.2-1B \
  --train_file data/raw/medsim-dialogues-llama70b.jsonl \
  --output_dir models/patient-llama-3.2-1B \
  --num_train_epochs 3 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.03 \
  --weight_decay 0.01 \
  --logging_steps 10 \
  --evaluation_strategy steps \
  --eval_steps 100 \
  --save_steps 100 \
  --bf16 \
  --use_peft \
  --lora_r 32 \
  --lora_alpha 16 \
  --lora_dropout 0.1 \
  --use_flash_attention
```

Alternatively, you can edit the parameters in `config.yaml` and run:

```bash
python src/train.py
```

### 3. Evaluation and Benchmarking

To evaluate the fine-tuned model:

```bash
python src/benchmark.py \
  --model_path models/patient-llama-3.2-1B \
  --test_file data/raw/medsim-dialogues-llama70b.jsonl \
  --output_dir evaluation_results \
  --num_samples 100 \
  --use_4bit
```

This will generate evaluation metrics including:
- ROUGE scores (text similarity)
- BERTScore (semantic similarity)
- Role consistency (how well the model stays in character)

### 4. Inference

To use the fine-tuned model for patient roleplay:

```bash
# Interactive mode
python src/inference.py \
  --model_path models/patient-llama-3.2-1B \
  --script_file path/to/patient_script.json \
  --interactive \
  --use_4bit

# Single question mode
python src/inference.py \
  --model_path models/patient-llama-3.2-1B \
  --script_file path/to/patient_script.json \
  --question "How are you feeling today?" \
  --use_4bit
```

#### Using vLLM
Alternatively, you can use vLLM for faster inference:

```bash
docker run --rm --gpus all -p 8001:8001 -v ./output:/models vllm/vllm-openai:latest --model HuggingFaceTB/SmolLM2-360M --port 8001 --chat-template "{% if messages[0]['role'] == 'system' %}{{'<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n'}}{% endif %}{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|im_start|>Doctor: ' + message['content'] + '<|im_end|>\n' }}{% elif message['role'] == 'assistant' %}{{ '<|im_start|>Patient: ' + message['content'] + '<|im_end|>\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>Patient:\n' }}{% endif %}" --enable-lora --lora-modules medsim=/models/finetuned-adapter --max-lora-rank 64

## Model Architecture

This project uses [Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) as the base model and applies Parameter-Efficient Fine-Tuning (PEFT) using LoRA (Low-Rank Adaptation) to reduce memory requirements and training time.

The model is trained to:
1. Understand a patient profile (script)
2. Stay in character as that patient
3. Respond appropriately to doctor questions
4. Maintain consistency throughout the dialogue

## Training Approach

The training approach involves:

1. Converting doctor-patient dialogues into prompt-response pairs
2. Using the patient script as context in the prompt
3. Training the model to generate patient responses
4. Using validation during training to ensure quality responses

## Validation Metrics

The following metrics are used to evaluate the quality of generated patient responses:

- **Consistency**: How well the responses maintain consistency with previous dialogue
- **Relevance**: How relevant the response is to the doctor's question
- **Script adherence**: How well the response adheres to the patient script
- **Role consistency**: How well the model stays in character as the patient

## Acknowledgements

This project builds on work from:
- Meta's Llama 3.2 model
- Hugging Face Transformers library
- Parameter-Efficient Fine-Tuning techniques

## License

[MIT License](LICENSE)