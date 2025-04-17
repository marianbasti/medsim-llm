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
│   ├── evaluate_patient_model.py      # Evaluation utilities
│   ├── synth_dialogues.py # Script for generating synthetic dialogues
│   ├── train.py         # Fine-tuning script
│   ├── validation.py    # Validation utilities
│   └── merge_peft.py  # Model merging utilities
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
  --train_file data/raw/dialogues.jsonl \
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

### 3. Model Merging

### 4. Evaluation and Benchmarking


#### Using vLLM
Alternatively, you can use vLLM for faster inference:

```bash
docker run --runtime nvidia --gpus all -v ./models:/models -p 8000:8000 --ipc=host --rm vllm/vllm-openai:latest --model /models/merged_model --enable-chunked-prefill  --gpu-memory-utilization 0.7 --max-model-len 2048 --chat-template /models/merged_model/template.jinja
```

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