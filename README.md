# MedSim-LLM

A project for generating and evaluating medical dialogues between doctors and patients using Large Language Models.

## Project Structure
```
medsim-llm/
├── data/
│   ├── raw/          # Raw dialogue datasets
│   └── processed/    # Processed dialogue datasets
├── src/
│   ├── synth_dialogues.py  # Dialogue generation script
│   └── evaluate.py         # Evaluation script
└── tests/           # Test files
```

## Features
- Generate realistic patient-doctor dialogues in Argentinian Spanish
- Create patient cards with detailed medical history
- Evaluate dialogues based on character consistency, recall, coherence, and realism
- Support for both competent and inept doctor conversation styles

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd medsim-llm
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Generating Dialogues
```bash
python src/synth_dialogues.py --n_samples 20500 --dataset_output output.json --base_url http://localhost:7000/v1/
```

### Evaluating Dialogues
```bash
python src/evaluate.py --dataset_input input.jsonl --base_url http://localhost:7000/v1/
```

## License
[Insert License Information]