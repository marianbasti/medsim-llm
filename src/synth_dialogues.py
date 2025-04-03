import random
import argparse
import logging
import json
from tqdm import tqdm
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from config import Config
from utils import save_batch, generate_sample
from schemas import argentinian_names, illneses

# Configure logging with more informative format
logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('dialogue_generation.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=20500, help="Number of dataset samples to generate")
    parser.add_argument("--dataset_output", type=str, default='dialogo_medico-paciente_es.jsonl', help="Path to JSON output")
    parser.add_argument("--config", type=str, default='config.yaml', help="Path to config file")
    parser.add_argument("--base_url", type=str, default='http://localhost:7000/v1/', help="Base URL for API requests")
    parser.add_argument("--log_level", type=str, default="INFO", 
                      choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                      help="Set the logging level")
    
    args = parser.parse_args()
    
    # Set logging level from command line
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    logger.debug(f"Set logging level to: {args.log_level}")
    
    # Load config
    config = Config(args.config)
    logger.info(f"Loaded configuration from: {args.config}")
    
    # Log config details for debugging
    logger.debug(f"LLM base URL: {config.get('llm.base_url')}")
    logger.debug(f"LLM timeout: {config.get('llm.timeout', 70000)}ms")
    
    client = OpenAI(
        base_url=config.get('llm.base_url'),
        api_key=config.get('llm.api_key'),
        timeout=config.get('llm.timeout', 70000)
    )
    logger.debug("OpenAI client initialized")
    
    # Get available model from the API
    try:
        logger.debug("Fetching available models from API")
        models_response = client.models.list()
        model_id = models_response.data[0].id
        logger.info(f"Using model: {model_id}")
    except Exception as e:
        logger.error(f"Error getting models: {str(e)}")
        # Log detailed error information
        if hasattr(e, 'response'):
            logger.error(f"Response status: {e.response.status_code}")
            logger.error(f"Response content: {e.response.text}")
        raise SystemExit("Failed to get model ID from API")
        
    # Create output directory if needed
    output_path = Path(args.dataset_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Output path ensured: {output_path}")
    
    # Log some samples for debugging
    logger.debug(f"Using {len(argentinian_names)} names and {len(illneses)} illness types")
    logger.debug(f"Sample name: {argentinian_names[0]}")
    logger.debug(f"Sample illness: {illneses[0][0]} - {illneses[0][1][:50]}...")
    
    # Generate sample cases
    logger.info(f"Generating {args.n_samples} dialogue samples")
    cases = [(random.choice(argentinian_names), 
              random.choice(illneses),
              random.randint(18, 71)) 
            for _ in range(args.n_samples)]
    
    current_batch = []
    batch_size = config.get('generation.batch_size', 10)
    logger.debug(f"Using batch size of {batch_size}")
    pbar = tqdm(total=args.n_samples, desc="Generating samples")
    
    try:
        num_workers = config.get('generation.num_workers', mp.cpu_count())
        logger.info(f"Using {num_workers} workers for parallel generation")
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            logger.debug("Started ThreadPoolExecutor")
            completed = 0
            errors = 0
            
            for sample in executor.map(lambda x: generate_sample(client, x[0], x[1], model_id, x[2]), cases):
                if sample:
                    current_batch.append(sample)
                    completed += 1
                    
                    if len(current_batch) >= batch_size:
                        logger.debug(f"Saving batch of {len(current_batch)} samples")
                        save_batch(current_batch, args.dataset_output)
                        current_batch = []
                else:
                    errors += 1
                    logger.debug(f"Error count: {errors}")
                        
                pbar.update(1)
                if completed > 0 and completed % 100 == 0:
                    logger.info(f"Progress: {completed} samples completed, {errors} errors")
                    
    except KeyboardInterrupt:
        logger.warning("Generation interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error during generation: {str(e)}", exc_info=True)
    finally:
        if current_batch:
            logger.debug(f"Saving final batch of {len(current_batch)} samples")
            save_batch(current_batch, args.dataset_output)
        pbar.close()
        logger.info(f"Generation complete. Results saved to {args.dataset_output}")
        
        # Final statistics
        try:
            with open(args.dataset_output, 'r', encoding='utf-8') as f:
                line_count = sum(1 for _ in f)
            logger.info(f"Final dataset contains {line_count} samples")
        except Exception:
            logger.warning("Could not count final samples in output file")