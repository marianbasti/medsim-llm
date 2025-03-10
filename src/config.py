#!/usr/bin/env python
"""
Configuration module for loading settings from config.yaml
"""
import os
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Configure logging with a consistent format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler('medsim.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, try to find config.yaml in current directory.
    
    Returns:
        Dict containing configuration
    """
    if not config_path:
        config_path = "config.yaml"
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
    except FileNotFoundError:
        logger.warning(f"Config file not found at {config_path}, using default configuration")
        return {}
    except yaml.YAMLError as e:
        logger.error(f"Error parsing config file: {str(e)}")
        return {}

def get_fine_tuning_args(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get fine-tuning arguments from config.
    
    Args:
        config: Configuration dict. If None, load from default location.
    
    Returns:
        Dict containing fine-tuning arguments
    """
    if config is None:
        config = load_config()
    
    fine_tuning = config.get("fine_tuning", {})
    
    # Extract model args
    model_args = fine_tuning.get("model", {})
    
    # Extract training args
    training_args = fine_tuning.get("training", {})
    
    # Extract data args
    data_args = fine_tuning.get("data", {})
    
    # Extract output args
    output_args = fine_tuning.get("output", {})
    
    return {
        "model": model_args,
        "training": training_args,
        "data": data_args,
        "output": output_args
    }

def get_training_args(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get training arguments from config.
    
    Args:
        config: Configuration dict. If None, load from default location.
    
    Returns:
        Dict containing training arguments
    """
    if config is None:
        config = load_config()
    
    fine_tuning = config.get("fine_tuning", {})
    return fine_tuning.get("training", {})

def get_data_args(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get data arguments from config.
    
    Args:
        config: Configuration dict. If None, load from default location.
    
    Returns:
        Dict containing data arguments
    """
    if config is None:
        config = load_config()
    
    fine_tuning = config.get("fine_tuning", {})
    return fine_tuning.get("data", {})

def get_model_args(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get model arguments from config.
    
    Args:
        config: Configuration dict. If None, load from default location.
    
    Returns:
        Dict containing model arguments
    """
    if config is None:
        config = load_config()
    
    fine_tuning = config.get("fine_tuning", {})
    return fine_tuning.get("model", {})

def get_validation_args(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get validation arguments from config.
    
    Args:
        config: Configuration dict. If None, load from default location.
    
    Returns:
        Dict containing validation arguments
    """
    if config is None:
        config = load_config()
    
    fine_tuning = config.get("fine_tuning", {})
    return fine_tuning.get("validation", {})

def get_generation_args(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get generation arguments from config.
    
    Args:
        config: Configuration dict. If None, load from default location.
    
    Returns:
        Dict containing generation arguments
    """
    if config is None:
        config = load_config()
    
    return config.get("generation", {})

def configure_logging(level_name: str = "INFO") -> None:
    """
    Configure logging with the specified level.
    
    Args:
        level_name: Name of the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    level = getattr(logging, level_name)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Update format for all handlers
    formatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    for handler in root_logger.handlers:
        handler.setFormatter(formatter)
    
    logger.info(f"Logging level set to {level_name}")


# Add Config class for backward compatibility with existing code
class Config:
    """Configuration class for backward compatibility"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize config from file.
        
        Args:
            config_path: Path to config file
        """
        self.config = load_config(config_path)
    
    def get(self, key_path: str, default=None):
        """
        Get a value from the config using a dot-separated path.
        
        Args:
            key_path: Dot-separated path to the key (e.g., "llm.api_key")
            default: Default value to return if key is not found
        
        Returns:
            The value at the specified path or the default
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default


if __name__ == "__main__":
    # Add command-line argument parser for log level
    import argparse
    parser = argparse.ArgumentParser(description="Config module utility")
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help='Set the logging level')
    args = parser.parse_args()
    
    # Configure logging with command line argument
    configure_logging(args.log_level)
    
    # Example usage
    config = load_config()
    logger.info("Config module loaded successfully")
    
    # Print some config values as examples
    model_name = get_model_args(config).get('name', 'Not specified')
    batch_size = get_training_args(config).get('batch_size', 'Not specified')
    train_file = get_data_args(config).get('train_file', 'Not specified')
    
    logger.info(f"Model: {model_name}")
    logger.info(f"Training batch size: {batch_size}")
    logger.info(f"Training data file: {train_file}")