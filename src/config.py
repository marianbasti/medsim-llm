#!/usr/bin/env python
"""
Configuration module for loading settings from config.yaml
"""
import os
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Configure logging with debug level
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler('config.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses default location.
        
    Returns:
        Dictionary containing configuration
    """
    if config_path is None:
        # Use default location: repo root/config.yaml
        root_dir = Path(__file__).parent.parent
        config_path = root_dir / "config.yaml"
        logger.debug(f"Using default config path: {config_path}")
    else:
        logger.debug(f"Loading config from specified path: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            logger.debug(f"Config loaded successfully with {len(config)} top-level keys")
        return config
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML config: {e}")
        raise

def get_fine_tuning_args(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get fine-tuning arguments from config.
    
    Args:
        config: Configuration dictionary. If None, loads from file.
        
    Returns:
        Dictionary with fine-tuning arguments
    """
    if config is None:
        logger.debug("Loading config for fine-tuning args")
        config = load_config()
    
    ft_args = config.get("fine_tuning", {})
    logger.debug(f"Retrieved fine-tuning args with {len(ft_args)} keys")
    return ft_args

def get_training_args(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get training arguments from config.
    
    Args:
        config: Configuration dictionary. If None, loads from file.
        
    Returns:
        Dictionary with training arguments
    """
    ft_config = get_fine_tuning_args(config)
    training_args = ft_config.get("training", {})
    logger.debug(f"Retrieved training args: {training_args}")
    return training_args

def get_data_args(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get data arguments from config.
    
    Args:
        config: Configuration dictionary. If None, loads from file.
        
    Returns:
        Dictionary with data arguments
    """
    ft_config = get_fine_tuning_args(config)
    data_args = ft_config.get("data", {})
    logger.debug(f"Retrieved data args with paths: {[k for k in data_args if 'path' in k or 'file' in k]}")
    return data_args

def get_model_args(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get model arguments from config.
    
    Args:
        config: Configuration dictionary. If None, loads from file.
        
    Returns:
        Dictionary with model arguments
    """
    ft_config = get_fine_tuning_args(config)
    model_args = ft_config.get("model", {})
    if model_args:
        logger.debug(f"Retrieved model args for: {model_args.get('name', 'unnamed model')}")
    else:
        logger.warning("No model args found in config")
    return model_args

def get_validation_args(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get validation arguments from config.
    
    Args:
        config: Configuration dictionary. If None, loads from file.
        
    Returns:
        Dictionary with validation arguments
    """
    ft_config = get_fine_tuning_args(config)
    validation_args = ft_config.get("validation", {})
    logger.debug(f"Retrieved validation args: {validation_args}")
    return validation_args

# Add Config class for backward compatibility with existing code
class Config:
    """Configuration class for backward compatibility"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize config from file.
        
        Args:
            config_path: Path to config file
        """
        logger.debug(f"Initializing Config object with path: {config_path}")
        self.config = load_config(config_path)
    
    def get(self, key_path: str, default=None):
        """
        Get value from config using dot notation path.
        
        Args:
            key_path: Path to key using dot notation, e.g., 'llm.api_key'
            default: Default value if key not found
        
        Returns:
            Value at key_path or default if not found
        """
        logger.debug(f"Getting config value for key: {key_path}")
        parts = key_path.split('.')
        value = self.config
        
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                logger.debug(f"Key {key_path} not found, returning default: {default}")
                return default
        
        if isinstance(value, dict) or isinstance(value, list):
            logger.debug(f"Retrieved complex value for {key_path} of type {type(value).__name__}")
        else:
            logger.debug(f"Retrieved value for {key_path}: {value}")
        return value


if __name__ == "__main__":
    # Add command-line argument parser for log level
    import argparse
    parser = argparse.ArgumentParser(description="Config module utility")
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help='Set the logging level')
    args = parser.parse_args()
    
    # Set logging level from command line
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Example usage
    config = load_config()
    print("Loaded config:")
    print(f"Model: {get_model_args(config).get('name', 'Not specified')}")
    print(f"Training batch size: {get_training_args(config).get('batch_size', 'Not specified')}")
    print(f"Data file: {get_data_args(config).get('train_file', 'Not specified')}")
    
    # Example of using the Config class
    config_obj = Config()
    print(f"Config using class: {config_obj.get('llm.base_url', 'Not specified')}")