#!/usr/bin/env python
"""
Configuration module for loading settings from config.yaml
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional


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
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_fine_tuning_args(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get fine-tuning arguments from config.
    
    Args:
        config: Configuration dictionary. If None, loads from file.
        
    Returns:
        Dictionary with fine-tuning arguments
    """
    if config is None:
        config = load_config()
    
    return config.get("fine_tuning", {})


def get_training_args(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get training arguments from config.
    
    Args:
        config: Configuration dictionary. If None, loads from file.
        
    Returns:
        Dictionary with training arguments
    """
    ft_config = get_fine_tuning_args(config)
    return ft_config.get("training", {})


def get_data_args(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get data arguments from config.
    
    Args:
        config: Configuration dictionary. If None, loads from file.
        
    Returns:
        Dictionary with data arguments
    """
    ft_config = get_fine_tuning_args(config)
    return ft_config.get("data", {})


def get_model_args(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get model arguments from config.
    
    Args:
        config: Configuration dictionary. If None, loads from file.
        
    Returns:
        Dictionary with model arguments
    """
    ft_config = get_fine_tuning_args(config)
    return ft_config.get("model", {})


def get_validation_args(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get validation arguments from config.
    
    Args:
        config: Configuration dictionary. If None, loads from file.
        
    Returns:
        Dictionary with validation arguments
    """
    ft_config = get_fine_tuning_args(config)
    return ft_config.get("validation", {})


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
        Get value from config using dot notation path.
        
        Args:
            key_path: Path to key using dot notation, e.g., 'llm.api_key'
            default: Default value if key not found
        
        Returns:
            Value at key_path or default if not found
        """
        parts = key_path.split('.')
        value = self.config
        
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        
        return value


if __name__ == "__main__":
    # Example usage
    config = load_config()
    print("Loaded config:")
    print(f"Model: {get_model_args(config)['name']}")
    print(f"Training batch size: {get_training_args(config)['batch_size']}")
    print(f"Data file: {get_data_args(config)['train_file']}")
    
    # Example of using the Config class
    config_obj = Config()
    print(f"Config using class: {config_obj.get('llm.base_url')}")