"""
Utility functions for the fraud detection pipeline.

This module contains logging setup, helper functions, and common utilities
used throughout the pipeline.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json
import pickle
from datetime import datetime

from .config import LOG_DIR, LOG_LEVEL, LOG_FORMAT


def setup_logging(
    name: str = "fraud_detection",
    log_level: str = LOG_LEVEL,
    log_dir: Path = LOG_DIR,
    log_format: str = LOG_FORMAT
) -> logging.Logger:
    """
    Set up logging configuration for both file and console output.
    
    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory to store log files
        log_format: Format string for log messages
        
    Returns:
        Configured logger instance
    """
    # Create log directory if it doesn't exist
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatters
    formatter = logging.Formatter(log_format)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{name}_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def save_model(model: Any, filepath: Path, model_name: str = "model") -> None:
    """
    Save a trained model to disk.
    
    Args:
        model: Trained model object
        filepath: Path where to save the model
        model_name: Name of the model for logging
    """
    try:
        # Create directory if it doesn't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
            
        logger = logging.getLogger(__name__)
        logger.info(f"Model '{model_name}' saved successfully to {filepath}")
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to save model '{model_name}': {str(e)}")
        raise


def load_model(filepath: Path, model_name: str = "model") -> Any:
    """
    Load a trained model from disk.
    
    Args:
        filepath: Path to the saved model
        model_name: Name of the model for logging
        
    Returns:
        Loaded model object
    """
    try:
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
            
        logger = logging.getLogger(__name__)
        logger.info(f"Model '{model_name}' loaded successfully from {filepath}")
        
        return model
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to load model '{model_name}': {str(e)}")
        raise


def save_metrics(metrics: Dict[str, float], filepath: Path) -> None:
    """
    Save evaluation metrics to a JSON file.
    
    Args:
        metrics: Dictionary of metric names and values
        filepath: Path where to save the metrics
    """
    try:
        # Create directory if it doesn't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=4)
            
        logger = logging.getLogger(__name__)
        logger.info(f"Metrics saved successfully to {filepath}")
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to save metrics: {str(e)}")
        raise


def load_metrics(filepath: Path) -> Dict[str, float]:
    """
    Load evaluation metrics from a JSON file.
    
    Args:
        filepath: Path to the metrics file
        
    Returns:
        Dictionary of metric names and values
    """
    try:
        with open(filepath, 'r') as f:
            metrics = json.load(f)
            
        logger = logging.getLogger(__name__)
        logger.info(f"Metrics loaded successfully from {filepath}")
        
        return metrics
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to load metrics: {str(e)}")
        raise


def create_experiment_dir(experiment_name: str, base_dir: Path) -> Path:
    """
    Create a directory for a new experiment.
    
    Args:
        experiment_name: Name of the experiment
        base_dir: Base directory for experiments
        
    Returns:
        Path to the created experiment directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = base_dir / f"{experiment_name}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Created experiment directory: {experiment_dir}")
    
    return experiment_dir


def validate_data_path(filepath: Path) -> bool:
    """
    Validate that a data file exists and is accessible.
    
    Args:
        filepath: Path to the data file
        
    Returns:
        True if file exists and is accessible, False otherwise
    """
    if not filepath.exists():
        logger = logging.getLogger(__name__)
        logger.error(f"Data file does not exist: {filepath}")
        return False
        
    if not filepath.is_file():
        logger = logging.getLogger(__name__)
        logger.error(f"Path is not a file: {filepath}")
        return False
        
    return True


def get_file_size_mb(filepath: Path) -> float:
    """
    Get file size in megabytes.
    
    Args:
        filepath: Path to the file
        
    Returns:
        File size in megabytes
    """
    try:
        size_bytes = filepath.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        return round(size_mb, 2)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to get file size: {str(e)}")
        return 0.0 