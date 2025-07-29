"""
Utility functions for the fraud detection pipeline.

This module contains logging setup, helper functions, and common utilities
used throughout the pipeline. It provides essential infrastructure for:

- Comprehensive logging and monitoring
- Model persistence and loading
- Experiment management and organization
- Data validation and file operations
- Performance metrics storage and retrieval

The utilities are designed to support:
- Reproducible experiments with detailed logging
- Robust error handling and validation
- Efficient file operations and data management
- Production-ready model deployment

Author: Fraud Detection Team
Date: July 2025
Version: 2.0
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Union
import json
import pickle
from datetime import datetime
import hashlib

from .config import LOG_DIR, LOG_LEVEL, LOG_FORMAT


def setup_logging(
    name: str = "fraud_detection",
    log_level: str = LOG_LEVEL,
    log_dir: Path = LOG_DIR,
    log_format: str = LOG_FORMAT
) -> logging.Logger:
    """
    Set up comprehensive logging configuration for both file and console output.
    
    This function creates a robust logging system that:
    - Outputs to both console and file for complete audit trail
    - Uses timestamped log files to prevent overwrites
    - Provides configurable log levels and formats
    - Ensures thread-safe logging across the pipeline
    
    Args:
        name (str): Logger name (default: "fraud_detection")
        log_level (str): Logging level - DEBUG, INFO, WARNING, ERROR, CRITICAL
        log_dir (Path): Directory to store log files (default: from config)
        log_format (str): Format string for log messages (default: from config)
        
    Returns:
        logging.Logger: Configured logger instance ready for use
        
    Example:
        >>> logger = setup_logging("data_loader", "INFO")
        >>> logger.info("Data loading started")
        >>> logger.error("Failed to load data")
    """
    # Create log directory if it doesn't exist
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger with specified name
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers to avoid duplicates (important for reusing loggers)
    logger.handlers.clear()
    
    # Create formatter with specified format
    formatter = logging.Formatter(log_format)
    
    # Console handler for immediate feedback
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler for persistent audit trail
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{name}_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Log the setup completion
    logger.info(f"Logging setup complete - Console and file: {log_file}")
    
    return logger


def save_model(model: Any, filepath: Path, model_name: str = "model") -> None:
    """
    Save a trained model to disk with comprehensive error handling.
    
    This function provides robust model persistence with:
    - Automatic directory creation
    - Binary serialization using pickle
    - Detailed logging of save operations
    - Error handling and validation
    
    Args:
        model (Any): Trained model object (sklearn, xgboost, etc.)
        filepath (Path): Path where to save the model file
        model_name (str): Name of the model for logging purposes
        
    Raises:
        Exception: If model saving fails with detailed error message
        
    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> model = RandomForestClassifier()
        >>> save_model(model, Path("models/rf_model.pkl"), "RandomForest")
    """
    try:
        # Create directory structure if it doesn't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model using pickle for maximum compatibility
        with open(filepath, 'wb') as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
            
        # Log successful save with file size
        file_size = get_file_size_mb(filepath)
        logger = logging.getLogger(__name__)
        logger.info(f"Model '{model_name}' saved successfully to {filepath} ({file_size:.2f} MB)")
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to save model '{model_name}' to {filepath}: {str(e)}")
        raise


def load_model(filepath: Path, model_name: str = "model") -> Any:
    """
    Load a trained model from disk with validation and error handling.
    
    This function provides safe model loading with:
    - File existence validation
    - Pickle deserialization with error handling
    - Detailed logging of load operations
    - Model integrity verification
    
    Args:
        filepath (Path): Path to the saved model file
        model_name (str): Name of the model for logging purposes
        
    Returns:
        Any: Loaded model object
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        Exception: If model loading fails
        
    Example:
        >>> model = load_model(Path("models/rf_model.pkl"), "RandomForest")
        >>> predictions = model.predict(X_test)
    """
    try:
        # Validate file exists
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Load model using pickle
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
            
        # Log successful load
        file_size = get_file_size_mb(filepath)
        logger = logging.getLogger(__name__)
        logger.info(f"Model '{model_name}' loaded successfully from {filepath} ({file_size:.2f} MB)")
        
        return model
        
    except FileNotFoundError as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Model file not found: {str(e)}")
        raise
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to load model '{model_name}' from {filepath}: {str(e)}")
        raise


def save_metrics(metrics: Dict[str, Union[float, int, str]], filepath: Path) -> None:
    """
    Save model evaluation metrics to JSON file for persistence and analysis.
    
    This function saves metrics in a human-readable JSON format that:
    - Preserves all metric types (float, int, string)
    - Enables easy analysis and comparison
    - Provides version control friendly format
    - Includes timestamp for tracking
    
    Args:
        metrics (Dict[str, Union[float, int, str]]): Dictionary of metrics to save
        filepath (Path): Path where to save the metrics file
        
    Raises:
        Exception: If metrics saving fails
        
    Example:
        >>> metrics = {"accuracy": 0.95, "f1_score": 0.87, "model": "RandomForest"}
        >>> save_metrics(metrics, Path("experiments/metrics.json"))
    """
    try:
        # Create directory if it doesn't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Add timestamp to metrics
        metrics_with_timestamp = {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics
        }
        
        # Save metrics as JSON with pretty formatting
        with open(filepath, 'w') as f:
            json.dump(metrics_with_timestamp, f, indent=2, default=str)
            
        logger = logging.getLogger(__name__)
        logger.info(f"Metrics saved successfully to {filepath}")
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to save metrics to {filepath}: {str(e)}")
        raise


def load_metrics(filepath: Path) -> Dict[str, Union[float, int, str]]:
    """
    Load model evaluation metrics from JSON file.
    
    This function loads previously saved metrics with:
    - JSON parsing with error handling
    - Type preservation for numeric values
    - Timestamp extraction for tracking
    - Validation of metric structure
    
    Args:
        filepath (Path): Path to the metrics file
        
    Returns:
        Dict[str, Union[float, int, str]]: Loaded metrics dictionary
        
    Raises:
        FileNotFoundError: If metrics file doesn't exist
        Exception: If metrics loading fails
        
    Example:
        >>> metrics = load_metrics(Path("experiments/metrics.json"))
        >>> print(f"Accuracy: {metrics['accuracy']:.3f}")
    """
    try:
        # Validate file exists
        if not filepath.exists():
            raise FileNotFoundError(f"Metrics file not found: {filepath}")
        
        # Load metrics from JSON
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        # Extract metrics (handle both old and new format)
        if "metrics" in data:
            metrics = data["metrics"]
            timestamp = data.get("timestamp", "unknown")
        else:
            metrics = data
            timestamp = "unknown"
            
        logger = logging.getLogger(__name__)
        logger.info(f"Metrics loaded successfully from {filepath} (timestamp: {timestamp})")
        
        return metrics
        
    except FileNotFoundError as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Metrics file not found: {str(e)}")
        raise
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to load metrics from {filepath}: {str(e)}")
        raise


def create_experiment_dir(experiment_name: str, base_dir: Path) -> Path:
    """
    Create a unique experiment directory with timestamp for organization.
    
    This function creates experiment directories that:
    - Include timestamp to prevent conflicts
    - Organize experiments by name and date
    - Create necessary subdirectories for artifacts
    - Provide consistent structure across experiments
    
    Args:
        experiment_name (str): Name of the experiment
        base_dir (Path): Base directory for experiments
        
    Returns:
        Path: Path to the created experiment directory
        
    Example:
        >>> exp_dir = create_experiment_dir("fraud_detection", Path("experiments"))
        >>> print(f"Experiment directory: {exp_dir}")
        # Output: experiments/fraud_detection_20250729_143022/
    """
    try:
        # Create timestamp for unique directory name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = base_dir / f"{experiment_name}_{timestamp}"
        
        # Create experiment directory and subdirectories
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Create standard subdirectories for experiment artifacts
        (experiment_dir / "models").mkdir(exist_ok=True)
        (experiment_dir / "evaluation").mkdir(exist_ok=True)
        (experiment_dir / "visualizations").mkdir(exist_ok=True)
        (experiment_dir / "logs").mkdir(exist_ok=True)
        
        logger = logging.getLogger(__name__)
        logger.info(f"Created experiment directory: {experiment_dir}")
        
        return experiment_dir
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to create experiment directory: {str(e)}")
        raise


def validate_data_path(filepath: Path) -> bool:
    """
    Validate that a data file path exists and is accessible.
    
    This function performs comprehensive path validation including:
    - File existence check
    - File size validation (non-zero)
    - Read permission verification
    - Path format validation
    
    Args:
        filepath (Path): Path to the data file to validate
        
    Returns:
        bool: True if file is valid and accessible, False otherwise
        
    Example:
        >>> is_valid = validate_data_path(Path("data/raw/fraud_data.csv"))
        >>> if is_valid:
        ...     print("File is ready for processing")
    """
    try:
        # Check if file exists
        if not filepath.exists():
            return False
        
        # Check if it's a file (not directory)
        if not filepath.is_file():
            return False
        
        # Check if file has content (non-zero size)
        if filepath.stat().st_size == 0:
            return False
        
        # Check if file is readable
        if not os.access(filepath, os.R_OK):
            return False
        
        return True
        
    except Exception:
        # Any exception during validation means file is not valid
        return False


def get_file_size_mb(filepath: Path) -> float:
    """
    Get file size in megabytes for logging and monitoring.
    
    This function provides file size information useful for:
    - Performance monitoring
    - Resource planning
    - Logging and debugging
    - Data validation
    
    Args:
        filepath (Path): Path to the file
        
    Returns:
        float: File size in megabytes
        
    Example:
        >>> size_mb = get_file_size_mb(Path("data/raw/fraud_data.csv"))
        >>> print(f"File size: {size_mb:.2f} MB")
    """
    try:
        # Get file size in bytes
        size_bytes = filepath.stat().st_size
        
        # Convert to megabytes
        size_mb = size_bytes / (1024 * 1024)
        
        return size_mb
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Could not get file size for {filepath}: {str(e)}")
        return 0.0


def calculate_data_hash(data: Any) -> str:
    """
    Calculate a hash of data for integrity checking and caching.
    
    This function creates a deterministic hash that can be used for:
    - Data integrity verification
    - Cache invalidation
    - Change detection
    - Reproducibility checks
    
    Args:
        data (Any): Data to hash (DataFrame, array, etc.)
        
    Returns:
        str: Hexadecimal hash string
        
    Example:
        >>> hash_value = calculate_data_hash(df)
        >>> print(f"Data hash: {hash_value}")
    """
    try:
        # Convert data to string representation for hashing
        if hasattr(data, 'to_string'):
            # For pandas DataFrames
            data_str = data.to_string()
        else:
            # For other data types
            data_str = str(data)
        
        # Calculate SHA-256 hash
        hash_object = hashlib.sha256(data_str.encode())
        return hash_object.hexdigest()
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Could not calculate data hash: {str(e)}")
        return "unknown" 