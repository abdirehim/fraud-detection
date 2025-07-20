"""
Configuration settings for the fraud detection pipeline.

This module contains all configuration parameters including data paths,
model hyperparameters, logging settings, and feature engineering options.
"""

import os
from pathlib import Path
from typing import Dict, Any

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Logging configuration
LOG_DIR = PROJECT_ROOT / "logs"
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Model configuration
MODEL_CONFIG = {
    "random_state": 42,
    "test_size": 0.2,
    "validation_size": 0.1,
    "stratify": True,
    "cv_folds": 5
}

# Feature engineering settings
FEATURE_CONFIG = {
    "numerical_features": [],
    "categorical_features": [],
    "target_column": "fraud",
    "drop_columns": ["id", "timestamp"],
    "scaling_method": "standard",  # standard, robust, minmax
    "encoding_method": "label"     # label, onehot, target
}

# Model hyperparameters optimized for imbalanced fraud detection
MODEL_HYPERPARAMS = {
    "random_forest": {
        "n_estimators": 200,
        "max_depth": 15,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "criterion": "entropy",
        "bootstrap": True,
        "oob_score": True
    },
    "xgboost": {
        "n_estimators": 300,
        "max_depth": 8,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 10,  # Higher weight for minority class
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "eval_metric": "auc"
    },
    "logistic_regression": {
        "C": 0.1,  # Stronger regularization for imbalanced data
        "class_weight": "balanced",
        "max_iter": 2000,
        "random_state": 42,
        "solver": "liblinear",
        "penalty": "l1"  # L1 regularization for feature selection
    }
}

# Evaluation metrics
EVALUATION_METRICS = [
    "accuracy",
    "precision",
    "recall", 
    "f1_score",
    "roc_auc",
    "pr_auc"
]

# Explainability settings
EXPLAINABILITY_CONFIG = {
    "use_shap": True,
    "shap_sample_size": 1000,
    "feature_importance_top_k": 20,
    "plot_style": "default"
}

def get_config() -> Dict[str, Any]:
    """
    Get the complete configuration dictionary.
    
    Returns:
        Dict containing all configuration parameters
    """
    return {
        "data_paths": {
            "raw_data_dir": str(RAW_DATA_DIR),
            "processed_data_dir": str(PROCESSED_DATA_DIR)
        },
        "logging": {
            "log_dir": str(LOG_DIR),
            "log_level": LOG_LEVEL,
            "log_format": LOG_FORMAT
        },
        "model": MODEL_CONFIG,
        "features": FEATURE_CONFIG,
        "hyperparams": MODEL_HYPERPARAMS,
        "evaluation": EVALUATION_METRICS,
        "explainability": EXPLAINABILITY_CONFIG
    } 