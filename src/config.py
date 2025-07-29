"""
Configuration settings for the fraud detection pipeline.

This module contains all configuration parameters including data paths,
model hyperparameters, logging settings, and feature engineering options.

The configuration is designed to handle imbalanced fraud detection datasets
with specific optimizations for:
- Class imbalance handling (fraud rate typically 1-15%)
- High-dimensional feature engineering (87+ features)
- Model interpretability requirements (SHAP analysis)
- Production deployment considerations

Author: Fraud Detection Team
Date: July 2025
Version: 2.0
"""

import os
from pathlib import Path
from typing import Dict, Any

# =============================================================================
# PROJECT STRUCTURE CONFIGURATION
# =============================================================================

# Project root directory - automatically determined from this file's location
PROJECT_ROOT = Path(__file__).parent.parent

# Data directory structure for organized data management
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"          # Original datasets before processing
PROCESSED_DATA_DIR = DATA_DIR / "processed"  # Cleaned and engineered datasets

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

# Logging setup for comprehensive audit trail and debugging
LOG_DIR = PROJECT_ROOT / "logs"
LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# =============================================================================
# MODEL TRAINING CONFIGURATION
# =============================================================================

# Core model training parameters optimized for fraud detection
MODEL_CONFIG = {
    "random_state": 42,        # Ensures reproducible results across runs
    "test_size": 0.2,          # 20% of data for final testing
    "validation_size": 0.1,    # 10% of training data for validation
    "stratify": True,          # Maintains class distribution in splits
    "cv_folds": 5              # 5-fold cross-validation for robust evaluation
}

# =============================================================================
# FEATURE ENGINEERING CONFIGURATION
# =============================================================================

# Feature processing settings for the 87 engineered features
FEATURE_CONFIG = {
    "numerical_features": [],      # Auto-detected during preprocessing
    "categorical_features": [],    # Auto-detected during preprocessing
    "target_column": "fraud",      # Binary target: 0=legitimate, 1=fraud
    "drop_columns": ["id", "timestamp"],  # Columns to exclude from modeling
    "scaling_method": "standard",  # Options: standard, robust, minmax
    "encoding_method": "label"     # Options: label, onehot, target
}

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================

# Optimized hyperparameters for each model type, specifically tuned for:
# - Imbalanced fraud detection (fraud rate: 1-15%)
# - High-dimensional feature space (87+ features)
# - Interpretability requirements (SHAP analysis)
# - Production performance constraints

MODEL_HYPERPARAMS = {
    "random_forest": {
        # Tree ensemble parameters
        "n_estimators": 200,           # Number of trees (higher = better but slower)
        "max_depth": 15,               # Maximum tree depth (prevents overfitting)
        "min_samples_split": 5,        # Minimum samples to split a node
        "min_samples_leaf": 2,         # Minimum samples in leaf nodes
        "class_weight": "balanced",    # Handles class imbalance automatically
        "criterion": "entropy",        # Split criterion (entropy vs gini)
        "bootstrap": True,             # Bootstrap sampling for diversity
        "oob_score": True              # Out-of-bag score for validation
    },
    "xgboost": {
        # Gradient boosting parameters
        "n_estimators": 300,           # Number of boosting rounds
        "max_depth": 8,                # Tree depth (shallower for interpretability)
        "learning_rate": 0.05,         # Slow learning rate for better generalization
        "subsample": 0.8,              # Row sampling for each tree
        "colsample_bytree": 0.8,       # Column sampling for each tree
        "scale_pos_weight": 10,        # Weight for positive class (fraud)
        "reg_alpha": 0.1,              # L1 regularization
        "reg_lambda": 1.0,             # L2 regularization
        "eval_metric": "auc"           # Evaluation metric during training
    },
    "logistic_regression": {
        # Linear model parameters
        "C": 0.1,                      # Inverse regularization strength (stronger regularization)
        "class_weight": "balanced",    # Handles class imbalance
        "max_iter": 2000,              # Maximum iterations for convergence
        "random_state": 42,            # Reproducible results
        "solver": "liblinear",         # Optimized for small datasets
        "penalty": "l1"                # L1 regularization for feature selection
    }
}

# =============================================================================
# EVALUATION METRICS CONFIGURATION
# =============================================================================

# Comprehensive evaluation metrics for imbalanced fraud detection
# Each metric provides different insights into model performance
EVALUATION_METRICS = [
    "accuracy",      # Overall correctness (may be misleading for imbalanced data)
    "precision",     # True positives / (True positives + False positives)
    "recall",        # True positives / (True positives + False negatives)
    "f1_score",      # Harmonic mean of precision and recall
    "roc_auc",       # Area under ROC curve (overall discriminative ability)
    "pr_auc"         # Area under Precision-Recall curve (better for imbalanced data)
]

# =============================================================================
# EXPLAINABILITY CONFIGURATION
# =============================================================================

# SHAP (SHapley Additive exPlanations) settings for model interpretability
# Critical for fraud detection to understand decision factors
EXPLAINABILITY_CONFIG = {
    "use_shap": True,                 # Enable SHAP analysis
    "shap_sample_size": 1000,         # Sample size for SHAP computation (speed vs accuracy)
    "feature_importance_top_k": 20,   # Top K features to display in importance plots
    "plot_style": "default"           # Plotting style for SHAP visualizations
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_config() -> Dict[str, Any]:
    """
    Get the complete configuration dictionary.
    
    This function consolidates all configuration parameters into a single
    dictionary for easy access throughout the pipeline. It's used by
    various components to ensure consistent configuration across the system.
    
    Returns:
        Dict[str, Any]: Complete configuration dictionary containing:
            - data_paths: File system paths for data storage
            - logging: Logging configuration parameters
            - model: Model training configuration
            - features: Feature engineering settings
            - hyperparams: Model-specific hyperparameters
            - evaluation: Evaluation metrics configuration
            - explainability: SHAP analysis settings
    
    Example:
        >>> config = get_config()
        >>> print(config['model']['test_size'])
        0.2
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

def validate_config() -> bool:
    """
    Validate the configuration settings for consistency and correctness.
    
    This function checks that all configuration parameters are valid
    and consistent with each other. It's called during pipeline initialization
    to catch configuration errors early.
    
    Returns:
        bool: True if configuration is valid, False otherwise
    
    Raises:
        ValueError: If configuration is invalid with specific error details
    """
    # Validate data paths exist
    if not RAW_DATA_DIR.exists():
        raise ValueError(f"Raw data directory does not exist: {RAW_DATA_DIR}")
    
    if not PROCESSED_DATA_DIR.exists():
        # Create processed data directory if it doesn't exist
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Validate model configuration
    if MODEL_CONFIG["test_size"] + MODEL_CONFIG["validation_size"] >= 1.0:
        raise ValueError("Test size + validation size must be less than 1.0")
    
    # Validate hyperparameters
    for model_name, params in MODEL_HYPERPARAMS.items():
        if model_name == "random_forest":
            if params["n_estimators"] <= 0:
                raise ValueError("Random Forest n_estimators must be positive")
        elif model_name == "xgboost":
            if params["learning_rate"] <= 0:
                raise ValueError("XGBoost learning_rate must be positive")
        elif model_name == "logistic_regression":
            if params["C"] <= 0:
                raise ValueError("Logistic Regression C must be positive")
    
    return True 