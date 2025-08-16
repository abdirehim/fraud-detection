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

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

# =============================================================================
# PROJECT STRUCTURE CONFIGURATION
# =============================================================================

# Project root directory - automatically determined from this file's location
PROJECT_ROOT = Path(__file__).parent.parent

# Data directory structure for organized data management
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"  # Original datasets before processing
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
    "random_state": 42,  # Ensures reproducible results across runs
    "test_size": 0.2,  # 20% of data for final testing
    "validation_size": 0.1,  # 10% of training data for validation
    "stratify": True,  # Maintains class distribution in splits
    "cv_folds": 5,  # 5-fold cross-validation for robust evaluation
}

# =============================================================================
# FEATURE ENGINEERING CONFIGURATION
# =============================================================================

# Feature processing settings for the 87 engineered features
FEATURE_CONFIG = {
    "numerical_features": [],  # Auto-detected during preprocessing
    "categorical_features": [],  # Auto-detected during preprocessing
    "target_column": "fraud",  # Binary target: 0=legitimate, 1=fraud
    "drop_columns": ["id", "timestamp"],  # Columns to exclude from modeling
    "scaling_method": "standard",  # Options: standard, robust, minmax
    "encoding_method": "label",  # Options: label, onehot, target
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
        "n_estimators": 200,  # Number of trees (higher = better but slower)
        "max_depth": 15,  # Maximum tree depth (prevents overfitting)
        "min_samples_split": 5,  # Minimum samples to split a node
        "min_samples_leaf": 2,  # Minimum samples in leaf nodes
        "class_weight": "balanced",  # Handles class imbalance automatically
        "criterion": "entropy",  # Split criterion (entropy vs gini)
        "bootstrap": True,  # Bootstrap sampling for diversity
        "oob_score": True,  # Out-of-bag score for validation
    },
    "xgboost": {
        # Gradient boosting parameters
        "n_estimators": 300,  # Number of boosting rounds
        "max_depth": 8,  # Tree depth (shallower for interpretability)
        "learning_rate": 0.05,  # Slow learning rate for better generalization
        "subsample": 0.8,  # Row sampling for each tree
        "colsample_bytree": 0.8,  # Column sampling for each tree
        "scale_pos_weight": 10,  # Weight for positive class (fraud)
        "reg_alpha": 0.1,  # L1 regularization
        "reg_lambda": 1.0,  # L2 regularization
        "eval_metric": "auc",  # Evaluation metric during training
    },
    "logistic_regression": {
        # Linear model parameters
        "C": 0.1,  # Inverse regularization strength (stronger regularization)
        "class_weight": "balanced",  # Handles class imbalance
        "max_iter": 2000,  # Maximum iterations for convergence
        "random_state": 42,  # Reproducible results
        "solver": "liblinear",  # Optimized for small datasets
        "penalty": "l1",  # L1 regularization for feature selection
    },
}

# =============================================================================
# EVALUATION METRICS CONFIGURATION
# =============================================================================

# Comprehensive evaluation metrics for imbalanced fraud detection
# Each metric provides different insights into model performance
EVALUATION_METRICS = [
    "accuracy",  # Overall correctness (may be misleading for imbalanced data)
    "precision",  # True positives / (True positives + False positives)
    "recall",  # True positives / (True positives + False negatives)
    "f1_score",  # Harmonic mean of precision and recall
    "roc_auc",  # Area under ROC curve (overall discriminative ability)
    "pr_auc",  # Area under Precision-Recall curve (better for imbalanced data)
]

# =============================================================================
# EXPLAINABILITY CONFIGURATION
# =============================================================================

# SHAP (SHapley Additive exPlanations) settings for model interpretability
# Critical for fraud detection to understand decision factors
EXPLAINABILITY_CONFIG = {
    "use_shap": True,  # Enable SHAP analysis
    "shap_sample_size": 1000,  # Sample size for SHAP computation (speed vs accuracy)
    "feature_importance_top_k": 20,  # Top K features to display in importance plots
    "plot_style": "default",  # Plotting style for SHAP visualizations
}

# =============================================================================
# CONFIGURATION MANAGER CLASS
# =============================================================================


class ConfigManager:
    """
    Centralized configuration management for the fraud detection pipeline.

    This class provides a modular approach to configuration management with:
    - Centralized access to all configuration parameters
    - Environment variable override support
    - Configuration validation and error handling
    - Dependency injection support for components
    - Runtime configuration updates

    The ConfigManager acts as a single source of truth for all configuration
    while maintaining backward compatibility with existing code.
    """

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize the ConfigManager with optional custom configuration.

        Args:
            config_dict: Optional custom configuration dictionary.
                        If None, uses default configuration from module constants.
        """
        self.logger = logging.getLogger(__name__)
        self._config = config_dict or self._build_default_config()
        self._validated = False

    def _build_default_config(self) -> Dict[str, Any]:
        """Build the default configuration from module constants."""
        return {
            "data_paths": {
                "project_root": str(PROJECT_ROOT),
                "data_dir": str(DATA_DIR),
                "raw_data_dir": str(RAW_DATA_DIR),
                "processed_data_dir": str(PROCESSED_DATA_DIR),
            },
            "logging": {
                "log_dir": str(LOG_DIR),
                "log_level": LOG_LEVEL,
                "log_format": LOG_FORMAT,
            },
            "model": MODEL_CONFIG.copy(),
            "features": FEATURE_CONFIG.copy(),
            "hyperparams": {k: v.copy() for k, v in MODEL_HYPERPARAMS.items()},
            "evaluation": EVALUATION_METRICS.copy(),
            "explainability": EXPLAINABILITY_CONFIG.copy(),
        }

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key with optional default.

        Supports nested key access using dot notation (e.g., 'model.test_size').
        Also checks environment variables for overrides.

        Args:
            key: Configuration key (supports dot notation for nested access)
            default: Default value if key is not found

        Returns:
            Configuration value or default

        Example:
            >>> config_manager = ConfigManager()
            >>> test_size = config_manager.get('model.test_size', 0.2)
            >>> log_level = config_manager.get('logging.log_level', 'INFO')
        """
        # Check environment variable override first
        env_key = key.upper().replace(".", "_")
        env_value = os.getenv(f"FRAUD_DETECTION_{env_key}")
        if env_value is not None:
            return self._convert_env_value(env_value)

        # Navigate nested dictionary using dot notation
        keys = key.split(".")
        value = self._config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            self.logger.debug(
                f"Configuration key '{key}' not found, using default: {default}"
            )
            return default

    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value by key.

        Supports nested key setting using dot notation.

        Args:
            key: Configuration key (supports dot notation for nested setting)
            value: Value to set

        Example:
            >>> config_manager = ConfigManager()
            >>> config_manager.set('model.test_size', 0.3)
            >>> config_manager.set('logging.log_level', 'DEBUG')
        """
        keys = key.split(".")
        config_section = self._config

        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config_section:
                config_section[k] = {}
            config_section = config_section[k]

        # Set the final value
        config_section[keys[-1]] = value
        self.logger.debug(f"Configuration key '{key}' set to: {value}")

        # Mark as needing revalidation
        self._validated = False

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific model.

        Args:
            model_name: Name of the model ('random_forest', 'xgboost', 'logistic_regression')

        Returns:
            Model-specific configuration dictionary

        Raises:
            ValueError: If model_name is not supported

        Example:
            >>> config_manager = ConfigManager()
            >>> rf_config = config_manager.get_model_config('random_forest')
            >>> print(rf_config['n_estimators'])
        """
        hyperparams = self.get("hyperparams", {})
        if model_name not in hyperparams:
            raise ValueError(
                f"Unsupported model: {model_name}. Available models: {list(hyperparams.keys())}"
            )

        # Combine general model config with model-specific hyperparameters
        model_config = self.get("model", {}).copy()
        model_config.update(hyperparams[model_name])

        return model_config

    def get_data_config(self) -> Dict[str, Any]:
        """
        Get data processing configuration.

        Returns:
            Data processing configuration including paths and feature settings

        Example:
            >>> config_manager = ConfigManager()
            >>> data_config = config_manager.get_data_config()
            >>> print(data_config['raw_data_dir'])
        """
        return {
            **self.get("data_paths", {}),
            **self.get("features", {}),
        }

    def get_logging_config(self) -> Dict[str, Any]:
        """
        Get logging configuration.

        Returns:
            Logging configuration dictionary

        Example:
            >>> config_manager = ConfigManager()
            >>> log_config = config_manager.get_logging_config()
            >>> print(log_config['log_level'])
        """
        return self.get("logging", {})

    def validate(self) -> bool:
        """
        Validate the current configuration for consistency and correctness.

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration is invalid with specific error details

        Example:
            >>> config_manager = ConfigManager()
            >>> is_valid = config_manager.validate()
        """
        try:
            # Validate data paths
            raw_data_dir = Path(self.get("data_paths.raw_data_dir"))
            processed_data_dir = Path(self.get("data_paths.processed_data_dir"))

            if not raw_data_dir.exists():
                raise ValueError(f"Raw data directory does not exist: {raw_data_dir}")

            if not processed_data_dir.exists():
                processed_data_dir.mkdir(parents=True, exist_ok=True)
                self.logger.info(
                    f"Created processed data directory: {processed_data_dir}"
                )

            # Validate model configuration
            test_size = self.get("model.test_size", 0.2)
            validation_size = self.get("model.validation_size", 0.1)

            if test_size + validation_size >= 1.0:
                raise ValueError(
                    f"Test size ({test_size}) + validation size ({validation_size}) must be less than 1.0"
                )

            # Validate hyperparameters for each model
            hyperparams = self.get("hyperparams", {})
            for model_name, params in hyperparams.items():
                self._validate_model_hyperparams(model_name, params)

            # Validate evaluation metrics
            metrics = self.get("evaluation", [])
            if not isinstance(metrics, list) or len(metrics) == 0:
                raise ValueError("Evaluation metrics must be a non-empty list")

            self._validated = True
            self.logger.info("Configuration validation successful")
            return True

        except Exception as e:
            self.logger.error(f"Configuration validation failed: {str(e)}")
            raise

    def _validate_model_hyperparams(
        self, model_name: str, params: Dict[str, Any]
    ) -> None:
        """Validate hyperparameters for a specific model."""
        if model_name == "random_forest":
            if params.get("n_estimators", 0) <= 0:
                raise ValueError(
                    f"Random Forest n_estimators must be positive, got: {params.get('n_estimators')}"
                )
        elif model_name == "xgboost":
            if params.get("learning_rate", 0) <= 0:
                raise ValueError(
                    f"XGBoost learning_rate must be positive, got: {params.get('learning_rate')}"
                )
        elif model_name == "logistic_regression":
            if params.get("C", 0) <= 0:
                raise ValueError(
                    f"Logistic Regression C must be positive, got: {params.get('C')}"
                )

    def _convert_env_value(self, value: str) -> Union[str, int, float, bool]:
        """Convert environment variable string to appropriate type."""
        # Try boolean
        if value.lower() in ("true", "false"):
            return value.lower() == "true"

        # Try integer
        try:
            return int(value)
        except ValueError:
            pass

        # Try float
        try:
            return float(value)
        except ValueError:
            pass

        # Return as string
        return value

    def update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """
        Update configuration from a dictionary.

        Args:
            config_dict: Dictionary containing configuration updates

        Example:
            >>> config_manager = ConfigManager()
            >>> updates = {'model': {'test_size': 0.3}, 'logging': {'log_level': 'DEBUG'}}
            >>> config_manager.update_from_dict(updates)
        """

        def deep_update(base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
            """Recursively update nested dictionaries."""
            for key, value in update_dict.items():
                if (
                    isinstance(value, dict)
                    and key in base_dict
                    and isinstance(base_dict[key], dict)
                ):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value

        deep_update(self._config, config_dict)
        self._validated = False
        self.logger.info("Configuration updated from dictionary")

    def to_dict(self) -> Dict[str, Any]:
        """
        Get the complete configuration as a dictionary.

        Returns:
            Complete configuration dictionary (deep copy for immutability)

        Example:
            >>> config_manager = ConfigManager()
            >>> config_dict = config_manager.to_dict()
        """
        import copy

        return copy.deepcopy(self._config)

    def is_validated(self) -> bool:
        """
        Check if the configuration has been validated.

        Returns:
            True if configuration has been validated successfully
        """
        return self._validated


# Global ConfigManager instance for backward compatibility
_global_config_manager = None


def get_config_manager() -> ConfigManager:
    """
    Get the global ConfigManager instance.

    Returns:
        Global ConfigManager instance

    Example:
        >>> config_manager = get_config_manager()
        >>> test_size = config_manager.get('model.test_size')
    """
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = ConfigManager()
    return _global_config_manager


# =============================================================================
# UTILITY FUNCTIONS (BACKWARD COMPATIBILITY)
# =============================================================================


def get_config() -> Dict[str, Any]:
    """
    Get the complete configuration dictionary.

    DEPRECATED: Use get_config_manager() for new code.
    This function is maintained for backward compatibility.

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
    # Use ConfigManager for consistency
    config_manager = get_config_manager()
    return config_manager.to_dict()


def validate_config() -> bool:
    """
    Validate the configuration settings for consistency and correctness.

    DEPRECATED: Use get_config_manager().validate() for new code.
    This function is maintained for backward compatibility.

    This function checks that all configuration parameters are valid
    and consistent with each other. It's called during pipeline initialization
    to catch configuration errors early.

    Returns:
        bool: True if configuration is valid, False otherwise

    Raises:
        ValueError: If configuration is invalid with specific error details
    """
    # For backward compatibility, use the original validation logic
    # that works with module constants (needed for existing tests)

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
