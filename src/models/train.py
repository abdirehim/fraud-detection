"""
Model training for fraud detection.

This module handles training of various machine learning models
for fraud detection with proper error handling and logging.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.utils.class_weight import compute_class_weight

from ..config import MODEL_CONFIG, MODEL_HYPERPARAMS
from ..utils import create_experiment_dir, save_model, setup_logging


class ModelTrainer:
    """
    Model trainer for fraud detection.

    This class handles training of various machine learning models
    with proper handling of imbalanced datasets.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ModelTrainer.

        Args:
            config: Configuration dictionary for training
        """
        self.config = config or MODEL_CONFIG
        self.logger = setup_logging("model_trainer")
        self.models = {}
        self.training_history = {}
        self.is_trained = False

        self.logger.info("Initialized ModelTrainer")

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str,
        test_size: Optional[float] = None,
        random_state: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare data for training by splitting into train/test sets.

        Args:
            df: Input DataFrame
            target_col: Name of the target column
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        try:
            test_size = test_size or self.config.get("test_size", 0.2)
            random_state = random_state or self.config.get("random_state", 42)

            self.logger.info("Preparing data for training")

            # Separate features and target
            X = df.drop(columns=[target_col])
            y = df[target_col]

            # Log class distribution
            class_counts = y.value_counts()
            self.logger.info(f"Class distribution: {class_counts.to_dict()}")
            self.logger.info(f"Fraud rate: {y.mean():.3f}")

            # Split data
            if self.config.get("stratify", True):
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=y
                )
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )

            # Handle NaN values in training data
            if X_train.isnull().any().any():
                self.logger.info("Handling NaN values in training data")
                X_train = X_train.fillna(X_train.median())
                self.logger.info(
                    f"Filled NaN values in {X_train.isnull().sum().sum()} cells"
                )

            # Convert target to binary if needed
            y_train = self.convert_target_to_binary(y_train)

            self.logger.info(f"Training set size: {len(X_train)}")
            self.logger.info(f"Test set size: {len(X_test)}")
            self.logger.info(f"Training fraud rate: {y_train.mean():.3f}")
            self.logger.info(f"Test fraud rate: {y_test.mean():.3f}")

            return X_train, X_test, y_train, y_test

        except Exception as e:
            self.logger.error(f"Error preparing data: {str(e)}")
            raise

    def convert_target_to_binary(self, y: pd.Series) -> pd.Series:
        """
        Convert target variable to binary format [0, 1].

        Args:
            y: Target series that may be encoded

        Returns:
            Binary target series with values [0, 1]
        """
        try:
            y_binary = y.copy()

            # Convert to binary if needed (handle encoded values)
            if len(y_binary.unique()) > 2:
                # If we have more than 2 unique values, convert to binary
                unique_values = sorted(y_binary.unique())
                y_binary = y_binary.map({unique_values[0]: 0, unique_values[1]: 1})
                self.logger.info(
                    f"Converted target to binary: {y_binary.value_counts().to_dict()}"
                )
            else:
                # Ensure it's integer type and properly mapped to 0/1
                unique_values = sorted(y_binary.unique())
                if unique_values != [0, 1]:
                    y_binary = y_binary.map({unique_values[0]: 0, unique_values[1]: 1})
                    self.logger.info(
                        f"Mapped target to binary: {y_binary.value_counts().to_dict()}"
                    )
                else:
                    y_binary = y_binary.astype(int)

            return y_binary

        except Exception as e:
            self.logger.error(f"Error converting target to binary: {str(e)}")
            raise

    def handle_imbalanced_data(
        self, X_train: pd.DataFrame, y_train: pd.Series, method: str = "smote"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Handle imbalanced dataset using various techniques.

        Args:
            X_train: Training features
            y_train: Training targets
            method: Resampling method ('smote', 'undersample', 'smoteenn', 'none')

        Returns:
            Tuple of resampled (X_train, y_train)
        """
        try:
            self.logger.info(f"Handling imbalanced data using {method}")

            # Ensure target is binary and integer type for SMOTE
            y_train_binary = self.convert_target_to_binary(y_train)

            self.logger.info(f"Target unique values: {y_train_binary.unique()}")
            self.logger.info(f"Target data type: {y_train_binary.dtype}")

            if method == "none":
                return X_train, y_train_binary

            elif method == "smote":
                smote = SMOTE(random_state=42, k_neighbors=3)
                X_resampled, y_resampled = smote.fit_resample(X_train, y_train_binary)

            elif method == "undersample":
                undersampler = RandomUnderSampler(random_state=42)
                X_resampled, y_resampled = undersampler.fit_resample(
                    X_train, y_train_binary
                )

            elif method == "smoteenn":
                smoteenn = SMOTEENN(random_state=42)
                X_resampled, y_resampled = smoteenn.fit_resample(
                    X_train, y_train_binary
                )

            elif method == "adasyn":
                from imblearn.over_sampling import ADASYN

                adasyn = ADASYN(random_state=42)
                X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train_binary)

            elif method == "borderline_smote":
                from imblearn.over_sampling import BorderlineSMOTE

                borderline_smote = BorderlineSMOTE(random_state=42, k_neighbors=3)
                X_resampled, y_resampled = borderline_smote.fit_resample(
                    X_train, y_train_binary
                )

            elif method == "class_weights":
                # Use class weights instead of resampling
                self.logger.info("Using class weights instead of resampling")
                return X_train, y_train_binary

            else:
                raise ValueError(f"Unsupported resampling method: {method}")

            # Log resampling results
            original_counts = y_train_binary.value_counts()
            resampled_counts = y_resampled.value_counts()

            self.logger.info(
                f"Original class distribution: {original_counts.to_dict()}"
            )
            self.logger.info(
                f"Resampled class distribution: {resampled_counts.to_dict()}"
            )
            self.logger.info(f"Resampled fraud rate: {y_resampled.mean():.3f}")

            return X_resampled, y_resampled

        except Exception as e:
            self.logger.error(f"Error handling imbalanced data: {str(e)}")
            raise

    def create_models(
        self, class_weights: Optional[Dict[int, float]] = None
    ) -> Dict[str, Any]:
        """
        Create model instances with configured hyperparameters and class weights.

        Args:
            class_weights: Dictionary mapping class labels to weights

        Returns:
            Dictionary of model instances
        """
        try:
            self.logger.info("Creating model instances")

            models = {}

            # Random Forest with class weights
            rf_params = MODEL_HYPERPARAMS.get("random_forest", {}).copy()
            if class_weights:
                rf_params["class_weight"] = class_weights
                self.logger.info(f"Random Forest class weights: {class_weights}")
            models["random_forest"] = RandomForestClassifier(
                random_state=self.config.get("random_state", 42), **rf_params
            )

            # Logistic Regression with class weights
            lr_params = MODEL_HYPERPARAMS.get("logistic_regression", {}).copy()
            if class_weights:
                lr_params["class_weight"] = class_weights
                self.logger.info(f"Logistic Regression class weights: {class_weights}")
            models["logistic_regression"] = LogisticRegression(**lr_params)

            # Add XGBoost if available
            try:
                import xgboost as xgb

                xgb_params = MODEL_HYPERPARAMS.get("xgboost", {}).copy()
                if class_weights:
                    # XGBoost uses scale_pos_weight for binary classification
                    pos_weight = class_weights.get(1, 1.0) / class_weights.get(0, 1.0)
                    xgb_params["scale_pos_weight"] = pos_weight
                    self.logger.info(f"XGBoost scale_pos_weight: {pos_weight}")
                models["xgboost"] = xgb.XGBClassifier(
                    random_state=self.config.get("random_state", 42), **xgb_params
                )
                self.logger.info("XGBoost model created")
            except ImportError:
                self.logger.warning("XGBoost not available. Skipping XGBoost model.")

            self.logger.info(f"Created {len(models)} models: {list(models.keys())}")
            return models

        except Exception as e:
            self.logger.error(f"Error creating models: {str(e)}")
            raise

    def train_model(
        self, model: Any, X_train: pd.DataFrame, y_train: pd.Series, model_name: str
    ) -> Dict[str, Any]:
        """
        Train a single model with cross-validation.

        Args:
            model: Model instance to train
            X_train: Training features
            y_train: Training targets
            model_name: Name of the model

        Returns:
            Dictionary containing training results
        """
        try:
            self.logger.info(f"Training {model_name}")

            # Perform cross-validation
            cv_folds = self.config.get("cv_folds", 5)
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

            cv_scores = cross_val_score(
                model, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1
            )

            self.logger.info(f"{model_name} CV F1 scores: {cv_scores}")
            self.logger.info(
                f"{model_name} CV F1 mean: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})"
            )

            # Train on full training set
            model.fit(X_train, y_train)

            # Store training results
            training_result = {
                "model": model,
                "cv_scores": cv_scores,
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
                "feature_names": X_train.columns.tolist(),
            }

            self.models[model_name] = training_result
            self.logger.info(f"{model_name} training completed successfully")

            return training_result

        except Exception as e:
            self.logger.error(f"Error training {model_name}: {str(e)}")
            raise

    def train_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        resampling_method: str = "smote",
    ) -> Dict[str, Any]:
        """
        Train all models with the given data.

        Args:
            X_train: Training features
            y_train: Training targets
            resampling_method: Method to handle imbalanced data

        Returns:
            Dictionary containing training results for all models
        """
        try:
            self.logger.info("Starting training of all models")

            # Handle imbalanced data
            X_resampled, y_resampled = self.handle_imbalanced_data(
                X_train, y_train, resampling_method
            )

            # Calculate class weights for imbalanced data
            class_counts = y_resampled.value_counts()
            total_samples = len(y_resampled)
            class_weights = {
                0: total_samples / (2 * class_counts[0]),
                1: total_samples / (2 * class_counts[1]),
            }
            self.logger.info(f"Calculated class weights: {class_weights}")

            # Create models with class weights
            models = self.create_models(class_weights)

            # Train each model
            for model_name, model in models.items():
                try:
                    self.train_model(model, X_resampled, y_resampled, model_name)
                except Exception as e:
                    self.logger.error(f"Failed to train {model_name}: {str(e)}")
                    continue

            self.is_trained = True
            self.logger.info(f"Training completed for {len(self.models)} models")

            return self.models

        except Exception as e:
            self.logger.error(f"Error in train_all_models: {str(e)}")
            raise

    def get_best_model(self, metric: str = "cv_mean") -> Tuple[str, Any]:
        """
        Get the best performing model based on the specified metric.

        Args:
            metric: Metric to use for comparison ('cv_mean', 'cv_std')

        Returns:
            Tuple of (model_name, model_instance)
        """
        try:
            if not self.is_trained:
                raise ValueError("No models have been trained yet")

            # Find best model
            best_model_name = None
            best_score = -np.inf if metric == "cv_mean" else np.inf

            for model_name, result in self.models.items():
                score = result[metric]

                if metric == "cv_mean":
                    if score > best_score:
                        best_score = score
                        best_model_name = model_name
                else:  # cv_std
                    if score < best_score:
                        best_score = score
                        best_model_name = model_name

            if best_model_name is None:
                raise ValueError("No valid models found")

            best_model = self.models[best_model_name]["model"]
            self.logger.info(
                f"Best model: {best_model_name} ({metric}: {best_score:.3f})"
            )

            return best_model_name, best_model

        except Exception as e:
            self.logger.error(f"Error getting best model: {str(e)}")
            raise

    def save_models(self, output_dir: Path) -> None:
        """
        Save all trained models to disk.

        Args:
            output_dir: Directory to save models
        """
        try:
            if not self.is_trained:
                raise ValueError("No models have been trained yet")

            output_dir.mkdir(parents=True, exist_ok=True)

            for model_name, result in self.models.items():
                model_path = output_dir / f"{model_name}.pkl"
                save_model(result["model"], model_path, model_name)

                # Save training results
                results_path = output_dir / f"{model_name}_results.json"
                import json

                results_dict = {
                    "cv_scores": result["cv_scores"].tolist(),
                    "cv_mean": float(result["cv_mean"]),
                    "cv_std": float(result["cv_std"]),
                    "feature_names": result["feature_names"],
                }
                with open(results_path, "w") as f:
                    json.dump(results_dict, f, indent=2)

            self.logger.info(f"Saved {len(self.models)} models to {output_dir}")

        except Exception as e:
            self.logger.error(f"Error saving models: {str(e)}")
            raise

    def load_models(self, models_dir: Path) -> None:
        """
        Load trained models from disk.

        Args:
            models_dir: Directory containing saved models
        """
        try:
            from ..utils import load_model

            self.models = {}

            # Find all model files
            model_files = list(models_dir.glob("*.pkl"))

            for model_file in model_files:
                model_name = model_file.stem
                model = load_model(model_file, model_name)

                # Load training results if available
                results_file = models_dir / f"{model_name}_results.json"
                if results_file.exists():
                    import json

                    with open(results_file, "r") as f:
                        results = json.load(f)

                    self.models[model_name] = {
                        "model": model,
                        "cv_scores": np.array(results["cv_scores"]),
                        "cv_mean": results["cv_mean"],
                        "cv_std": results["cv_std"],
                        "feature_names": results["feature_names"],
                    }
                else:
                    self.models[model_name] = {
                        "model": model,
                        "cv_scores": None,
                        "cv_mean": None,
                        "cv_std": None,
                        "feature_names": None,
                    }

            self.is_trained = len(self.models) > 0
            self.logger.info(f"Loaded {len(self.models)} models from {models_dir}")

        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")
            raise
