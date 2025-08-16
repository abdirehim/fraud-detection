"""
Model explainability and interpretation for fraud detection.

This module provides tools for understanding model predictions using
SHAP values, feature importance, and other interpretability techniques.
"""

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available. Install with: pip install shap")

from .config import EXPLAINABILITY_CONFIG
from .utils import setup_logging


class ModelExplainer:
    """
    Model explainer for fraud detection models.

    This class provides methods to explain model predictions using
    various interpretability techniques including SHAP values.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ModelExplainer.

        Args:
            config: Configuration dictionary for explainability
        """
        self.config = config or EXPLAINABILITY_CONFIG
        self.logger = setup_logging("model_explainer")
        self.shap_explainer = None
        self.feature_names = None

        if not SHAP_AVAILABLE and self.config.get("use_shap", True):
            self.logger.warning(
                "SHAP not available. SHAP-based explanations will be disabled."
            )
            self.config["use_shap"] = False

        self.logger.info("Initialized ModelExplainer")

    def fit_shap_explainer(
        self, model: Any, X_train: pd.DataFrame, model_type: str = "tree"
    ) -> None:
        """
        Fit SHAP explainer for the model.

        Args:
            model: Trained model
            X_train: Training features
            model_type: Type of model ('tree', 'linear', 'neural')
        """
        try:
            if not self.config.get("use_shap", True) or not SHAP_AVAILABLE:
                self.logger.info(
                    "SHAP explainer skipped (not configured or not available)"
                )
                return

            self.logger.info("Fitting SHAP explainer")
            self.feature_names = X_train.columns.tolist()

            # Sample data if too large
            sample_size = self.config.get("shap_sample_size", 1000)
            if len(X_train) > sample_size:
                X_sample = X_train.sample(n=sample_size, random_state=42)
                self.logger.info(f"Sampled {sample_size} instances for SHAP explainer")
            else:
                X_sample = X_train

            # Create explainer based on model type
            if model_type == "tree":
                self.shap_explainer = shap.TreeExplainer(model)
            elif model_type == "linear":
                self.shap_explainer = shap.LinearExplainer(model, X_sample)
            elif model_type == "neural":
                self.shap_explainer = shap.DeepExplainer(model, X_sample)
            else:
                # Use KernelExplainer as fallback
                self.shap_explainer = shap.KernelExplainer(model.predict, X_sample)

            self.logger.info(
                f"SHAP explainer fitted successfully for {model_type} model"
            )

        except Exception as e:
            self.logger.error(f"Error fitting SHAP explainer: {str(e)}")
            self.shap_explainer = None

    def get_shap_values(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Get SHAP values for the given data.

        Args:
            X: Input features

        Returns:
            SHAP values array or None if explainer not available
        """
        try:
            if self.shap_explainer is None:
                self.logger.warning("SHAP explainer not fitted")
                return None

            # Sample data if too large
            sample_size = self.config.get("shap_sample_size", 1000)
            if len(X) > sample_size:
                X_sample = X.sample(n=sample_size, random_state=42)
                self.logger.info(f"Sampled {sample_size} instances for SHAP values")
            else:
                X_sample = X

            shap_values = self.shap_explainer.shap_values(X_sample)

            # Handle different output formats
            if isinstance(shap_values, list):
                if len(shap_values) > 1:
                    shap_values = shap_values[
                        1
                    ]  # For binary classification, get positive class
                else:
                    shap_values = shap_values[0]

            self.logger.info(f"Computed SHAP values for {len(X_sample)} instances")
            return shap_values

        except Exception as e:
            self.logger.error(f"Error computing SHAP values: {str(e)}")
            return None

    def get_feature_importance(
        self,
        model: Any,
        feature_names: Optional[List[str]] = None,
        method: str = "shap",
    ) -> Dict[str, float]:
        """
        Get feature importance from the model.

        Args:
            model: Trained model
            feature_names: List of feature names
            method: Method to compute importance ('shap', 'permutation', 'builtin')

        Returns:
            Dictionary mapping feature names to importance scores
        """
        try:
            self.logger.info(f"Computing feature importance using {method} method")

            if method == "shap" and self.shap_explainer is not None:
                # Use SHAP values for feature importance
                if hasattr(model, "feature_importances_"):
                    # For tree-based models, use built-in importance
                    importance_scores = model.feature_importances_
                else:
                    # Use mean absolute SHAP values
                    X_sample = pd.DataFrame(
                        np.random.randn(100, len(feature_names)), columns=feature_names
                    )
                    shap_values = self.get_shap_values(X_sample)
                    if shap_values is not None:
                        importance_scores = np.mean(np.abs(shap_values), axis=0)
                    else:
                        raise ValueError(
                            "Could not compute SHAP-based feature importance"
                        )

            elif method == "builtin" and hasattr(model, "feature_importances_"):
                # Use model's built-in feature importance
                importance_scores = model.feature_importances_

            elif method == "builtin" and hasattr(model, "coef_"):
                # Use model coefficients (for linear models)
                importance_scores = np.abs(model.coef_[0])

            else:
                # Fallback to permutation importance
                from sklearn.inspection import permutation_importance

                X_sample = pd.DataFrame(
                    np.random.randn(100, len(feature_names)), columns=feature_names
                )
                y_sample = np.random.randint(0, 2, 100)
                result = permutation_importance(
                    model, X_sample, y_sample, n_repeats=5, random_state=42
                )
                importance_scores = result.importances_mean

            # Create feature importance dictionary
            feature_names = (
                feature_names
                or self.feature_names
                or [f"feature_{i}" for i in range(len(importance_scores))]
            )
            importance_dict = dict(zip(feature_names, importance_scores))

            # Sort by importance
            importance_dict = dict(
                sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            )

            self.logger.info(
                f"Computed feature importance for {len(importance_dict)} features"
            )
            return importance_dict

        except Exception as e:
            self.logger.error(f"Error computing feature importance: {str(e)}")
            return {}

    def plot_feature_importance(
        self,
        importance_dict: Dict[str, float],
        top_k: Optional[int] = None,
        figsize: Tuple[int, int] = (10, 6),
    ) -> None:
        """
        Plot feature importance.

        Args:
            importance_dict: Dictionary of feature importance scores
            top_k: Number of top features to plot
            figsize: Figure size
        """
        try:
            top_k = top_k or self.config.get("feature_importance_top_k", 20)

            # Get top k features
            top_features = dict(list(importance_dict.items())[:top_k])

            # Create plot
            plt.figure(figsize=figsize)
            features = list(top_features.keys())
            scores = list(top_features.values())

            # Create horizontal bar plot
            y_pos = np.arange(len(features))
            plt.barh(y_pos, scores)
            plt.yticks(y_pos, features)
            plt.xlabel("Feature Importance")
            plt.title(f"Top {len(features)} Feature Importance")
            plt.gca().invert_yaxis()  # Invert y-axis to show most important at top

            plt.tight_layout()
            plt.show()

            self.logger.info(
                f"Plotted feature importance for top {len(features)} features"
            )

        except Exception as e:
            self.logger.error(f"Error plotting feature importance: {str(e)}")

    def plot_shap_summary(
        self,
        X: pd.DataFrame,
        shap_values: Optional[np.ndarray] = None,
        figsize: Tuple[int, int] = (10, 8),
    ) -> None:
        """
        Plot SHAP summary plot.

        Args:
            X: Input features
            shap_values: Pre-computed SHAP values
            figsize: Figure size
        """
        try:
            if not SHAP_AVAILABLE:
                self.logger.warning("SHAP not available for summary plot")
                return

            if shap_values is None:
                shap_values = self.get_shap_values(X)
                if shap_values is None:
                    return

            # Sample data if too large
            sample_size = self.config.get("shap_sample_size", 1000)
            if len(X) > sample_size:
                X_sample = X.sample(n=sample_size, random_state=42)
                shap_values_sample = shap_values[:sample_size]
            else:
                X_sample = X
                shap_values_sample = shap_values

            # Create SHAP summary plot
            plt.figure(figsize=figsize)
            shap.summary_plot(shap_values_sample, X_sample, show=False)
            plt.title("SHAP Summary Plot")
            plt.tight_layout()
            plt.show()

            self.logger.info("Plotted SHAP summary")

        except Exception as e:
            self.logger.error(f"Error plotting SHAP summary: {str(e)}")

    def explain_prediction(
        self,
        X: pd.DataFrame,
        prediction: int,
        shap_values: Optional[np.ndarray] = None,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """
        Explain a single prediction.

        Args:
            X: Input features for the prediction
            prediction: Model prediction (0 or 1)
            shap_values: Pre-computed SHAP values
            top_k: Number of top features to include in explanation

        Returns:
            Dictionary containing explanation details
        """
        try:
            explanation = {
                "prediction": int(prediction),  # Ensure prediction is int
                "prediction_label": "Fraud" if int(prediction) == 1 else "Legitimate",
                "top_features": [],
                "shap_values": None,
            }

            if self.shap_explainer is not None and SHAP_AVAILABLE:
                # Get SHAP values for this prediction
                if shap_values is None:
                    shap_values = self.shap_explainer.shap_values(X)
                    if isinstance(shap_values, list):
                        if len(shap_values) > 1:
                            shap_values = shap_values[1]  # For binary classification
                        else:
                            shap_values = shap_values[0]

                # Get feature contributions
                feature_contributions = shap_values[0]  # First instance
                feature_names = X.columns.tolist()

                # Create list of (feature_name, contribution) tuples
                contributions = list(zip(feature_names, feature_contributions))
                contributions.sort(key=lambda x: abs(x[1]), reverse=True)

                # Get top k features
                top_features = contributions[:top_k]
                explanation["top_features"] = [
                    {
                        "feature": feature,
                        "contribution": float(contribution),
                        "abs_contribution": float(abs(contribution)),
                    }
                    for feature, contribution in top_features
                ]
                explanation["shap_values"] = feature_contributions.tolist()

            self.logger.info(f"Explained prediction: {explanation['prediction_label']}")
            return explanation

        except Exception as e:
            self.logger.error(f"Error explaining prediction: {str(e)}")
            return {"prediction": int(prediction), "error": str(e)}

    def plot_prediction_explanation(
        self, explanation: Dict[str, Any], figsize: Tuple[int, int] = (10, 6)
    ) -> None:
        """
        Plot explanation for a single prediction.

        Args:
            explanation: Explanation dictionary from explain_prediction
            figsize: Figure size
        """
        try:
            if not explanation.get("top_features"):
                self.logger.warning("No features to plot in explanation")
                return

            # Create plot
            plt.figure(figsize=figsize)

            features = [item["feature"] for item in explanation["top_features"]]
            contributions = [
                item["contribution"] for item in explanation["top_features"]
            ]
            colors = ["red" if c > 0 else "blue" for c in contributions]

            # Create horizontal bar plot
            y_pos = np.arange(len(features))
            bars = plt.barh(y_pos, contributions, color=colors)
            plt.yticks(y_pos, features)
            plt.xlabel("SHAP Value (Impact on Prediction)")
            plt.title(f'Prediction Explanation: {explanation["prediction_label"]}')
            plt.axvline(x=0, color="black", linestyle="-", alpha=0.3)

            # Add value labels on bars
            for i, (bar, contribution) in enumerate(zip(bars, contributions)):
                plt.text(
                    contribution,
                    i,
                    f"{contribution:.3f}",
                    ha="left" if contribution > 0 else "right",
                    va="center",
                )

            plt.tight_layout()
            plt.show()

            self.logger.info("Plotted prediction explanation")

        except Exception as e:
            self.logger.error(f"Error plotting prediction explanation: {str(e)}")

    def generate_explanation_report(
        self,
        model: Any,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive explanation report.

        Args:
            model: Trained model
            X: Input features
            y: Target values (optional)
            output_path: Path to save the report (optional)

        Returns:
            Dictionary containing the explanation report
        """
        try:
            self.logger.info("Generating comprehensive explanation report")

            report = {
                "model_info": {
                    "model_type": type(model).__name__,
                    "n_features": X.shape[1],
                    "n_samples": X.shape[0],
                },
                "feature_importance": {},
                "shap_analysis": {},
                "sample_explanations": [],
            }

            # Get feature importance
            feature_importance = self.get_feature_importance(model, X.columns.tolist())
            report["feature_importance"] = feature_importance

            # SHAP analysis
            if self.config.get("use_shap", True) and SHAP_AVAILABLE:
                shap_values = self.get_shap_values(X)
                if shap_values is not None:
                    report["shap_analysis"] = {
                        "shap_values_computed": True,
                        "mean_abs_shap": float(np.mean(np.abs(shap_values))),
                        "shap_values_shape": shap_values.shape,
                    }

                    # Explain a few sample predictions
                    sample_indices = np.random.choice(
                        len(X), min(5, len(X)), replace=False
                    )
                    for idx in sample_indices:
                        sample_X = X.iloc[idx : idx + 1]
                        if hasattr(model, "predict"):
                            sample_pred = model.predict(sample_X)[0]
                            sample_explanation = self.explain_prediction(
                                sample_X, sample_pred, shap_values[idx : idx + 1]
                            )
                            report["sample_explanations"].append(sample_explanation)

            # Save report if path provided
            if output_path:
                import json

                with open(output_path, "w") as f:
                    json.dump(report, f, indent=2, default=str)
                self.logger.info(f"Explanation report saved to {output_path}")

            self.logger.info("Explanation report generated successfully")
            return report

        except Exception as e:
            self.logger.error(f"Error generating explanation report: {str(e)}")
            return {"error": str(e)}
