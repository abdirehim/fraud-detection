"""
Model training and evaluation for fraud detection.

This package contains modules for training, evaluating, and managing
machine learning models for fraud detection tasks.
"""

from .evaluate import ModelEvaluator
from .train import ModelTrainer

__all__ = ["ModelTrainer", "ModelEvaluator"]
