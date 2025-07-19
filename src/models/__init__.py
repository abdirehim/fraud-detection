"""
Model training and evaluation for fraud detection.

This package contains modules for training, evaluating, and managing
machine learning models for fraud detection tasks.
"""

from .train import ModelTrainer
from .evaluate import ModelEvaluator

__all__ = ['ModelTrainer', 'ModelEvaluator'] 