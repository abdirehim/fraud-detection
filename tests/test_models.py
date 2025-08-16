"""
Tests for model training and evaluation components.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.models.evaluate import ModelEvaluator
from src.models.train import ModelTrainer


class TestModelTrainer:
    """Test cases for ModelTrainer."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        X = pd.DataFrame(
            np.random.randn(100, 5), columns=[f"feature_{i}" for i in range(5)]
        )
        y = pd.Series(np.random.randint(0, 2, 100), name="target")
        return X, y

    @pytest.mark.unit
    def test_trainer_initialization(self):
        """Test ModelTrainer initialization."""
        trainer = ModelTrainer()
        assert trainer is not None
        assert not trainer.is_trained

    @pytest.mark.unit
    def test_prepare_data(self, sample_data):
        """Test data preparation."""
        X, y = sample_data
        df = pd.concat([X, y], axis=1)

        trainer = ModelTrainer()
        X_train, X_test, y_train, y_test = trainer.prepare_data(df, "target")

        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)

    @pytest.mark.unit
    def test_create_models(self):
        """Test model creation."""
        trainer = ModelTrainer()
        models = trainer.create_models()

        assert "random_forest" in models
        assert "logistic_regression" in models
        assert isinstance(models["random_forest"], RandomForestClassifier)


class TestModelEvaluator:
    """Test cases for ModelEvaluator."""

    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions for testing."""
        y_true = pd.Series([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.4, 0.8, 0.7])
        return y_true, y_pred, y_prob

    @pytest.mark.unit
    def test_evaluator_initialization(self):
        """Test ModelEvaluator initialization."""
        evaluator = ModelEvaluator()
        assert evaluator is not None

    @pytest.mark.unit
    def test_compute_metrics(self, sample_predictions):
        """Test metrics computation."""
        y_true, y_pred, y_prob = sample_predictions
        evaluator = ModelEvaluator()

        metrics = evaluator.compute_metrics(y_true, y_pred, y_prob)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "roc_auc" in metrics

    @pytest.mark.unit
    def test_optimize_threshold(self, sample_predictions):
        """Test threshold optimization."""
        y_true, _, y_prob = sample_predictions
        evaluator = ModelEvaluator()

        threshold, score = evaluator.optimize_threshold(y_true, y_prob)

        assert 0 <= threshold <= 1
        assert score >= 0
