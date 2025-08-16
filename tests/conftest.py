"""
Pytest configuration and fixtures for fraud detection tests.

This module provides shared fixtures and configuration for all tests
in the fraud detection project.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.data_loader import DataLoader
from src.explainability import ModelExplainer
from src.models.evaluate import ModelEvaluator
from src.models.train import ModelTrainer
from src.preprocess import DataPreprocessor


# Test data fixtures
@pytest.fixture
def sample_fraud_data():
    """Create sample fraud detection data for testing."""
    np.random.seed(42)
    n_samples = 100

    data = {
        "user_id": [f"user_{i}" for i in range(n_samples)],
        "signup_time": pd.date_range("2023-01-01", periods=n_samples, freq="H"),
        "purchase_time": pd.date_range("2023-01-01", periods=n_samples, freq="H"),
        "purchase_value": np.random.uniform(10, 1000, n_samples),
        "device_id": [f"device_{i%20}" for i in range(n_samples)],
        "source": np.random.choice(["SEO", "Ads", "Direct"], n_samples),
        "browser": np.random.choice(["Chrome", "Firefox", "Safari"], n_samples),
        "sex": np.random.choice(["M", "F"], n_samples),
        "age": np.random.randint(18, 80, n_samples),
        "ip_address": [f"192.168.1.{i%255}" for i in range(n_samples)],
        "class": np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),  # 10% fraud rate
    }

    return pd.DataFrame(data)


@pytest.fixture
def sample_creditcard_data():
    """Create sample credit card data for testing."""
    np.random.seed(42)
    n_samples = 100

    data = {
        "Time": np.random.uniform(0, 172800, n_samples),  # 48 hours in seconds
        "Amount": np.random.uniform(1, 500, n_samples),
        "Class": np.random.choice(
            [0, 1], n_samples, p=[0.998, 0.002]
        ),  # Very low fraud rate
    }

    # Add V1-V5 features (PCA components)
    for i in range(1, 6):
        data[f"V{i}"] = np.random.normal(0, 1, n_samples)

    return pd.DataFrame(data)


@pytest.fixture
def sample_ip_mapping():
    """Create sample IP to country mapping data."""
    data = {
        "lower_bound_ip_address": ["192.168.1.0", "10.0.0.0", "172.16.0.0"],
        "upper_bound_ip_address": [
            "192.168.255.255",
            "10.255.255.255",
            "172.31.255.255",
        ],
        "country": ["USA", "Canada", "UK"],
    }

    return pd.DataFrame(data)


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_data_loader():
    """Create a mock DataLoader for testing."""
    mock_loader = Mock(spec=DataLoader)
    mock_loader.load_fraud_data.return_value = pd.DataFrame(
        {
            "user_id": ["user_1", "user_2"],
            "purchase_value": [100.0, 200.0],
            "class": [0, 1],
        }
    )
    return mock_loader


@pytest.fixture
def mock_preprocessor():
    """Create a mock DataPreprocessor for testing."""
    mock_preprocessor = Mock(spec=DataPreprocessor)
    mock_preprocessor.fit_transform.return_value = pd.DataFrame(
        {"feature_1": [0.5, -0.5], "feature_2": [1.0, -1.0], "class": [0, 1]}
    )
    return mock_preprocessor


@pytest.fixture
def mock_model():
    """Create a mock ML model for testing."""
    mock_model = Mock()
    mock_model.predict.return_value = np.array([0, 1])
    mock_model.predict_proba.return_value = np.array([[0.9, 0.1], [0.3, 0.7]])
    mock_model.feature_importances_ = np.array([0.6, 0.4])
    return mock_model


@pytest.fixture
def trained_model_trainer(sample_fraud_data):
    """Create a trained ModelTrainer for testing."""
    trainer = ModelTrainer()
    X_train = sample_fraud_data.drop("class", axis=1)
    y_train = sample_fraud_data["class"]

    # Mock the training process to avoid actual model training in tests
    with patch.object(trainer, "train_all_models") as mock_train:
        mock_train.return_value = {
            "random_forest": {
                "model": Mock(),
                "cv_scores": np.array([0.8, 0.85, 0.82, 0.88, 0.84]),
                "cv_mean": 0.838,
                "cv_std": 0.028,
            }
        }
        trainer.train_all_models(X_train, y_train)

    return trainer


# Configuration fixtures
@pytest.fixture
def test_config():
    """Provide test configuration."""
    return {
        "data_paths": {
            "raw_data_dir": "data/raw",
            "processed_data_dir": "data/processed",
        },
        "model": {
            "test_size": 0.2,
            "random_state": 42,
            "cv_folds": 3,  # Reduced for faster testing
        },
        "features": {"target_column": "class", "scaling_method": "standard"},
        "logging": {"level": "WARNING"},  # Reduce log noise in tests
    }


# Parametrized fixtures for different model types
@pytest.fixture(params=["random_forest", "xgboost", "logistic_regression"])
def model_type(request):
    """Parametrized fixture for different model types."""
    return request.param


@pytest.fixture(params=["smote", "undersample", "class_weights"])
def resampling_method(request):
    """Parametrized fixture for different resampling methods."""
    return request.param


# File system fixtures
@pytest.fixture
def create_test_csv(temp_data_dir, sample_fraud_data):
    """Create a test CSV file."""
    csv_path = temp_data_dir / "test_data.csv"
    sample_fraud_data.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def create_test_parquet(temp_data_dir, sample_fraud_data):
    """Create a test Parquet file."""
    parquet_path = temp_data_dir / "test_data.parquet"
    sample_fraud_data.to_parquet(parquet_path, index=False)
    return parquet_path


# Skip conditions
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line(
        "markers", "data_dependent: mark test as requiring real data files"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add unit marker to all tests by default
        if not any(
            marker.name in ["integration", "slow"] for marker in item.iter_markers()
        ):
            item.add_marker(pytest.mark.unit)

        # Add data_dependent marker to tests that use real data
        if "load_fraud_data" in item.name or "load_creditcard_data" in item.name:
            item.add_marker(pytest.mark.data_dependent)


# Utility functions for tests
def assert_dataframe_equal(df1, df2, check_dtype=True, check_index=True):
    """Assert that two DataFrames are equal with better error messages."""
    try:
        pd.testing.assert_frame_equal(
            df1, df2, check_dtype=check_dtype, check_index=check_index
        )
    except AssertionError as e:
        pytest.fail(f"DataFrames are not equal: {str(e)}")


def assert_model_performance(metrics, min_accuracy=0.5, min_f1=0.3):
    """Assert that model performance meets minimum thresholds."""
    assert (
        metrics.get("accuracy", 0) >= min_accuracy
    ), f"Accuracy {metrics.get('accuracy')} below threshold {min_accuracy}"
    assert (
        metrics.get("f1_score", 0) >= min_f1
    ), f"F1-score {metrics.get('f1_score')} below threshold {min_f1}"


# Environment setup
@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Set up test environment variables."""
    monkeypatch.setenv("PYTHONPATH", str(Path.cwd()))
    monkeypatch.setenv("TESTING", "1")
