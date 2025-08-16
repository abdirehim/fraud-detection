"""
Performance tests for the fraud detection pipeline.
"""

import time

import numpy as np
import pandas as pd
import pytest

from src.data_loader import DataLoader
from src.preprocess import DataPreprocessor


class TestPerformance:
    """Performance test cases."""

    @pytest.fixture
    def large_dataset(self):
        """Create a large dataset for performance testing."""
        np.random.seed(42)
        n_samples = 10000
        n_features = 50

        data = {
            "user_id": [f"user_{i}" for i in range(n_samples)],
            "purchase_value": np.random.exponential(100, n_samples),
            "age": np.random.randint(18, 80, n_samples),
            "class": np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        }

        # Add random features
        for i in range(n_features):
            data[f"feature_{i}"] = np.random.randn(n_samples)

        return pd.DataFrame(data)

    @pytest.mark.performance
    def test_data_loading_performance(self, large_dataset, tmp_path):
        """Test data loading performance."""
        # Save test data
        test_file = tmp_path / "test_data.csv"
        large_dataset.to_csv(test_file, index=False)

        loader = DataLoader(data_dir=tmp_path)

        start_time = time.time()
        df = loader.load_csv_data("test_data.csv")
        load_time = time.time() - start_time

        assert len(df) == len(large_dataset)
        assert load_time < 5.0  # Should load within 5 seconds

    @pytest.mark.performance
    def test_preprocessing_performance(self, large_dataset):
        """Test preprocessing performance."""
        preprocessor = DataPreprocessor()

        start_time = time.time()
        processed_df = preprocessor.fit_transform(large_dataset, "class")
        process_time = time.time() - start_time

        assert len(processed_df) > 0
        assert process_time < 30.0  # Should process within 30 seconds

    @pytest.mark.performance
    def test_memory_usage(self, large_dataset):
        """Test memory usage during processing."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        preprocessor = DataPreprocessor()
        processed_df = preprocessor.fit_transform(large_dataset, "class")

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        assert memory_increase < 500  # Should not use more than 500MB additional memory
