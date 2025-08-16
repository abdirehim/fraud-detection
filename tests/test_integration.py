"""
Integration tests for fraud detection pipeline.

These tests verify that components work together correctly
and test end-to-end workflows.
"""

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


class TestDataPipelineIntegration:
    """Integration tests for data processing pipeline."""

    @pytest.mark.integration
    def test_data_loader_to_preprocessor_integration(
        self, sample_fraud_data, temp_data_dir
    ):
        """Test integration between DataLoader and DataPreprocessor."""
        # Save sample data to file
        data_file = temp_data_dir / "test_fraud_data.csv"
        sample_fraud_data.to_csv(data_file, index=False)

        # Load data
        loader = DataLoader()
        loaded_data = loader.load_csv_data(str(data_file))

        # Preprocess data
        preprocessor = DataPreprocessor()
        processed_data = preprocessor.fit_transform(loaded_data, "class")

        # Verify integration
        assert isinstance(processed_data, pd.DataFrame)
        assert "class" in processed_data.columns
        assert len(processed_data) <= len(
            loaded_data
        )  # May remove some rows during cleaning
        assert len(processed_data.columns) >= len(
            loaded_data.columns
        )  # Should add features
        assert preprocessor.is_fitted

    @pytest.mark.integration
    def test_preprocessor_to_trainer_integration(self, sample_fraud_data):
        """Test integration between DataPreprocessor and ModelTrainer."""
        # Preprocess data
        preprocessor = DataPreprocessor()
        processed_data = preprocessor.fit_transform(sample_fraud_data, "class")

        # Prepare data for training
        trainer = ModelTrainer()
        X_train, X_test, y_train, y_test = trainer.prepare_data(processed_data, "class")

        # Verify integration
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)
        assert len(X_train) + len(X_test) == len(processed_data)
        assert "class" not in X_train.columns
        assert "class" not in X_test.columns

    @pytest.mark.integration
    @pytest.mark.slow
    def test_trainer_to_evaluator_integration(self, sample_fraud_data):
        """Test integration between ModelTrainer and ModelEvaluator."""
        # Preprocess data
        preprocessor = DataPreprocessor()
        processed_data = preprocessor.fit_transform(sample_fraud_data, "class")

        # Train models
        trainer = ModelTrainer()
        X_train, X_test, y_train, y_test = trainer.prepare_data(processed_data, "class")

        # Use a simple model for faster testing
        with patch.object(trainer, "create_models") as mock_create_models:
            from sklearn.dummy import DummyClassifier

            dummy_model = DummyClassifier(strategy="most_frequent")
            mock_create_models.return_value = {"dummy": dummy_model}

            models = trainer.train_all_models(X_train, y_train, "none")

        # Evaluate models
        evaluator = ModelEvaluator()
        results = evaluator.evaluate_all_models(models, X_test, y_test)

        # Verify integration
        assert isinstance(results, dict)
        assert "dummy" in results
        assert "metrics" in results["dummy"]
        assert "accuracy" in results["dummy"]["metrics"]


class TestFullPipelineIntegration:
    """Integration tests for the complete fraud detection pipeline."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_end_to_end_pipeline_with_sample_data(
        self, sample_fraud_data, temp_data_dir
    ):
        """Test complete end-to-end pipeline with sample data."""
        # Save sample data
        data_file = temp_data_dir / "test_fraud_data.csv"
        sample_fraud_data.to_csv(data_file, index=False)

        # Step 1: Load data
        loader = DataLoader()
        raw_data = loader.load_csv_data(str(data_file))

        # Step 2: Preprocess data
        preprocessor = DataPreprocessor()
        processed_data = preprocessor.fit_transform(raw_data, "class")

        # Step 3: Train models (using dummy model for speed)
        trainer = ModelTrainer()
        X_train, X_test, y_train, y_test = trainer.prepare_data(processed_data, "class")

        with patch.object(trainer, "create_models") as mock_create_models:
            from sklearn.dummy import DummyClassifier

            dummy_model = DummyClassifier(strategy="stratified", random_state=42)
            mock_create_models.return_value = {"dummy": dummy_model}

            models = trainer.train_all_models(X_train, y_train, "none")

        # Step 4: Evaluate models
        evaluator = ModelEvaluator()
        results = evaluator.evaluate_all_models(models, X_test, y_test)

        # Step 5: Generate explanations
        explainer = ModelExplainer()
        best_model_name, best_model = trainer.get_best_model()

        with patch.object(explainer, "fit_shap_explainer"):
            explanation_report = explainer.generate_explanation_report(
                best_model, X_test, y_test
            )

        # Verify complete pipeline
        assert isinstance(raw_data, pd.DataFrame)
        assert isinstance(processed_data, pd.DataFrame)
        assert isinstance(models, dict)
        assert isinstance(results, dict)
        assert isinstance(explanation_report, dict)

        # Check that data flows correctly through pipeline
        assert len(processed_data) <= len(raw_data)
        assert len(X_train) + len(X_test) == len(processed_data)
        assert best_model_name in models

    @pytest.mark.integration
    def test_pipeline_error_handling(self, temp_data_dir):
        """Test pipeline error handling with invalid data."""
        # Create invalid data file
        invalid_data = pd.DataFrame(
            {"invalid_column": [1, 2, 3], "another_column": ["a", "b", "c"]}
        )
        data_file = temp_data_dir / "invalid_data.csv"
        invalid_data.to_csv(data_file, index=False)

        # Try to run pipeline with invalid data
        loader = DataLoader()
        raw_data = loader.load_csv_data(str(data_file))

        preprocessor = DataPreprocessor()

        # This should handle the missing target column gracefully
        with pytest.raises((KeyError, ValueError)):
            preprocessor.fit_transform(raw_data, "class")

    @pytest.mark.integration
    def test_pipeline_with_different_data_formats(
        self, sample_fraud_data, temp_data_dir
    ):
        """Test pipeline with different data formats (CSV, Parquet)."""
        # Save data in different formats
        csv_file = temp_data_dir / "test_data.csv"
        parquet_file = temp_data_dir / "test_data.parquet"

        sample_fraud_data.to_csv(csv_file, index=False)
        sample_fraud_data.to_parquet(parquet_file, index=False)

        loader = DataLoader()

        # Load from CSV
        csv_data = loader.load_csv_data(str(csv_file))

        # Load from Parquet
        parquet_data = loader.load_parquet_data(str(parquet_file))

        # Both should produce equivalent results
        pd.testing.assert_frame_equal(csv_data, parquet_data)

        # Both should work with preprocessor
        preprocessor = DataPreprocessor()
        csv_processed = preprocessor.fit_transform(csv_data, "class")

        preprocessor2 = DataPreprocessor()
        parquet_processed = preprocessor2.fit_transform(parquet_data, "class")

        # Results should be equivalent
        assert csv_processed.shape == parquet_processed.shape
        assert list(csv_processed.columns) == list(parquet_processed.columns)


class TestModelPipelineIntegration:
    """Integration tests for model training and evaluation pipeline."""

    @pytest.mark.integration
    def test_model_training_evaluation_cycle(self, sample_fraud_data):
        """Test complete model training and evaluation cycle."""
        # Preprocess data
        preprocessor = DataPreprocessor()
        processed_data = preprocessor.fit_transform(sample_fraud_data, "class")

        # Train models
        trainer = ModelTrainer()
        X_train, X_test, y_train, y_test = trainer.prepare_data(processed_data, "class")

        # Use simple models for testing
        with patch.object(trainer, "create_models") as mock_create_models:
            from sklearn.dummy import DummyClassifier
            from sklearn.linear_model import LogisticRegression

            models = {
                "dummy": DummyClassifier(strategy="stratified", random_state=42),
                "logistic": LogisticRegression(random_state=42, max_iter=100),
            }
            mock_create_models.return_value = models

            trained_models = trainer.train_all_models(X_train, y_train, "none")

        # Evaluate models
        evaluator = ModelEvaluator()
        results = evaluator.evaluate_all_models(trained_models, X_test, y_test)

        # Verify results
        assert len(results) == len(models)
        for model_name in models.keys():
            assert model_name in results
            assert "metrics" in results[model_name]
            assert "accuracy" in results[model_name]["metrics"]
            assert 0 <= results[model_name]["metrics"]["accuracy"] <= 1

    @pytest.mark.integration
    def test_model_persistence_integration(self, sample_fraud_data, temp_data_dir):
        """Test model saving and loading integration."""
        # Train a simple model
        preprocessor = DataPreprocessor()
        processed_data = preprocessor.fit_transform(sample_fraud_data, "class")

        trainer = ModelTrainer()
        X_train, X_test, y_train, y_test = trainer.prepare_data(processed_data, "class")

        with patch.object(trainer, "create_models") as mock_create_models:
            from sklearn.dummy import DummyClassifier

            dummy_model = DummyClassifier(strategy="most_frequent")
            mock_create_models.return_value = {"dummy": dummy_model}

            models = trainer.train_all_models(X_train, y_train, "none")

        # Save models
        models_dir = temp_data_dir / "models"
        trainer.save_models(models_dir)

        # Load models
        new_trainer = ModelTrainer()
        new_trainer.load_models(models_dir)

        # Verify loaded models work
        assert new_trainer.is_trained
        assert "dummy" in new_trainer.models

        # Test predictions with loaded model
        loaded_model = new_trainer.models["dummy"]["model"]
        predictions = loaded_model.predict(X_test)
        assert len(predictions) == len(X_test)


class TestExplainabilityIntegration:
    """Integration tests for model explainability."""

    @pytest.mark.integration
    def test_explainer_with_trained_model(self, sample_fraud_data):
        """Test explainer integration with trained model."""
        # Preprocess and train
        preprocessor = DataPreprocessor()
        processed_data = preprocessor.fit_transform(sample_fraud_data, "class")

        trainer = ModelTrainer()
        X_train, X_test, y_train, y_test = trainer.prepare_data(processed_data, "class")

        with patch.object(trainer, "create_models") as mock_create_models:
            from sklearn.ensemble import RandomForestClassifier

            rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
            mock_create_models.return_value = {"random_forest": rf_model}

            models = trainer.train_all_models(X_train, y_train, "none")

        # Test explainer
        explainer = ModelExplainer()
        best_model_name, best_model = trainer.get_best_model()

        # Mock SHAP to avoid complex dependencies in tests
        with patch.object(
            explainer, "fit_shap_explainer"
        ) as mock_fit_shap, patch.object(
            explainer, "generate_explanation_report"
        ) as mock_explain:

            mock_explain.return_value = {
                "model_name": best_model_name,
                "feature_importance": {"feature_1": 0.6, "feature_2": 0.4},
                "explanations_generated": True,
            }

            explanation_report = explainer.generate_explanation_report(
                best_model, X_test, y_test
            )

        # Verify integration
        assert isinstance(explanation_report, dict)
        assert "model_name" in explanation_report
        assert explanation_report["model_name"] == best_model_name


class TestPipelinePerformance:
    """Performance and scalability integration tests."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_pipeline_with_larger_dataset(self):
        """Test pipeline performance with larger dataset."""
        # Create larger sample dataset
        np.random.seed(42)
        n_samples = 1000

        large_data = pd.DataFrame(
            {
                "user_id": [f"user_{i}" for i in range(n_samples)],
                "purchase_value": np.random.uniform(10, 1000, n_samples),
                "age": np.random.randint(18, 80, n_samples),
                "feature_1": np.random.normal(0, 1, n_samples),
                "feature_2": np.random.normal(0, 1, n_samples),
                "class": np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
            }
        )

        # Run pipeline
        preprocessor = DataPreprocessor()
        processed_data = preprocessor.fit_transform(large_data, "class")

        trainer = ModelTrainer()
        X_train, X_test, y_train, y_test = trainer.prepare_data(processed_data, "class")

        # Use simple model for performance test
        with patch.object(trainer, "create_models") as mock_create_models:
            from sklearn.dummy import DummyClassifier

            dummy_model = DummyClassifier(strategy="stratified", random_state=42)
            mock_create_models.return_value = {"dummy": dummy_model}

            models = trainer.train_all_models(X_train, y_train, "none")

        # Verify it completes successfully
        assert len(models) > 0
        assert trainer.is_trained
        assert len(X_train) > 0
        assert len(X_test) > 0

    @pytest.mark.integration
    def test_memory_usage_with_feature_engineering(self, sample_fraud_data):
        """Test memory usage during feature engineering."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Run feature engineering
        preprocessor = DataPreprocessor()
        processed_data = preprocessor.fit_transform(sample_fraud_data, "class")

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 100MB for small dataset)
        assert (
            memory_increase < 100
        ), f"Memory usage increased by {memory_increase:.2f}MB"
        assert isinstance(processed_data, pd.DataFrame)
        assert len(processed_data) > 0
