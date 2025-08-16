"""
Tests for utility functions.
"""

import json
import logging
import pickle
import tempfile
from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import pytest

from src.utils import (
    create_experiment_dir,
    ensure_directory_exists,
    load_model,
    save_metrics,
    save_model,
    setup_logging,
)


class TestLogging:
    """Test cases for logging utilities."""

    @pytest.mark.unit
    def test_setup_logging_returns_logger(self):
        """Test that setup_logging returns a logger instance."""
        logger = setup_logging("test_logger")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_logger"

    @pytest.mark.unit
    def test_setup_logging_with_custom_level(self):
        """Test setup_logging with custom log level."""
        logger = setup_logging("test_logger", level="DEBUG")
        assert logger.level <= logging.DEBUG

    @pytest.mark.unit
    def test_setup_logging_creates_log_file(self):
        """Test that setup_logging creates log file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"
            logger = setup_logging("test_logger", log_file=str(log_file))

            # Log a message to trigger file creation
            logger.info("Test message")

            # Check if log file was created
            assert log_file.exists()

    @pytest.mark.unit
    def test_setup_logging_multiple_calls_same_name(self):
        """Test that multiple calls with same name return same logger."""
        logger1 = setup_logging("same_name")
        logger2 = setup_logging("same_name")
        assert logger1 is logger2


class TestModelPersistence:
    """Test cases for model saving and loading."""

    @pytest.mark.unit
    def test_save_model_creates_file(self, mock_model, temp_data_dir):
        """Test that save_model creates a pickle file."""
        model_path = temp_data_dir / "test_model.pkl"

        result = save_model(mock_model, model_path, "test_model")

        assert result is True
        assert model_path.exists()

    @pytest.mark.unit
    def test_save_model_creates_directory(self, mock_model):
        """Test that save_model creates directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "subdir" / "test_model.pkl"

            result = save_model(mock_model, model_path, "test_model")

            assert result is True
            assert model_path.exists()

    @pytest.mark.unit
    def test_save_model_handles_errors(self, temp_data_dir):
        """Test that save_model handles errors gracefully."""
        model_path = temp_data_dir / "test_model.pkl"

        # Try to save an unpicklable object
        unpicklable_object = lambda x: x  # Lambda functions can't be pickled

        result = save_model(unpicklable_object, model_path, "test_model")

        assert result is False
        assert not model_path.exists()

    @pytest.mark.unit
    def test_load_model_success(self, mock_model, temp_data_dir):
        """Test successful model loading."""
        model_path = temp_data_dir / "test_model.pkl"

        # First save a model
        save_model(mock_model, model_path, "test_model")

        # Then load it
        loaded_model = load_model(model_path, "test_model")

        assert loaded_model is not None
        # Check that it has the same attributes as the mock
        assert hasattr(loaded_model, "predict")
        assert hasattr(loaded_model, "predict_proba")

    @pytest.mark.unit
    def test_load_model_file_not_found(self, temp_data_dir):
        """Test load_model with non-existent file."""
        model_path = temp_data_dir / "nonexistent_model.pkl"

        loaded_model = load_model(model_path, "test_model")

        assert loaded_model is None

    @pytest.mark.unit
    def test_load_model_corrupted_file(self, temp_data_dir):
        """Test load_model with corrupted file."""
        model_path = temp_data_dir / "corrupted_model.pkl"

        # Create a corrupted pickle file
        with open(model_path, "w") as f:
            f.write("This is not a valid pickle file")

        loaded_model = load_model(model_path, "test_model")

        assert loaded_model is None


class TestMetricsPersistence:
    """Test cases for metrics saving."""

    @pytest.mark.unit
    def test_save_metrics_creates_json_file(self, temp_data_dir):
        """Test that save_metrics creates a JSON file."""
        metrics = {
            "accuracy": 0.95,
            "precision": 0.87,
            "recall": 0.92,
            "f1_score": 0.89,
        }
        metrics_path = temp_data_dir / "test_metrics.json"

        result = save_metrics(metrics, metrics_path)

        assert result is True
        assert metrics_path.exists()

        # Verify content
        with open(metrics_path, "r") as f:
            saved_metrics = json.load(f)

        assert saved_metrics == metrics

    @pytest.mark.unit
    def test_save_metrics_handles_numpy_types(self, temp_data_dir):
        """Test that save_metrics handles numpy data types."""
        import numpy as np

        metrics = {
            "accuracy": np.float64(0.95),
            "precision": np.float32(0.87),
            "count": np.int64(100),
        }
        metrics_path = temp_data_dir / "test_metrics.json"

        result = save_metrics(metrics, metrics_path)

        assert result is True
        assert metrics_path.exists()

    @pytest.mark.unit
    def test_save_metrics_creates_directory(self):
        """Test that save_metrics creates directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics_path = Path(temp_dir) / "subdir" / "test_metrics.json"
            metrics = {"accuracy": 0.95}

            result = save_metrics(metrics, metrics_path)

            assert result is True
            assert metrics_path.exists()

    @pytest.mark.unit
    def test_save_metrics_handles_errors(self, temp_data_dir):
        """Test that save_metrics handles errors gracefully."""
        metrics_path = temp_data_dir / "test_metrics.json"

        # Try to save an unserializable object
        metrics = {"model": lambda x: x}  # Lambda functions can't be JSON serialized

        result = save_metrics(metrics, metrics_path)

        assert result is False
        assert not metrics_path.exists()


class TestDirectoryUtilities:
    """Test cases for directory utilities."""

    @pytest.mark.unit
    def test_ensure_directory_exists_creates_directory(self):
        """Test that ensure_directory_exists creates directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            new_dir = Path(temp_dir) / "new_directory"

            result = ensure_directory_exists(new_dir)

            assert result is True
            assert new_dir.exists()
            assert new_dir.is_dir()

    @pytest.mark.unit
    def test_ensure_directory_exists_with_existing_directory(self, temp_data_dir):
        """Test ensure_directory_exists with existing directory."""
        result = ensure_directory_exists(temp_data_dir)

        assert result is True
        assert temp_data_dir.exists()

    @pytest.mark.unit
    def test_ensure_directory_exists_with_file_path(self, temp_data_dir):
        """Test ensure_directory_exists when path is a file."""
        file_path = temp_data_dir / "test_file.txt"
        file_path.write_text("test content")

        result = ensure_directory_exists(file_path)

        # Should fail because path exists but is not a directory
        assert result is False

    @pytest.mark.unit
    def test_create_experiment_dir_creates_timestamped_directory(self):
        """Test that create_experiment_dir creates timestamped directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)

            experiment_dir = create_experiment_dir(base_dir, "test_experiment")

            assert experiment_dir.exists()
            assert experiment_dir.is_dir()
            assert "test_experiment" in experiment_dir.name
            assert experiment_dir.parent == base_dir

    @pytest.mark.unit
    def test_create_experiment_dir_creates_subdirectories(self):
        """Test that create_experiment_dir creates expected subdirectories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)

            experiment_dir = create_experiment_dir(base_dir, "test_experiment")

            # Check for expected subdirectories
            expected_subdirs = ["models", "evaluation", "logs"]
            for subdir in expected_subdirs:
                subdir_path = experiment_dir / subdir
                assert subdir_path.exists(), f"Subdirectory {subdir} should exist"
                assert (
                    subdir_path.is_dir()
                ), f"Subdirectory {subdir} should be a directory"

    @pytest.mark.unit
    def test_create_experiment_dir_unique_names(self):
        """Test that create_experiment_dir creates unique directory names."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)

            # Create two experiment directories quickly
            dir1 = create_experiment_dir(base_dir, "test_experiment")
            dir2 = create_experiment_dir(base_dir, "test_experiment")

            assert dir1 != dir2
            assert dir1.exists()
            assert dir2.exists()


class TestUtilityEdgeCases:
    """Test edge cases and error conditions for utilities."""

    @pytest.mark.unit
    def test_setup_logging_with_invalid_level(self):
        """Test setup_logging with invalid log level."""
        # Should not raise an exception, but use default level
        logger = setup_logging("test_logger", level="INVALID_LEVEL")
        assert isinstance(logger, logging.Logger)

    @pytest.mark.unit
    def test_save_model_with_none_model(self, temp_data_dir):
        """Test save_model with None model."""
        model_path = temp_data_dir / "test_model.pkl"

        result = save_model(None, model_path, "test_model")

        assert result is False
        assert not model_path.exists()

    @pytest.mark.unit
    def test_save_metrics_with_empty_dict(self, temp_data_dir):
        """Test save_metrics with empty dictionary."""
        metrics_path = temp_data_dir / "test_metrics.json"

        result = save_metrics({}, metrics_path)

        assert result is True
        assert metrics_path.exists()

        with open(metrics_path, "r") as f:
            saved_metrics = json.load(f)

        assert saved_metrics == {}

    @pytest.mark.unit
    @patch("src.utils.Path.mkdir")
    def test_ensure_directory_exists_permission_error(self, mock_mkdir):
        """Test ensure_directory_exists with permission error."""
        mock_mkdir.side_effect = PermissionError("Permission denied")

        result = ensure_directory_exists(Path("/fake/path"))

        assert result is False
