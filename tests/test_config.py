"""
Tests for configuration management.
"""

import tempfile
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from src.config import (
    EVALUATION_METRICS,
    MODEL_HYPERPARAMS,
    ConfigManager,
    get_config,
    get_config_manager,
    validate_config,
)


class TestConfiguration:
    """Test cases for configuration management."""

    @pytest.mark.unit
    def test_get_config_returns_dict(self):
        """Test that get_config returns a dictionary."""
        config = get_config()
        assert isinstance(config, dict)
        assert "data_paths" in config
        assert "logging" in config
        assert "model" in config
        assert "features" in config
        assert "hyperparams" in config
        assert "evaluation" in config
        assert "explainability" in config

    @pytest.mark.unit
    def test_config_has_required_keys(self):
        """Test that configuration has all required keys."""
        config = get_config()

        # Check data paths
        assert "raw_data_dir" in config["data_paths"]
        assert "processed_data_dir" in config["data_paths"]

        # Check model config
        assert "test_size" in config["model"]
        assert "random_state" in config["model"]

        # Check features config
        assert "target_column" in config["features"]
        assert "scaling_method" in config["features"]

    @pytest.mark.unit
    def test_model_hyperparams_structure(self):
        """Test that model hyperparameters have correct structure."""
        assert isinstance(MODEL_HYPERPARAMS, dict)
        assert "random_forest" in MODEL_HYPERPARAMS
        assert "xgboost" in MODEL_HYPERPARAMS
        assert "logistic_regression" in MODEL_HYPERPARAMS

        # Check random forest params
        rf_params = MODEL_HYPERPARAMS["random_forest"]
        assert "n_estimators" in rf_params
        assert "max_depth" in rf_params
        assert isinstance(rf_params["n_estimators"], int)
        assert rf_params["n_estimators"] > 0

    @pytest.mark.unit
    def test_evaluation_metrics_list(self):
        """Test that evaluation metrics are properly defined."""
        assert isinstance(EVALUATION_METRICS, list)
        assert len(EVALUATION_METRICS) > 0
        assert "accuracy" in EVALUATION_METRICS
        assert "precision" in EVALUATION_METRICS
        assert "recall" in EVALUATION_METRICS
        assert "f1_score" in EVALUATION_METRICS

    @pytest.mark.unit
    def test_validate_config_success(self):
        """Test successful configuration validation."""
        with patch("src.config.RAW_DATA_DIR") as mock_raw_dir, patch(
            "src.config.PROCESSED_DATA_DIR"
        ) as mock_processed_dir:

            mock_raw_dir.exists.return_value = True
            mock_processed_dir.mkdir = lambda **kwargs: None
            mock_processed_dir.exists.return_value = False

            result = validate_config()
            assert result is True

    @pytest.mark.unit
    def test_validate_config_missing_raw_data_dir(self):
        """Test configuration validation with missing raw data directory."""
        with patch("src.config.RAW_DATA_DIR") as mock_raw_dir:
            mock_raw_dir.exists.return_value = False

            with pytest.raises(ValueError, match="Raw data directory does not exist"):
                validate_config()

    @pytest.mark.unit
    def test_validate_config_invalid_test_size(self):
        """Test configuration validation with invalid test size."""
        with patch("src.config.RAW_DATA_DIR") as mock_raw_dir, patch(
            "src.config.PROCESSED_DATA_DIR"
        ) as mock_processed_dir, patch("src.config.MODEL_CONFIG") as mock_model_config:

            mock_raw_dir.exists.return_value = True
            mock_processed_dir.exists.return_value = True
            mock_model_config.__getitem__.side_effect = lambda key: {
                "test_size": 0.8,
                "validation_size": 0.3,
            }[key]

            with pytest.raises(
                ValueError, match="Test size \\+ validation size must be less than 1.0"
            ):
                validate_config()

    @pytest.mark.unit
    def test_hyperparameter_validation(self):
        """Test that hyperparameters have valid values."""
        # Test Random Forest hyperparameters
        rf_params = MODEL_HYPERPARAMS["random_forest"]
        assert rf_params["n_estimators"] > 0
        assert rf_params["max_depth"] > 0
        assert rf_params["min_samples_split"] >= 2
        assert rf_params["min_samples_leaf"] >= 1

        # Test XGBoost hyperparameters
        xgb_params = MODEL_HYPERPARAMS["xgboost"]
        assert xgb_params["n_estimators"] > 0
        assert xgb_params["max_depth"] > 0
        assert 0 < xgb_params["learning_rate"] <= 1
        assert 0 < xgb_params["subsample"] <= 1

        # Test Logistic Regression hyperparameters
        lr_params = MODEL_HYPERPARAMS["logistic_regression"]
        assert lr_params["C"] > 0
        assert lr_params["max_iter"] > 0


class TestConfigurationEdgeCases:
    """Test edge cases and error conditions for configuration."""

    @pytest.mark.unit
    def test_config_immutability(self):
        """Test that modifying returned config doesn't affect original."""
        config1 = get_config()
        config2 = get_config()

        # Modify one config
        config1["model"]["test_size"] = 0.99

        # Check that the other config is unchanged
        assert config2["model"]["test_size"] != 0.99

    @pytest.mark.unit
    def test_config_with_environment_variables(self):
        """Test configuration with environment variable overrides."""
        with patch.dict("os.environ", {"FRAUD_DETECTION_LOG_LEVEL": "DEBUG"}):
            # This test would require implementing environment variable support
            # For now, just test that config loading doesn't break
            config = get_config()
            assert config is not None

    @pytest.mark.unit
    def test_config_paths_are_strings(self):
        """Test that all path configurations are strings."""
        config = get_config()

        assert isinstance(config["data_paths"]["raw_data_dir"], str)
        assert isinstance(config["data_paths"]["processed_data_dir"], str)
        assert isinstance(config["logging"]["log_dir"], str)

    @pytest.mark.unit
    def test_hyperparams_type_validation(self):
        """Test that hyperparameters have correct types."""
        for model_name, params in MODEL_HYPERPARAMS.items():
            for param_name, param_value in params.items():
                # Check that numeric parameters are actually numeric
                if param_name in [
                    "n_estimators",
                    "max_depth",
                    "min_samples_split",
                    "min_samples_leaf",
                    "max_iter",
                ]:
                    assert isinstance(
                        param_value, int
                    ), f"{model_name}.{param_name} should be int"
                elif param_name in [
                    "learning_rate",
                    "subsample",
                    "colsample_bytree",
                    "reg_alpha",
                    "reg_lambda",
                    "C",
                    "scale_pos_weight",
                ]:
                    assert isinstance(
                        param_value, (int, float)
                    ), f"{model_name}.{param_name} should be numeric"


class TestConfigManager:
    """Test cases for the new ConfigManager class."""

    @pytest.mark.unit
    def test_config_manager_initialization(self):
        """Test ConfigManager initialization."""
        config_manager = ConfigManager()
        assert config_manager is not None
        assert not config_manager.is_validated()

    @pytest.mark.unit
    def test_config_manager_get_with_dot_notation(self):
        """Test ConfigManager get method with dot notation."""
        config_manager = ConfigManager()

        # Test nested access
        test_size = config_manager.get("model.test_size")
        assert test_size == 0.2

        # Test with default
        non_existent = config_manager.get("non.existent.key", "default")
        assert non_existent == "default"

    @pytest.mark.unit
    def test_config_manager_set_with_dot_notation(self):
        """Test ConfigManager set method with dot notation."""
        config_manager = ConfigManager()

        # Set a value
        config_manager.set("model.test_size", 0.3)
        assert config_manager.get("model.test_size") == 0.3

        # Set a nested value
        config_manager.set("new.nested.key", "value")
        assert config_manager.get("new.nested.key") == "value"

    @pytest.mark.unit
    def test_config_manager_get_model_config(self):
        """Test ConfigManager get_model_config method."""
        config_manager = ConfigManager()

        rf_config = config_manager.get_model_config("random_forest")
        assert "n_estimators" in rf_config
        assert "test_size" in rf_config  # Should include general model config

        # Test invalid model
        with pytest.raises(ValueError, match="Unsupported model"):
            config_manager.get_model_config("invalid_model")

    @pytest.mark.unit
    def test_config_manager_environment_override(self):
        """Test ConfigManager environment variable override."""
        with patch.dict("os.environ", {"FRAUD_DETECTION_MODEL_TEST_SIZE": "0.4"}):
            config_manager = ConfigManager()
            test_size = config_manager.get("model.test_size")
            assert test_size == 0.4

    @pytest.mark.unit
    def test_config_manager_validation(self):
        """Test ConfigManager validation."""
        config_manager = ConfigManager()

        # Mock the paths to exist
        with patch("pathlib.Path.exists", return_value=True):
            result = config_manager.validate()
            assert result is True
            assert config_manager.is_validated()

    @pytest.mark.unit
    def test_config_manager_update_from_dict(self):
        """Test ConfigManager update_from_dict method."""
        config_manager = ConfigManager()

        updates = {"model": {"test_size": 0.25}, "logging": {"log_level": "DEBUG"}}

        config_manager.update_from_dict(updates)

        assert config_manager.get("model.test_size") == 0.25
        assert config_manager.get("logging.log_level") == "DEBUG"

    @pytest.mark.unit
    def test_global_config_manager(self):
        """Test global ConfigManager instance."""
        cm1 = get_config_manager()
        cm2 = get_config_manager()

        # Should return the same instance
        assert cm1 is cm2
