"""
Custom exceptions for the fraud detection pipeline.
"""


class FraudDetectionError(Exception):
    """Base exception for fraud detection pipeline."""

    pass


class DataValidationError(FraudDetectionError):
    """Raised when data validation fails."""

    pass


class ModelTrainingError(FraudDetectionError):
    """Raised when model training fails."""

    pass


class ConfigurationError(FraudDetectionError):
    """Raised when configuration is invalid."""

    pass


class PredictionError(FraudDetectionError):
    """Raised when prediction fails."""

    pass
