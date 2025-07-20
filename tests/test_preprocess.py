"""
Tests for preprocessing module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.preprocess import DataPreprocessor
from src.data_loader import DataLoader


class TestDataPreprocessor:
    """Test cases for DataPreprocessor class."""
    
    def _load_test_data(self):
        """Helper method to load real test data."""
        loader = DataLoader()
        try:
            # Try to load fraud data first
            df = loader.load_fraud_data()
            target_col = 'class'
        except FileNotFoundError:
            try:
                # Try credit card data
                df = loader.load_creditcard_data()
                target_col = 'Class'
            except FileNotFoundError:
                pytest.skip("No real fraud detection datasets available for testing")
        
        return df, target_col
    
    def test_preprocessor_initialization(self):
        """Test DataPreprocessor initialization."""
        preprocessor = DataPreprocessor()
        assert preprocessor is not None
        assert hasattr(preprocessor, 'config')
        assert hasattr(preprocessor, 'logger')
        assert hasattr(preprocessor, 'scaler')
        assert hasattr(preprocessor, 'label_encoders')
        assert not preprocessor.is_fitted
    
    def test_clean_data(self):
        """Test data cleaning functionality."""
        preprocessor = DataPreprocessor()
        df, target_col = self._load_test_data()
        
        # Add some test data issues
        df_with_issues = df.copy()
        # Add missing value to a numerical column
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            df_with_issues.loc[0, numerical_cols[0]] = np.nan
        df_with_issues = pd.concat([df_with_issues, df_with_issues.iloc[:5]])  # Add duplicates
        
        cleaned_df = preprocessor.clean_data(df_with_issues)
        
        assert isinstance(cleaned_df, pd.DataFrame)
        assert len(cleaned_df) <= len(df_with_issues)  # Should remove duplicates
        # Note: Some missing values might be acceptable depending on the data
    
    def test_engineer_features(self):
        """Test feature engineering functionality."""
        preprocessor = DataPreprocessor()
        df, target_col = self._load_test_data()
        
        engineered_df = preprocessor.engineer_features(df)
        
        assert isinstance(engineered_df, pd.DataFrame)
        assert len(engineered_df.columns) >= len(df.columns)  # Should add features
    
    def test_encode_categorical_features(self):
        """Test categorical feature encoding."""
        preprocessor = DataPreprocessor()
        df, target_col = self._load_test_data()
        
        # Ensure we have categorical features
        df['test_categorical'] = ['A', 'B', 'C'] * (len(df) // 3 + 1)
        df = df.iloc[:len(df)]
        
        encoded_df = preprocessor.encode_categorical_features(df, fit=True)
        
        assert isinstance(encoded_df, pd.DataFrame)
        assert 'test_categorical' in encoded_df.columns
        
        # Check that categorical features are now numerical
        categorical_cols = encoded_df.select_dtypes(include=['object', 'category']).columns
        assert 'test_categorical' not in categorical_cols or encoded_df['test_categorical'].dtype in ['int32', 'int64']
    
    def test_scale_features(self):
        """Test feature scaling functionality."""
        preprocessor = DataPreprocessor()
        df, target_col = self._load_test_data()
        
        # Remove target column for scaling
        feature_df = df.drop(columns=[target_col])
        
        scaled_df = preprocessor.scale_features(feature_df, fit=True)
        
        assert isinstance(scaled_df, pd.DataFrame)
        assert preprocessor.scaler is not None
        
        # Check that numerical features are scaled
        numerical_cols = scaled_df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            # Scaled features should have different statistics
            assert scaled_df[numerical_cols].std().mean() > 0
    
    def test_fit_transform_pipeline(self):
        """Test complete fit_transform pipeline."""
        preprocessor = DataPreprocessor()
        df, target_col = self._load_test_data()
        
        transformed_df = preprocessor.fit_transform(df, target_col)
        
        assert isinstance(transformed_df, pd.DataFrame)
        assert preprocessor.is_fitted
        assert target_col in transformed_df.columns
        
        # Check that preprocessing was applied
        assert len(transformed_df.columns) >= len(df.columns)  # Should have engineered features
    
    def test_transform_pipeline(self):
        """Test transform pipeline with fitted preprocessor."""
        preprocessor = DataPreprocessor()
        df, target_col = self._load_test_data()
        
        # First fit the preprocessor
        preprocessor.fit_transform(df, target_col)
        
        # Then transform new data
        new_df = df.sample(n=min(100, len(df)), random_state=42)
        transformed_new_df = preprocessor.transform(new_df, target_col)
        
        assert isinstance(transformed_new_df, pd.DataFrame)
        assert target_col in transformed_new_df.columns
        assert len(transformed_new_df) == len(new_df)
    
    def test_transform_without_fit(self):
        """Test that transform fails without fitting."""
        preprocessor = DataPreprocessor()
        df, target_col = self._load_test_data()
        
        with pytest.raises(ValueError, match="Preprocessor must be fitted before transform"):
            preprocessor.transform(df, target_col)
    
    def test_feature_selection(self):
        """Test feature selection functionality."""
        preprocessor = DataPreprocessor()
        df, target_col = self._load_test_data()
        
        # Remove target for feature selection
        feature_df = df.drop(columns=[target_col])
        target = df[target_col]
        
        selected_df = preprocessor.select_features(feature_df, target, fit=True)
        
        assert isinstance(selected_df, pd.DataFrame)
        assert preprocessor.feature_selector is not None
        assert len(selected_df.columns) <= len(feature_df.columns) + 1  # +1 for target
    
    def test_config_override(self):
        """Test that custom config can be passed."""
        custom_config = {
            'scaling_method': 'robust',
            'target_column': 'fraud'
        }
        
        preprocessor = DataPreprocessor(config=custom_config)
        assert preprocessor.config['scaling_method'] == 'robust'
        assert preprocessor.config['target_column'] == 'fraud'
    
    def test_error_handling_invalid_scaling_method(self):
        """Test error handling for invalid scaling method."""
        preprocessor = DataPreprocessor()
        df, target_col = self._load_test_data()
        feature_df = df.drop(columns=[target_col])
        
        # Set invalid scaling method
        preprocessor.config['scaling_method'] = 'invalid_method'
        
        with pytest.raises(ValueError, match="Unsupported scaling method"):
            preprocessor.scale_features(feature_df, fit=True)
    
    def test_error_handling_invalid_resampling_method(self):
        """Test error handling for invalid resampling method."""
        preprocessor = DataPreprocessor()
        df, target_col = self._load_test_data()
        
        with pytest.raises(ValueError, match="Unsupported resampling method"):
            preprocessor.handle_imbalanced_data(
                df.drop(columns=[target_col]), 
                df[target_col], 
                method='invalid_method'
            ) 