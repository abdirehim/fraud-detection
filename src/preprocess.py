"""
Data preprocessing and feature engineering for fraud detection.

This module handles data cleaning, feature engineering, and preparation
for machine learning models in fraud detection tasks.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from typing import Tuple, Dict, Any, Optional, List
import logging

from .config import FEATURE_CONFIG
from .utils import setup_logging


class DataPreprocessor:
    """
    Data preprocessor for fraud detection datasets.
    
    This class handles data cleaning, feature engineering, and preparation
    for machine learning models.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the DataPreprocessor.
        
        Args:
            config: Configuration dictionary for preprocessing
        """
        self.config = config or FEATURE_CONFIG
        self.logger = setup_logging("preprocessor")
        self.scaler = None
        self.label_encoders = {}
        self.imputer = None
        self.feature_selector = None
        self.is_fitted = False
        
        self.logger.info("Initialized DataPreprocessor")
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the input dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        try:
            self.logger.info("Starting data cleaning process")
            df_clean = df.copy()
            
            # Remove duplicate rows
            initial_rows = len(df_clean)
            df_clean = df_clean.drop_duplicates()
            removed_duplicates = initial_rows - len(df_clean)
            self.logger.info(f"Removed {removed_duplicates} duplicate rows")
            
            # Handle missing values
            missing_counts = df_clean.isnull().sum()
            if missing_counts.sum() > 0:
                self.logger.info(f"Missing values found: {missing_counts[missing_counts > 0].to_dict()}")
                
                # For numerical columns, fill with median
                numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
                for col in numerical_cols:
                    if df_clean[col].isnull().sum() > 0:
                        median_val = df_clean[col].median()
                        df_clean[col].fillna(median_val, inplace=True)
                        self.logger.info(f"Filled missing values in {col} with median: {median_val}")
                
                # For categorical columns, fill with mode
                categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns
                for col in categorical_cols:
                    if df_clean[col].isnull().sum() > 0:
                        mode_val = df_clean[col].mode()[0]
                        df_clean[col].fillna(mode_val, inplace=True)
                        self.logger.info(f"Filled missing values in {col} with mode: {mode_val}")
            
            # Remove columns with too many missing values
            high_missing_cols = df_clean.columns[df_clean.isnull().sum() > len(df_clean) * 0.5]
            if len(high_missing_cols) > 0:
                df_clean = df_clean.drop(columns=high_missing_cols)
                self.logger.info(f"Removed columns with high missing values: {list(high_missing_cols)}")
            
            self.logger.info(f"Data cleaning completed. Final shape: {df_clean.shape}")
            return df_clean
            
        except Exception as e:
            self.logger.error(f"Error during data cleaning: {str(e)}")
            raise
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features for fraud detection.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        try:
            self.logger.info("Starting feature engineering")
            df_engineered = df.copy()
            
            # Time-based features
            if 'hour_of_day' in df_engineered.columns:
                df_engineered['is_night'] = (df_engineered['hour_of_day'] >= 22) | (df_engineered['hour_of_day'] <= 6)
                df_engineered['is_weekend'] = df_engineered['day_of_week'].isin([5, 6])
                self.logger.info("Created time-based features")
            
            # Amount-based features
            if 'amount' in df_engineered.columns:
                df_engineered['amount_log'] = np.log1p(df_engineered['amount'])
                df_engineered['amount_squared'] = df_engineered['amount'] ** 2
                df_engineered['high_amount'] = df_engineered['amount'] > df_engineered['amount'].quantile(0.95)
                self.logger.info("Created amount-based features")
            
            # Distance-based features
            if 'distance_from_home' in df_engineered.columns:
                df_engineered['far_from_home'] = df_engineered['distance_from_home'] > df_engineered['distance_from_home'].quantile(0.9)
                self.logger.info("Created distance-based features")
            
            # Interaction features
            if all(col in df_engineered.columns for col in ['online_order', 'used_pin_number']):
                df_engineered['online_pin'] = df_engineered['online_order'] & df_engineered['used_pin_number']
                self.logger.info("Created interaction features")
            
            # Ratio features
            if 'ratio_to_median_purchase_price' in df_engineered.columns:
                df_engineered['high_ratio'] = df_engineered['ratio_to_median_purchase_price'] > 2.0
                self.logger.info("Created ratio-based features")
            
            self.logger.info(f"Feature engineering completed. New shape: {df_engineered.shape}")
            return df_engineered
            
        except Exception as e:
            self.logger.error(f"Error during feature engineering: {str(e)}")
            raise
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit the encoders (True for training, False for inference)
            
        Returns:
            DataFrame with encoded categorical features
        """
        try:
            self.logger.info("Starting categorical feature encoding")
            df_encoded = df.copy()
            
            categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns
            
            for col in categorical_cols:
                if col == self.config.get('target_column'):
                    continue  # Skip target column
                
                if fit:
                    # Fit and transform
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                    self.label_encoders[col] = le
                    self.logger.info(f"Fitted LabelEncoder for column: {col}")
                else:
                    # Transform only (for inference)
                    if col in self.label_encoders:
                        le = self.label_encoders[col]
                        # Handle unseen categories
                        df_encoded[col] = df_encoded[col].astype(str).map(
                            lambda x: le.transform([x])[0] if x in le.classes_ else -1
                        )
                    else:
                        self.logger.warning(f"No encoder found for column: {col}")
            
            self.logger.info("Categorical feature encoding completed")
            return df_encoded
            
        except Exception as e:
            self.logger.error(f"Error during categorical encoding: {str(e)}")
            raise
    
    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale numerical features.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit the scaler (True for training, False for inference)
            
        Returns:
            DataFrame with scaled features
        """
        try:
            self.logger.info("Starting feature scaling")
            df_scaled = df.copy()
            
            # Get numerical columns (excluding target)
            numerical_cols = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
            target_col = self.config.get('target_column')
            if target_col in numerical_cols:
                numerical_cols.remove(target_col)
            
            if not numerical_cols:
                self.logger.warning("No numerical features found for scaling")
                return df_scaled
            
            # Choose scaler based on configuration
            scaling_method = self.config.get('scaling_method', 'standard')
            if scaling_method == 'standard':
                scaler = StandardScaler()
            elif scaling_method == 'robust':
                scaler = RobustScaler()
            elif scaling_method == 'minmax':
                scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unsupported scaling method: {scaling_method}")
            
            if fit:
                # Fit and transform
                df_scaled[numerical_cols] = scaler.fit_transform(df_scaled[numerical_cols])
                self.scaler = scaler
                self.logger.info(f"Fitted {scaling_method} scaler")
            else:
                # Transform only (for inference)
                if self.scaler is not None:
                    df_scaled[numerical_cols] = self.scaler.transform(df_scaled[numerical_cols])
                    self.logger.info("Applied fitted scaler")
                else:
                    self.logger.warning("No fitted scaler found")
            
            self.logger.info("Feature scaling completed")
            return df_scaled
            
        except Exception as e:
            self.logger.error(f"Error during feature scaling: {str(e)}")
            raise
    
    def select_features(self, df: pd.DataFrame, target_col: str, fit: bool = True) -> pd.DataFrame:
        """
        Select the most important features.
        
        Args:
            df: Input DataFrame
            target_col: Name of the target column
            fit: Whether to fit the selector (True for training, False for inference)
            
        Returns:
            DataFrame with selected features
        """
        try:
            self.logger.info("Starting feature selection")
            
            # Prepare features and target
            feature_cols = [col for col in df.columns if col != target_col]
            X = df[feature_cols]
            y = df[target_col]
            
            if fit:
                # Fit feature selector
                k = min(20, len(feature_cols))  # Select top 20 features or all if less
                self.feature_selector = SelectKBest(score_func=f_classif, k=k)
                X_selected = self.feature_selector.fit_transform(X, y)
                
                # Get selected feature names
                selected_features = X.columns[self.feature_selector.get_support()].tolist()
                self.logger.info(f"Selected {len(selected_features)} features: {selected_features}")
                
                # Create new DataFrame with selected features
                df_selected = df[selected_features + [target_col]]
                
            else:
                # Transform only (for inference)
                if self.feature_selector is not None:
                    X_selected = self.feature_selector.transform(X)
                    selected_features = X.columns[self.feature_selector.get_support()].tolist()
                    df_selected = df[selected_features + [target_col]]
                    self.logger.info("Applied fitted feature selector")
                else:
                    self.logger.warning("No fitted feature selector found")
                    df_selected = df
            
            self.logger.info("Feature selection completed")
            return df_selected
            
        except Exception as e:
            self.logger.error(f"Error during feature selection: {str(e)}")
            raise
    
    def fit_transform(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        Fit the preprocessor and transform the data.
        
        Args:
            df: Input DataFrame
            target_col: Name of the target column
            
        Returns:
            Transformed DataFrame
        """
        try:
            self.logger.info("Starting fit_transform pipeline")
            
            # Clean data
            df_clean = self.clean_data(df)
            
            # Engineer features
            df_engineered = self.engineer_features(df_clean)
            
            # Encode categorical features
            df_encoded = self.encode_categorical_features(df_engineered, fit=True)
            
            # Scale features
            df_scaled = self.scale_features(df_encoded, fit=True)
            
            # Select features
            df_final = self.select_features(df_scaled, target_col, fit=True)
            
            self.is_fitted = True
            self.logger.info("Fit_transform pipeline completed successfully")
            
            return df_final
            
        except Exception as e:
            self.logger.error(f"Error in fit_transform pipeline: {str(e)}")
            raise
    
    def transform(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        Transform new data using fitted preprocessor.
        
        Args:
            df: Input DataFrame
            target_col: Name of the target column
            
        Returns:
            Transformed DataFrame
        """
        try:
            if not self.is_fitted:
                raise ValueError("Preprocessor must be fitted before transform")
            
            self.logger.info("Starting transform pipeline")
            
            # Clean data
            df_clean = self.clean_data(df)
            
            # Engineer features
            df_engineered = self.engineer_features(df_clean)
            
            # Encode categorical features
            df_encoded = self.encode_categorical_features(df_engineered, fit=False)
            
            # Scale features
            df_scaled = self.scale_features(df_encoded, fit=False)
            
            # Select features
            df_final = self.select_features(df_scaled, target_col, fit=False)
            
            self.logger.info("Transform pipeline completed successfully")
            
            return df_final
            
        except Exception as e:
            self.logger.error(f"Error in transform pipeline: {str(e)}")
            raise 