"""
Data preprocessing and feature engineering for fraud detection.

This module handles data cleaning, feature engineering, and preparation
for machine learning models in fraud detection tasks.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

from .config import FEATURE_CONFIG, PROCESSED_DATA_DIR
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
                self.logger.info(
                    f"Missing values found: {missing_counts[missing_counts > 0].to_dict()}"
                )

                # For numerical columns, fill with median
                numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
                for col in numerical_cols:
                    if df_clean[col].isnull().sum() > 0:
                        median_val = df_clean[col].median()
                        df_clean[col].fillna(median_val, inplace=True)
                        self.logger.info(
                            f"Filled missing values in {col} with median: {median_val}"
                        )

                # For categorical columns, fill with mode
                categorical_cols = df_clean.select_dtypes(
                    include=["object", "category"]
                ).columns
                for col in categorical_cols:
                    if df_clean[col].isnull().sum() > 0:
                        mode_val = df_clean[col].mode()[0]
                        df_clean[col].fillna(mode_val, inplace=True)
                        self.logger.info(
                            f"Filled missing values in {col} with mode: {mode_val}"
                        )

            # Remove columns with too many missing values
            high_missing_cols = df_clean.columns[
                df_clean.isnull().sum() > len(df_clean) * 0.5
            ]
            if len(high_missing_cols) > 0:
                df_clean = df_clean.drop(columns=high_missing_cols)
                self.logger.info(
                    f"Removed columns with high missing values: {list(high_missing_cols)}"
                )

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

            # Detect dataset type based on columns
            is_fraud_data = (
                "signup_time" in df_engineered.columns
                and "purchase_time" in df_engineered.columns
            )
            is_creditcard_data = (
                "Time" in df_engineered.columns and "V1" in df_engineered.columns
            )

            if is_fraud_data:
                self.logger.info(
                    "Detected e-commerce fraud dataset - creating e-commerce specific features"
                )
                df_engineered = self._engineer_fraud_data_features(df_engineered)
            elif is_creditcard_data:
                self.logger.info(
                    "Detected credit card fraud dataset - creating credit card specific features"
                )
                df_engineered = self._engineer_creditcard_features(df_engineered)
            else:
                self.logger.info("Unknown dataset type - creating generic features")
                df_engineered = self._engineer_generic_features(df_engineered)

            self.logger.info(
                f"Feature engineering completed. New shape: {df_engineered.shape}"
            )
            return df_engineered

        except Exception as e:
            self.logger.error(f"Error during feature engineering: {str(e)}")
            raise

    def _engineer_fraud_data_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features specific to e-commerce fraud data."""
        df_engineered = df.copy()

        # Time-based features from signup and purchase times
        if (
            "signup_time" in df_engineered.columns
            and "purchase_time" in df_engineered.columns
        ):
            # Extract time components
            df_engineered["signup_hour"] = df_engineered["signup_time"].dt.hour
            df_engineered["signup_day"] = df_engineered["signup_time"].dt.day
            df_engineered["signup_month"] = df_engineered["signup_time"].dt.month
            df_engineered["signup_weekday"] = df_engineered["signup_time"].dt.weekday

            df_engineered["purchase_hour"] = df_engineered["purchase_time"].dt.hour
            df_engineered["purchase_day"] = df_engineered["purchase_time"].dt.day
            df_engineered["purchase_month"] = df_engineered["purchase_time"].dt.month
            df_engineered["purchase_weekday"] = df_engineered[
                "purchase_time"
            ].dt.weekday

            # Time difference features
            df_engineered["days_to_purchase"] = (
                df_engineered["purchase_time"] - df_engineered["signup_time"]
            ).dt.days
            df_engineered["hours_to_purchase"] = (
                df_engineered["purchase_time"] - df_engineered["signup_time"]
            ).dt.total_seconds() / 3600

            # Time-based flags
            df_engineered["is_night_signup"] = (df_engineered["signup_hour"] >= 22) | (
                df_engineered["signup_hour"] <= 6
            )
            df_engineered["is_night_purchase"] = (
                df_engineered["purchase_hour"] >= 22
            ) | (df_engineered["purchase_hour"] <= 6)
            df_engineered["is_weekend_signup"] = df_engineered["signup_weekday"].isin(
                [5, 6]
            )
            df_engineered["is_weekend_purchase"] = df_engineered[
                "purchase_weekday"
            ].isin([5, 6])

            # Quick purchase flag (suspicious if purchase happens very quickly after signup)
            df_engineered["quick_purchase"] = df_engineered["hours_to_purchase"] < 1
            df_engineered["very_quick_purchase"] = (
                df_engineered["hours_to_purchase"] < 0.1
            )  # Less than 6 minutes

            self.logger.info("Created time-based features for e-commerce data")

        # Purchase value features
        if "purchase_value" in df_engineered.columns:
            df_engineered["purchase_value_log"] = np.log1p(
                df_engineered["purchase_value"]
            )
            df_engineered["purchase_value_squared"] = (
                df_engineered["purchase_value"] ** 2
            )
            df_engineered["high_value_purchase"] = df_engineered[
                "purchase_value"
            ] > df_engineered["purchase_value"].quantile(0.95)
            df_engineered["low_value_purchase"] = df_engineered[
                "purchase_value"
            ] < df_engineered["purchase_value"].quantile(0.05)

            # Value categories
            df_engineered["value_category"] = pd.cut(
                df_engineered["purchase_value"],
                bins=[0, 10, 50, 100, 500, float("inf")],
                labels=["very_low", "low", "medium", "high", "very_high"],
            )

            self.logger.info("Created purchase value features")

        # User behavior features
        if "user_id" in df_engineered.columns:
            # User purchase count
            user_purchase_counts = df_engineered["user_id"].value_counts()
            df_engineered["user_purchase_count"] = df_engineered["user_id"].map(
                user_purchase_counts
            )
            df_engineered["is_repeat_user"] = df_engineered["user_purchase_count"] > 1

            # User average purchase value
            user_avg_values = df_engineered.groupby("user_id")["purchase_value"].mean()
            df_engineered["user_avg_purchase_value"] = df_engineered["user_id"].map(
                user_avg_values
            )

            self.logger.info("Created user behavior features")

        # Device behavior features
        if "device_id" in df_engineered.columns:
            # Device usage count
            device_usage_counts = df_engineered["device_id"].value_counts()
            df_engineered["device_usage_count"] = df_engineered["device_id"].map(
                device_usage_counts
            )
            df_engineered["is_shared_device"] = df_engineered["device_usage_count"] > 1

            self.logger.info("Created device behavior features")

        # Source and browser features
        if "source" in df_engineered.columns:
            df_engineered["source_encoded"] = pd.Categorical(
                df_engineered["source"]
            ).codes
            self.logger.info("Created source features")

        if "browser" in df_engineered.columns:
            df_engineered["browser_encoded"] = pd.Categorical(
                df_engineered["browser"]
            ).codes
            self.logger.info("Created browser features")

        # Age features
        if "age" in df_engineered.columns:
            df_engineered["age_group"] = pd.cut(
                df_engineered["age"],
                bins=[0, 25, 35, 50, 65, 100],
                labels=["young", "young_adult", "adult", "senior", "elderly"],
            )
            df_engineered["age_group_encoded"] = pd.Categorical(
                df_engineered["age_group"]
            ).codes
            self.logger.info("Created age features")

        # Sex features
        if "sex" in df_engineered.columns:
            df_engineered["sex_encoded"] = pd.Categorical(df_engineered["sex"]).codes
            self.logger.info("Created sex features")

        # IP address features (if available)
        if "ip_address" in df_engineered.columns:
            # IP usage count
            ip_usage_counts = df_engineered["ip_address"].value_counts()
            df_engineered["ip_usage_count"] = df_engineered["ip_address"].map(
                ip_usage_counts
            )
            df_engineered["is_shared_ip"] = df_engineered["ip_usage_count"] > 1

            self.logger.info("Created IP address features")

        # Country features (if available)
        if "country" in df_engineered.columns:
            df_engineered["country_encoded"] = pd.Categorical(
                df_engineered["country"]
            ).codes
            self.logger.info("Created country features")

        # Advanced fraud-specific features
        df_engineered = self._create_advanced_fraud_features(df_engineered)

        return df_engineered

    def _create_advanced_fraud_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced fraud-specific features for better fraud detection.

        Args:
            df: DataFrame with basic features

        Returns:
            DataFrame with advanced fraud features
        """
        try:
            self.logger.info("Creating advanced fraud-specific features")
            df_advanced = df.copy()

            # 1. TRANSACTION VELOCITY FEATURES (Simplified approach)
            if (
                "user_id" in df_advanced.columns
                and "purchase_time" in df_advanced.columns
            ):
                # Convert to datetime if needed
                if not pd.api.types.is_datetime64_any_dtype(
                    df_advanced["purchase_time"]
                ):
                    df_advanced["purchase_time"] = pd.to_datetime(
                        df_advanced["purchase_time"]
                    )

                # Sort by user and time
                df_sorted = df_advanced.sort_values(["user_id", "purchase_time"])

                # Calculate time differences between consecutive transactions per user
                df_sorted["prev_purchase_time"] = df_sorted.groupby("user_id")[
                    "purchase_time"
                ].shift(1)
                df_sorted["time_between_transactions"] = (
                    df_sorted["purchase_time"] - df_sorted["prev_purchase_time"]
                ).dt.total_seconds()

                # Simple velocity features (avoid complex rolling windows)
                user_velocity = (
                    df_sorted.groupby("user_id")
                    .agg(
                        {
                            "purchase_time": "count",
                            "time_between_transactions": ["mean", "min", "std"],
                        }
                    )
                    .reset_index()
                )

                user_velocity.columns = [
                    "user_id",
                    "user_total_transactions",
                    "avg_time_between_transactions",
                    "min_time_between_transactions",
                    "std_time_between_transactions",
                ]

                # Merge back using a safer approach
                df_advanced = df_advanced.merge(user_velocity, on="user_id", how="left")

                # Create velocity flags
                df_advanced["high_transaction_count"] = (
                    df_advanced["user_total_transactions"] > 5
                )
                df_advanced["fast_transactions"] = (
                    df_advanced["avg_time_between_transactions"] < 3600
                )  # < 1 hour
                df_advanced["very_fast_transactions"] = (
                    df_advanced["min_time_between_transactions"] < 300
                )  # < 5 min

                self.logger.info("Created transaction velocity features")

            # 2. BEHAVIORAL ANOMALY FEATURES
            if (
                "user_id" in df_advanced.columns
                and "purchase_value" in df_advanced.columns
            ):
                # User behavior patterns
                user_stats = (
                    df_advanced.groupby("user_id")["purchase_value"]
                    .agg(["mean", "std", "min", "max", "count"])
                    .reset_index()
                )
                user_stats.columns = [
                    "user_id",
                    "user_avg_amount",
                    "user_std_amount",
                    "user_min_amount",
                    "user_max_amount",
                    "user_transaction_count",
                ]

                # Merge user stats
                df_advanced = df_advanced.merge(user_stats, on="user_id", how="left")

                # Anomaly detection features
                df_advanced["amount_deviation"] = abs(
                    df_advanced["purchase_value"] - df_advanced["user_avg_amount"]
                ) / (df_advanced["user_std_amount"] + 1e-8)
                df_advanced["high_amount_deviation"] = (
                    df_advanced["amount_deviation"] > 2.0
                )
                df_advanced["unusual_purchase_pattern"] = (
                    df_advanced["purchase_value"] > df_advanced["user_max_amount"] * 1.5
                ) | (
                    df_advanced["purchase_value"] < df_advanced["user_min_amount"] * 0.5
                )

                self.logger.info("Created behavioral anomaly features")

            # 3. GEOGRAPHIC ANOMALY FEATURES
            if "user_id" in df_advanced.columns and "country" in df_advanced.columns:
                # User's typical location
                user_countries = (
                    df_advanced.groupby("user_id")["country"]
                    .agg(["count", "nunique"])
                    .reset_index()
                )
                user_countries.columns = [
                    "user_id",
                    "user_country_count",
                    "user_unique_countries",
                ]

                # Merge country stats
                df_advanced = df_advanced.merge(
                    user_countries, on="user_id", how="left"
                )

                # Geographic anomaly flags
                df_advanced["multiple_countries"] = (
                    df_advanced["user_unique_countries"] > 1
                )
                df_advanced["unusual_country"] = (
                    df_advanced["user_unique_countries"] == 1
                )

                self.logger.info("Created geographic anomaly features")

            # 4. DEVICE FINGERPRINTING FEATURES
            if "user_id" in df_advanced.columns and "device_id" in df_advanced.columns:
                # Device sharing patterns
                device_users = (
                    df_advanced.groupby("device_id")["user_id"]
                    .agg(["count", "nunique"])
                    .reset_index()
                )
                device_users.columns = [
                    "device_id",
                    "device_transaction_count",
                    "device_unique_users",
                ]

                # Merge device stats
                df_advanced = df_advanced.merge(
                    device_users, on="device_id", how="left"
                )

                # Device anomaly flags
                df_advanced["shared_device"] = df_advanced["device_unique_users"] > 1
                df_advanced["high_device_usage"] = (
                    df_advanced["device_transaction_count"] > 5
                )
                df_advanced["device_user_ratio"] = df_advanced[
                    "device_unique_users"
                ] / (df_advanced["device_transaction_count"] + 1e-8)

                self.logger.info("Created device fingerprinting features")

            # 5. TIME-BASED FRAUD PATTERNS
            if "purchase_time" in df_advanced.columns:
                # Time-based fraud indicators
                df_advanced["is_late_night"] = (
                    df_advanced["purchase_time"].dt.hour >= 22
                ) | (df_advanced["purchase_time"].dt.hour <= 6)
                df_advanced["is_weekend"] = df_advanced[
                    "purchase_time"
                ].dt.dayofweek.isin([5, 6])
                df_advanced["is_holiday_hours"] = df_advanced[
                    "purchase_time"
                ].dt.hour.isin([0, 1, 2, 3, 4, 5, 23])

                # Time since signup (if available)
                if "signup_time" in df_advanced.columns:
                    if not pd.api.types.is_datetime64_any_dtype(
                        df_advanced["signup_time"]
                    ):
                        df_advanced["signup_time"] = pd.to_datetime(
                            df_advanced["signup_time"]
                        )

                    df_advanced["time_since_signup_seconds"] = (
                        df_advanced["purchase_time"] - df_advanced["signup_time"]
                    ).dt.total_seconds()

                    # Fast fraud indicators
                    df_advanced["fast_fraud_risk"] = (
                        df_advanced["time_since_signup_seconds"] < 3600
                    )  # Within 1 hour
                    df_advanced["very_fast_fraud_risk"] = (
                        df_advanced["time_since_signup_seconds"] < 300
                    )  # Within 5 minutes

                    self.logger.info("Created time-based fraud pattern features")

            # 6. AMOUNT-BASED FRAUD PATTERNS
            if "purchase_value" in df_advanced.columns:
                # Amount thresholds and patterns
                df_advanced["is_round_amount"] = (
                    df_advanced["purchase_value"] % 100 == 0
                )
                df_advanced["is_test_amount"] = df_advanced["purchase_value"].isin(
                    [0.01, 1.00, 10.00, 100.00]
                )
                df_advanced["is_high_value"] = df_advanced[
                    "purchase_value"
                ] > df_advanced["purchase_value"].quantile(0.95)
                df_advanced["is_low_value"] = df_advanced[
                    "purchase_value"
                ] < df_advanced["purchase_value"].quantile(0.05)

                # Amount distribution features
                df_advanced["amount_percentile"] = df_advanced["purchase_value"].rank(
                    pct=True
                )
                df_advanced["amount_z_score"] = (
                    df_advanced["purchase_value"] - df_advanced["purchase_value"].mean()
                ) / (df_advanced["purchase_value"].std() + 1e-8)

                self.logger.info("Created amount-based fraud pattern features")

            # 7. COMPOSITE FRAUD RISK SCORES
            # Combine multiple risk factors
            risk_factors = []

            if "high_transaction_count" in df_advanced.columns:
                risk_factors.append(df_advanced["high_transaction_count"].astype(int))
            if "high_amount_deviation" in df_advanced.columns:
                risk_factors.append(df_advanced["high_amount_deviation"].astype(int))
            if "multiple_countries" in df_advanced.columns:
                risk_factors.append(df_advanced["multiple_countries"].astype(int))
            if "shared_device" in df_advanced.columns:
                risk_factors.append(df_advanced["shared_device"].astype(int))
            if "is_late_night" in df_advanced.columns:
                risk_factors.append(df_advanced["is_late_night"].astype(int))
            if "fast_fraud_risk" in df_advanced.columns:
                risk_factors.append(df_advanced["fast_fraud_risk"].astype(int))
            if "is_high_value" in df_advanced.columns:
                risk_factors.append(df_advanced["is_high_value"].astype(int))

            if risk_factors:
                df_advanced["fraud_risk_score"] = sum(risk_factors)
                df_advanced["high_fraud_risk"] = df_advanced["fraud_risk_score"] >= 3
                df_advanced["very_high_fraud_risk"] = (
                    df_advanced["fraud_risk_score"] >= 5
                )

                self.logger.info("Created composite fraud risk scores")

            self.logger.info(
                f"Advanced fraud features completed. Total features: {len(df_advanced.columns)}"
            )
            return df_advanced

        except Exception as e:
            self.logger.error(f"Error creating advanced fraud features: {str(e)}")
            return df

    def _engineer_creditcard_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features specific to credit card fraud data."""
        df_engineered = df.copy()

        # Time-based features
        if "Time" in df_engineered.columns:
            # Extract time components (assuming Time is in seconds from start)
            df_engineered["time_hour"] = (df_engineered["Time"] // 3600) % 24
            df_engineered["time_day"] = (
                df_engineered["Time"] // 86400
            ) % 7  # Day of week

            # Time-based flags
            df_engineered["is_night_time"] = (df_engineered["time_hour"] >= 22) | (
                df_engineered["time_hour"] <= 6
            )
            df_engineered["is_weekend"] = df_engineered["time_day"].isin([5, 6])

            self.logger.info("Created time-based features for credit card data")

        # Amount features
        if "Amount" in df_engineered.columns:
            df_engineered["amount_log"] = np.log1p(df_engineered["Amount"])
            df_engineered["amount_squared"] = df_engineered["Amount"] ** 2
            df_engineered["high_amount"] = df_engineered["Amount"] > df_engineered[
                "Amount"
            ].quantile(0.95)
            df_engineered["low_amount"] = df_engineered["Amount"] < df_engineered[
                "Amount"
            ].quantile(0.05)

            # Amount categories
            df_engineered["amount_category"] = pd.cut(
                df_engineered["Amount"],
                bins=[0, 10, 50, 100, 500, float("inf")],
                labels=["very_low", "low", "medium", "high", "very_high"],
            )

            self.logger.info("Created amount features for credit card data")

        # PCA feature interactions (V1-V5)
        v_columns = [col for col in df_engineered.columns if col.startswith("V")]
        if len(v_columns) >= 2:
            # Create interaction features between PCA components
            for i, col1 in enumerate(
                v_columns[:3]
            ):  # Limit to first 3 to avoid too many features
                for col2 in v_columns[i + 1 : 4]:
                    interaction_name = f"{col1}_{col2}_interaction"
                    df_engineered[interaction_name] = (
                        df_engineered[col1] * df_engineered[col2]
                    )

            # Statistical features from PCA components
            df_engineered["v_features_mean"] = df_engineered[v_columns].mean(axis=1)
            df_engineered["v_features_std"] = df_engineered[v_columns].std(axis=1)
            df_engineered["v_features_max"] = df_engineered[v_columns].max(axis=1)
            df_engineered["v_features_min"] = df_engineered[v_columns].min(axis=1)

            self.logger.info("Created PCA feature interactions")

        return df_engineered

    def _engineer_generic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer generic features for unknown dataset types."""
        df_engineered = df.copy()

        # Generic numerical features
        numerical_cols = df_engineered.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if col not in ["class", "Class", "fraud"]:  # Skip target columns
                df_engineered[f"{col}_log"] = np.log1p(np.abs(df_engineered[col]))
                df_engineered[f"{col}_squared"] = df_engineered[col] ** 2

        # Generic categorical features
        categorical_cols = df_engineered.select_dtypes(
            include=["object", "category"]
        ).columns
        for col in categorical_cols:
            if col not in ["class", "Class", "fraud"]:  # Skip target columns
                df_engineered[f"{col}_encoded"] = pd.Categorical(
                    df_engineered[col]
                ).codes

        self.logger.info("Created generic features")
        return df_engineered

    def encode_categorical_features(
        self, df: pd.DataFrame, fit: bool = True
    ) -> pd.DataFrame:
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

            categorical_cols = df_encoded.select_dtypes(
                include=["object", "category"]
            ).columns

            for col in categorical_cols:
                # Skip target column - check both config and common target names
                if col == self.config.get("target_column") or col in [
                    "class",
                    "Class",
                    "target",
                    "Target",
                    "label",
                    "Label",
                ]:
                    self.logger.info(f"Skipping target column: {col}")
                    continue

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
                        df_encoded[col] = (
                            df_encoded[col]
                            .astype(str)
                            .map(
                                lambda x: (
                                    le.transform([x])[0] if x in le.classes_ else -1
                                )
                            )
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
            numerical_cols = df_scaled.select_dtypes(
                include=[np.number]
            ).columns.tolist()
            target_col = self.config.get("target_column")
            if target_col in numerical_cols:
                numerical_cols.remove(target_col)

            if not numerical_cols:
                self.logger.warning("No numerical features found for scaling")
                return df_scaled

            # Choose scaler based on configuration
            scaling_method = self.config.get("scaling_method", "standard")
            if scaling_method == "standard":
                scaler = StandardScaler()
            elif scaling_method == "robust":
                scaler = RobustScaler()
            elif scaling_method == "minmax":
                scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unsupported scaling method: {scaling_method}")

            if fit:
                # Fit and transform
                df_scaled[numerical_cols] = scaler.fit_transform(
                    df_scaled[numerical_cols]
                )
                self.scaler = scaler
                self.logger.info(f"Fitted {scaling_method} scaler")
            else:
                # Transform only (for inference)
                if self.scaler is not None:
                    df_scaled[numerical_cols] = self.scaler.transform(
                        df_scaled[numerical_cols]
                    )
                    self.logger.info("Applied fitted scaler")
                else:
                    self.logger.warning("No fitted scaler found")

            self.logger.info("Feature scaling completed")
            return df_scaled

        except Exception as e:
            self.logger.error(f"Error during feature scaling: {str(e)}")
            raise

    def select_features(
        self, df: pd.DataFrame, target_col: str, fit: bool = True
    ) -> pd.DataFrame:
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

            # Remove datetime columns as they cause issues with sklearn
            datetime_cols = df.select_dtypes(include=["datetime64"]).columns
            if len(datetime_cols) > 0:
                df = df.drop(columns=datetime_cols)
                self.logger.info(f"Removed datetime columns: {list(datetime_cols)}")

            # Prepare features and target
            feature_cols = [col for col in df.columns if col != target_col]
            X = df[feature_cols]
            y = df[target_col]

            # Convert to numeric only for feature selection
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                self.logger.warning("No numeric features available for selection")
                return df

            X_numeric = X[numeric_cols]

            # Handle NaN values in numeric features
            if X_numeric.isnull().any().any():
                self.logger.info("Handling NaN values in numeric features")
                # Fill NaN with median for each column
                X_numeric = X_numeric.fillna(X_numeric.median())
                self.logger.info(
                    f"Filled NaN values in {X_numeric.isnull().sum().sum()} cells"
                )

            if fit:
                # Fit feature selector
                k = min(30, len(numeric_cols))  # Select top 30 features or all if less
                self.feature_selector = SelectKBest(score_func=f_classif, k=k)
                self.feature_selector.fit_transform(X_numeric, y)

                # Get selected feature names
                selected_features = numeric_cols[
                    self.feature_selector.get_support()
                ].tolist()
                self.logger.info(
                    f"Selected {len(selected_features)} features: {selected_features}"
                )

                # Create new DataFrame with selected features and target
                df_selected = df[selected_features + [target_col]]

            else:
                # Transform only (for inference)
                if self.feature_selector is not None:
                    X_selected = self.feature_selector.transform(X_numeric)
                    selected_features = numeric_cols[
                        self.feature_selector.get_support()
                    ].tolist()
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

    def handle_imbalanced_data(
        self, X: pd.DataFrame, y: pd.Series, method: str = "smote"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Handle imbalanced data using various resampling techniques.

        Args:
            X: Feature DataFrame
            y: Target Series
            method: Resampling method ('smote', 'undersample', 'smoteenn', 'none')

        Returns:
            Tuple of resampled features and target
        """
        try:
            self.logger.info(f"Handling imbalanced data using method: {method}")

            if method == "none":
                return X, y

            # Import resampling methods
            from imblearn.combine import SMOTEENN
            from imblearn.over_sampling import SMOTE
            from imblearn.under_sampling import RandomUnderSampler

            if method == "smote":
                resampler = SMOTE(random_state=42)
            elif method == "undersample":
                resampler = RandomUnderSampler(random_state=42)
            elif method == "smoteenn":
                resampler = SMOTEENN(random_state=42)
            else:
                raise ValueError(f"Unsupported resampling method: {method}")

            # Resample the data
            X_resampled, y_resampled = resampler.fit_resample(X, y)

            self.logger.info(
                f"Resampled data: {len(X_resampled)} samples (original: {len(X)})"
            )
            self.logger.info(
                f"New class distribution: {y_resampled.value_counts().to_dict()}"
            )

            return X_resampled, y_resampled

        except Exception as e:
            self.logger.error(f"Error during imbalanced data handling: {str(e)}")
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

            # Save preprocessed data to processed directory
            processed_file_path = PROCESSED_DATA_DIR / "preprocessed_fraud_data.csv"
            df_final.to_csv(processed_file_path, index=False)
            self.logger.info(f"Saved preprocessed data to: {processed_file_path}")

            # Save feature importance information
            if hasattr(self, "feature_selector") and self.feature_selector is not None:
                feature_cols = [col for col in df_final.columns if col != target_col]
                numeric_cols = (
                    df_final[feature_cols].select_dtypes(include=[np.number]).columns
                )

                if len(numeric_cols) > 0:
                    X = df_final[numeric_cols]
                    y = df_final[target_col]

                    # Handle NaN values before calculating feature importance
                    if X.isnull().any().any():
                        self.logger.info(
                            "Handling NaN values for feature importance calculation"
                        )
                        X = X.fillna(X.median())

                    f_scores, p_values = f_classif(X, y)

                    feature_importance_data = {
                        "selected_features": numeric_cols.tolist(),
                        "f_scores": f_scores.tolist(),
                        "p_values": p_values.tolist(),
                        "total_features": len(numeric_cols),
                        "selection_method": "f_classif",
                        "target_column": target_col,
                    }

                    feature_importance_file = (
                        PROCESSED_DATA_DIR / "feature_importance.json"
                    )
                    with open(feature_importance_file, "w") as f:
                        json.dump(feature_importance_data, f, indent=2)
                    self.logger.info(
                        f"Saved feature importance to: {feature_importance_file}"
                    )

            # Save preprocessing metadata
            metadata = {
                "original_shape": df.shape,
                "final_shape": df_final.shape,
                "features_engineered": len(df_engineered.columns)
                - len(df_clean.columns),
                "features_selected": len(df_final.columns),
                "target_column": target_col,
                "preprocessing_steps": [
                    "cleaning",
                    "feature_engineering",
                    "encoding",
                    "scaling",
                    "selection",
                ],
                "timestamp": datetime.now().isoformat(),
                "scaler_type": type(self.scaler).__name__ if self.scaler else None,
                "feature_selector_type": (
                    type(self.feature_selector).__name__
                    if self.feature_selector
                    else None
                ),
            }

            metadata_file = PROCESSED_DATA_DIR / "preprocessing_metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
            self.logger.info(f"Saved preprocessing metadata to: {metadata_file}")

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
