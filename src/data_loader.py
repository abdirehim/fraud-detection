"""
Data loading utilities for the fraud detection pipeline.

This module handles loading and validation of financial transaction data
with proper error handling and logging. It provides comprehensive data
management capabilities including:

- Multi-format data loading (CSV, Parquet)
- Data quality validation and cleaning
- Geographic data enrichment (IP-to-country mapping)
- Synthetic fraud scenario generation
- Data persistence and caching

The module is specifically designed for fraud detection datasets with:
- Imbalanced class distributions (1-15% fraud rate)
- High-dimensional feature spaces (87+ engineered features)
- Temporal and geographic patterns
- Device and user behavior analysis

Author: Fraud Detection Team
Date: July 2025
Version: 2.0
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import logging
import ipaddress
from datetime import datetime, timedelta
import random

from .config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from .utils import setup_logging, validate_data_path, get_file_size_mb


class DataLoader:
    """
    Data loader class for handling financial transaction datasets.
    
    This class provides methods to load, validate, and preprocess
    financial data for fraud detection tasks. It includes advanced
    features for handling imbalanced datasets and generating synthetic
    fraud scenarios to improve model performance.
    
    Key Features:
    - Multi-format data loading with error handling
    - Data quality assessment and cleaning
    - Geographic data enrichment
    - Synthetic fraud generation
    - Comprehensive logging and validation
    
    Attributes:
        data_dir (Path): Directory containing the data files
        logger (logging.Logger): Logger instance for this class
        _cached_data (Dict): Cache for loaded datasets to avoid reloading
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize the DataLoader with configuration and logging setup.
        
        Args:
            data_dir (Optional[Path]): Directory containing the data files.
                If None, uses the default RAW_DATA_DIR from config.
        
        Example:
            >>> loader = DataLoader()
            >>> loader = DataLoader(Path("/custom/data/path"))
        """
        self.data_dir = data_dir or RAW_DATA_DIR
        self.logger = setup_logging("data_loader")
        self._cached_data = {}  # Cache for loaded datasets
        
        # Validate data directory exists
        if not self.data_dir.exists():
            self.logger.warning(f"Data directory does not exist: {self.data_dir}")
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created data directory: {self.data_dir}")
        
        self.logger.info(f"Initialized DataLoader with data directory: {self.data_dir}")
        
    def load_csv_data(
        self, 
        filename: str, 
        encoding: str = "utf-8",
        **kwargs
    ) -> pd.DataFrame:
        """
        Load data from a CSV file with comprehensive error handling and validation.
        
        This method provides robust CSV loading with:
        - File existence validation
        - Encoding handling
        - Data quality checks
        - Comprehensive error reporting
        - Performance logging
        
        Args:
            filename (str): Name of the CSV file to load
            encoding (str): File encoding (default: "utf-8")
            **kwargs: Additional arguments to pass to pd.read_csv
                Common options:
                - sep: Delimiter (default: ',')
                - header: Row number to use as column names
                - index_col: Column to use as index
                - dtype: Data types for columns
                - parse_dates: Columns to parse as dates
        
        Returns:
            pd.DataFrame: Loaded and validated DataFrame
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            pd.errors.EmptyDataError: If the file is empty
            pd.errors.ParserError: If there are parsing errors
            ValueError: If loaded data fails validation checks
        
        Example:
            >>> loader = DataLoader()
            >>> df = loader.load_csv_data("fraud_data.csv", sep=",", parse_dates=["timestamp"])
        """
        filepath = self.data_dir / filename
        
        try:
            # Step 1: Validate file path and existence
            if not validate_data_path(filepath):
                raise FileNotFoundError(f"Data file not found: {filepath}")
            
            # Step 2: Log file information for debugging
            file_size = get_file_size_mb(filepath)
            self.logger.info(f"Loading CSV file: {filename} (Size: {file_size:.2f} MB)")
            
            # Step 3: Load the data with specified parameters
            start_time = datetime.now()
            df = pd.read_csv(filepath, encoding=encoding, **kwargs)
            load_time = (datetime.now() - start_time).total_seconds()
            
            # Step 4: Validate loaded data quality
            if df.empty:
                raise pd.errors.EmptyDataError(f"CSV file is empty: {filename}")
            
            # Step 5: Log successful loading with details
            self.logger.info(
                f"Successfully loaded data: {df.shape[0]:,} rows, "
                f"{df.shape[1]} columns in {load_time:.2f}s"
            )
            self.logger.info(f"Columns: {list(df.columns)}")
            
            # Step 6: Basic data quality checks
            self._log_data_quality_summary(df, filename)
            
            return df
            
        except FileNotFoundError as e:
            self.logger.error(f"File not found: {str(e)}")
            raise
        except pd.errors.EmptyDataError as e:
            self.logger.error(f"Empty data file: {str(e)}")
            raise
        except pd.errors.ParserError as e:
            self.logger.error(f"CSV parsing error: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error loading data: {str(e)}")
            raise
    
    def load_parquet_data(
        self, 
        filename: str,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load data from a Parquet file with error handling.
        
        Args:
            filename: Name of the Parquet file
            **kwargs: Additional arguments to pass to pd.read_parquet
            
        Returns:
            Loaded DataFrame
        """
        filepath = self.data_dir / filename
        
        try:
            # Validate file path
            if not validate_data_path(filepath):
                raise FileNotFoundError(f"Data file not found: {filepath}")
            
            # Log file information
            file_size = get_file_size_mb(filepath)
            self.logger.info(f"Loading Parquet file: {filename} (Size: {file_size} MB)")
            
            # Load the data
            df = pd.read_parquet(filepath, **kwargs)
            
            # Validate loaded data
            if df.empty:
                raise pd.errors.EmptyDataError(f"Parquet file is empty: {filename}")
            
            self.logger.info(f"Successfully loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
            self.logger.info(f"Columns: {list(df.columns)}")
            
            return df
            
        except FileNotFoundError as e:
            self.logger.error(f"File not found: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error loading Parquet data: {str(e)}")
            raise

    def load_fraud_data(self) -> pd.DataFrame:
        """
        Load the main e-commerce fraud detection dataset.
        
        This method loads the primary fraud dataset containing:
        - User transaction data
        - Device and browser information
        - Temporal patterns (signup and purchase times)
        - Transaction amounts and sources
        - Fraud labels (target variable)
        
        The dataset is specifically designed for fraud detection with:
        - Imbalanced classes (typically 1-15% fraud rate)
        - High-dimensional feature space
        - Temporal and behavioral patterns
        
        Returns:
            pd.DataFrame: Fraud detection dataset with columns:
                - user_id: Unique user identifier
                - signup_time: User registration timestamp
                - purchase_time: Transaction timestamp
                - purchase_value: Transaction amount
                - device_id: Device identifier
                - source: Traffic source (SEO, Ads, etc.)
                - browser: Browser type
                - sex: User gender
                - age: User age
                - ip_address: IP address
                - class: Target variable (0=legitimate, 1=fraud)
        
        Raises:
            FileNotFoundError: If fraud data file is missing
            Exception: For other loading errors
        
        Example:
            >>> loader = DataLoader()
            >>> fraud_df = loader.load_fraud_data()
            >>> print(f"Loaded {len(fraud_df)} transactions")
        """
        try:
            self.logger.info("Loading e-commerce fraud dataset...")
            
            # Load the main fraud dataset with proper data types
            df = self.load_csv_data(
                "Fraud_Data.csv",
                parse_dates=["signup_time", "purchase_time"],
                dtype={
                    "user_id": "string",
                    "device_id": "string", 
                    "source": "category",
                    "browser": "category",
                    "sex": "category",
                    "ip_address": "string"
                }
            )
            
            # Apply comprehensive data cleaning and validation
            df = self.clean_raw_data(df, dataset_type="fraud")
            
            # Ensure class column is binary (0, 1) for modeling
            if 'class' in df.columns:
                df['class'] = df['class'].map({0: 0, 1: 1}).fillna(0)
            
            # Generate synthetic fraud scenarios to improve class balance
            df = self.generate_synthetic_fraud_scenarios(df, target_col='class')
            
            # Log final dataset characteristics
            if 'class' in df.columns:
                fraud_rate = df['class'].mean() * 100
                self.logger.info(f"Final fraud rate: {fraud_rate:.2f}% ({df['class'].sum()} fraud cases)")
                self.logger.info(f"Final dataset shape: {df.shape}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading fraud data: {str(e)}")
            raise

    def load_creditcard_data(self) -> pd.DataFrame:
        """
        Load the credit card fraud detection dataset.
        
        This dataset contains anonymized credit card transaction data
        with PCA-transformed features (V1-V28) for privacy protection.
        The dataset is highly imbalanced with very low fraud rates.
        
        Returns:
            pd.DataFrame: Credit card fraud dataset with columns:
                - V1-V28: PCA-transformed features (anonymized)
                - Amount: Transaction amount
                - Class: Target variable (0=legitimate, 1=fraud)
        
        Raises:
            FileNotFoundError: If credit card data file is missing
            Exception: For other loading errors
        
        Example:
            >>> loader = DataLoader()
            >>> cc_df = loader.load_creditcard_data()
            >>> print(f"Credit card fraud rate: {(cc_df['Class'] == 1).mean():.4f}")
        """
        try:
            self.logger.info("Loading credit card fraud dataset...")
            
            # Load credit card dataset with optimized data types
            df = self.load_csv_data("creditcard.csv")
            
            # Validate expected structure (PCA features V1-V28)
            expected_v_columns = [f"V{i}" for i in range(1, 29)]
            missing_v_columns = set(expected_v_columns) - set(df.columns)
            if missing_v_columns:
                raise ValueError(f"Missing PCA feature columns: {missing_v_columns}")
            
            # Apply data cleaning for credit card dataset
            df = self.clean_raw_data(df, dataset_type="creditcard")
            
            # Log dataset characteristics
            fraud_rate = (df['Class'] == 1).mean() * 100
            self.logger.info(f"Credit card fraud rate: {fraud_rate:.4f}%")
            self.logger.info(f"Credit card dataset shape: {df.shape}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading credit card data: {str(e)}")
            raise

    def load_ip_country_mapping(self) -> pd.DataFrame:
        """
        Load IP address to country mapping data for geographic feature engineering.
        
        This dataset maps IP address ranges to countries, enabling:
        - Geographic risk assessment
        - Cross-border transaction detection
        - Location-based fraud patterns
        - IP reputation analysis
        
        Returns:
            pd.DataFrame: IP-to-country mapping with columns:
                - ip_start: Starting IP address in range
                - ip_end: Ending IP address in range
                - country: Country name or code
                - ip_start_int: Integer representation of start IP
                - ip_end_int: Integer representation of end IP
        
        Raises:
            FileNotFoundError: If IP mapping file is missing
            Exception: For other loading errors
        
        Example:
            >>> loader = DataLoader()
            >>> ip_mapping = loader.load_ip_country_mapping()
            >>> print(f"Loaded {len(ip_mapping)} IP ranges")
        """
        try:
            self.logger.info("Loading IP-to-country mapping data...")
            
            # Load IP mapping data
            df = self.load_csv_data("IpAddress_to_Country.csv")
            
            # Validate expected columns
            expected_columns = ["ip_start", "ip_end", "country"]
            missing_columns = set(expected_columns) - set(df.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns in IP mapping: {missing_columns}")
            
            # Convert IP addresses to integers for efficient lookup
            # This enables fast range-based searches
            df['ip_start_int'] = df['ip_start'].apply(lambda x: int(ipaddress.IPv4Address(x)))
            df['ip_end_int'] = df['ip_end'].apply(lambda x: int(ipaddress.IPv4Address(x)))
            
            # Sort by IP range for efficient binary search
            df = df.sort_values('ip_start_int').reset_index(drop=True)
            
            self.logger.info(f"Loaded {len(df)} IP address ranges")
            self.logger.info(f"Countries covered: {df['country'].nunique()}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading IP mapping data: {str(e)}")
            raise

    def ip_to_country(self, ip_address: str, ip_mapping_df: pd.DataFrame) -> str:
        """
        Convert IP address to country using the mapping dataset.
        
        Args:
            ip_address: IP address string
            ip_mapping_df: DataFrame with IP address mappings
            
        Returns:
            Country name or 'Unknown' if not found
        """
        try:
            ip_int = int(ipaddress.IPv4Address(ip_address))
            
            # Find matching country
            mask = (ip_mapping_df['lower_bound_ip_int'] <= ip_int) & \
                   (ip_mapping_df['upper_bound_ip_int'] >= ip_int)
            
            if mask.any():
                return ip_mapping_df.loc[mask.idxmax(), 'country']
            else:
                return 'Unknown'
                
        except Exception:
            return 'Unknown'

    def merge_fraud_with_geolocation(self, fraud_df: pd.DataFrame, ip_mapping_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge fraud data with geolocation information.
        
        Args:
            fraud_df: E-commerce fraud dataset
            ip_mapping_df: IP address to country mapping
            
        Returns:
            Merged DataFrame with country information
        """
        try:
            self.logger.info("Merging fraud data with geolocation information")
            
            # Create a copy to avoid modifying original
            merged_df = fraud_df.copy()
            
            # Add country information
            if 'ip_address' in merged_df.columns:
                merged_df['country'] = merged_df['ip_address'].apply(
                    lambda x: self.ip_to_country(x, ip_mapping_df)
                )
                
                # Log country distribution
                country_counts = merged_df['country'].value_counts()
                self.logger.info(f"Top 10 countries by transaction count:")
                for country, count in country_counts.head(10).items():
                    self.logger.info(f"  {country}: {count}")
                
                # Calculate fraud rate by country
                if 'class' in merged_df.columns:
                    # Ensure class column is binary (0, 1)
                    merged_df['class'] = merged_df['class'].map({0: 0, 1: 1}).fillna(0)
                    
                    fraud_by_country = merged_df.groupby('country')['class'].agg(['count', 'mean'])
                    fraud_by_country.columns = ['transaction_count', 'fraud_rate']
                    fraud_by_country = fraud_by_country.sort_values('fraud_rate', ascending=False)
                    
                    self.logger.info(f"Top 10 countries by fraud rate:")
                    for country, row in fraud_by_country.head(10).iterrows():
                        self.logger.info(f"  {country}: {row['fraud_rate']:.3f} ({row['transaction_count']} transactions)")
            
            return merged_df
            
        except Exception as e:
            self.logger.error(f"Error merging fraud data with geolocation: {str(e)}")
            raise

    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Load all fraud detection datasets.
        
        Returns:
            Dictionary containing all datasets
        """
        try:
            self.logger.info("Loading all fraud detection datasets")
            
            datasets = {}
            
            # Load individual datasets
            try:
                datasets['fraud_data'] = self.load_fraud_data()
            except FileNotFoundError:
                self.logger.warning("Fraud_Data.csv not found, skipping e-commerce dataset")
            
            try:
                datasets['creditcard_data'] = self.load_creditcard_data()
            except FileNotFoundError:
                self.logger.warning("creditcard.csv not found, skipping credit card dataset")
            
            try:
                datasets['ip_mapping'] = self.load_ip_country_mapping()
            except FileNotFoundError:
                self.logger.warning("IpAddress_to_Country.csv not found, skipping IP mapping")
            
            # Merge fraud data with geolocation if both are available
            if 'fraud_data' in datasets and 'ip_mapping' in datasets:
                try:
                    datasets['fraud_data_with_geo'] = self.merge_fraud_with_geolocation(
                        datasets['fraud_data'], datasets['ip_mapping']
                    )
                    
                    # Save merged data with geolocation
                    merged_file_path = PROCESSED_DATA_DIR / "cleaned_fraud_data_with_geo.csv"
                    datasets['fraud_data_with_geo'].to_csv(merged_file_path, index=False)
                    self.logger.info(f"Saved merged fraud data with geolocation to: {merged_file_path}")
                    
                except Exception as e:
                    self.logger.warning(f"Could not merge fraud data with geolocation: {str(e)}")
            
            self.logger.info(f"Successfully loaded {len(datasets)} datasets")
            return datasets
            
        except Exception as e:
            self.logger.error(f"Error loading all datasets: {str(e)}")
            raise

    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate the quality of loaded data.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary containing validation results
        """
        validation_results = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicate_rows": df.duplicated().sum(),
            "data_types": df.dtypes.to_dict(),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024)
        }
        
        # Check for completely empty columns
        empty_columns = df.columns[df.isnull().all()].tolist()
        validation_results["empty_columns"] = empty_columns
        
        # Check for columns with high missing values (>50%)
        high_missing_columns = df.columns[df.isnull().sum() > len(df) * 0.5].tolist()
        validation_results["high_missing_columns"] = high_missing_columns
        
        self.logger.info("Data quality validation completed")
        self.logger.info(f"Validation results: {validation_results}")
        
        return validation_results
    
    def save_processed_data(
        self, 
        df: pd.DataFrame, 
        filename: str, 
        output_dir: Optional[Path] = None,
        format: str = "parquet"
    ) -> None:
        """
        Save processed data to disk.
        
        Args:
            df: DataFrame to save
            filename: Output filename
            output_dir: Output directory (defaults to processed data dir)
            format: Output format ('csv' or 'parquet')
        """
        output_dir = output_dir or PROCESSED_DATA_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            if format.lower() == "csv":
                filepath = output_dir / filename
                df.to_csv(filepath, index=False)
            elif format.lower() == "parquet":
                filepath = output_dir / filename.replace('.csv', '.parquet')
                df.to_parquet(filepath, index=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            file_size = get_file_size_mb(filepath)
            self.logger.info(f"Saved processed data: {filepath} (Size: {file_size} MB)")
            
        except Exception as e:
            self.logger.error(f"Failed to save processed data: {str(e)}")
            raise
    
    def get_data_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive information about the dataset.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary containing dataset information
        """
        info = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicate_rows": df.duplicated().sum(),
            "numerical_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist(),
            "datetime_columns": df.select_dtypes(include=['datetime']).columns.tolist()
        }
        
        # Add basic statistics for numerical columns
        if info["numerical_columns"]:
            info["numerical_stats"] = df[info["numerical_columns"]].describe().to_dict()
        
        return info

    def clean_raw_data(self, df: pd.DataFrame, dataset_type: str = "fraud") -> pd.DataFrame:
        """
        Clean raw data with comprehensive validation and cleaning.
        
        Args:
            df: Raw DataFrame to clean
            dataset_type: Type of dataset ('fraud', 'creditcard')
            
        Returns:
            Cleaned DataFrame
        """
        try:
            self.logger.info(f"Starting comprehensive data cleaning for {dataset_type} dataset")
            df_clean = df.copy()
            initial_shape = df_clean.shape
            
            # 1. Clean timestamps
            df_clean = self._clean_timestamps(df_clean)
            
            # 2. Clean and validate values
            df_clean = self._clean_values(df_clean, dataset_type)
            
            # 3. Remove invalid rows
            df_clean = self._remove_invalid_rows(df_clean, dataset_type)
            
            # 4. Validate data integrity
            df_clean = self._validate_data_integrity(df_clean, dataset_type)
            
            # 5. Apply fraud-specific data quality improvements
            df_clean = self._improve_fraud_data_quality(df_clean, dataset_type)
            
            final_shape = df_clean.shape
            removed_rows = initial_shape[0] - final_shape[0]
            
            if removed_rows > 0:
                self.logger.info(f"Removed {removed_rows} invalid rows during cleaning")
            
            self.logger.info(f"Data cleaning completed. Shape: {initial_shape} -> {final_shape}")
            return df_clean
            
        except Exception as e:
            self.logger.error(f"Error during data cleaning: {str(e)}")
            raise
    
    def _improve_fraud_data_quality(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """
        Apply fraud-specific data quality improvements.
        
        Args:
            df: DataFrame to improve
            dataset_type: Type of dataset
            
        Returns:
            Improved DataFrame
        """
        try:
            if dataset_type == "fraud":
                # Remove suspicious patterns that might indicate data quality issues
                
                # 1. Remove transactions with impossible time differences
                if 'signup_time' in df.columns and 'purchase_time' in df.columns:
                    df['time_diff_hours'] = (pd.to_datetime(df['purchase_time']) - 
                                           pd.to_datetime(df['signup_time'])).dt.total_seconds() / 3600
                    
                    # Remove transactions where purchase happens before signup
                    invalid_time_mask = df['time_diff_hours'] < 0
                    if invalid_time_mask.any():
                        self.logger.warning(f"Removing {invalid_time_mask.sum()} transactions with invalid time sequence")
                        df = df[~invalid_time_mask]
                
                # 2. Remove transactions with suspicious purchase values
                if 'purchase_value' in df.columns:
                    # Remove zero or negative purchase values
                    invalid_value_mask = df['purchase_value'] <= 0
                    if invalid_value_mask.any():
                        self.logger.warning(f"Removing {invalid_value_mask.sum()} transactions with invalid purchase values")
                        df = df[~invalid_value_mask]
                
                # 3. Remove transactions with missing critical fields
                critical_fields = ['user_id', 'device_id', 'ip_address']
                for field in critical_fields:
                    if field in df.columns:
                        missing_mask = df[field].isnull() | (df[field] == '')
                        if missing_mask.any():
                            self.logger.warning(f"Removing {missing_mask.sum()} transactions with missing {field}")
                            df = df[~missing_mask]
                
                # 4. Remove duplicate transactions (same user, same time, same amount)
                if all(field in df.columns for field in ['user_id', 'purchase_time', 'purchase_value']):
                    duplicate_mask = df.duplicated(subset=['user_id', 'purchase_time', 'purchase_value'], keep='first')
                    if duplicate_mask.any():
                        self.logger.warning(f"Removing {duplicate_mask.sum()} duplicate transactions")
                        df = df[~duplicate_mask]
                
                # 5. Add data quality flags
                df['data_quality_score'] = 1.0
                
                # Reduce quality score for suspicious patterns
                if 'purchase_value' in df.columns:
                    # High value transactions might need more scrutiny
                    high_value_mask = df['purchase_value'] > df['purchase_value'].quantile(0.95)
                    df.loc[high_value_mask, 'data_quality_score'] *= 0.9
                
                if 'age' in df.columns:
                    # Very young or old users might be suspicious
                    age_suspicious_mask = (df['age'] < 18) | (df['age'] > 80)
                    df.loc[age_suspicious_mask, 'data_quality_score'] *= 0.8
                
                self.logger.info(f"Applied fraud-specific data quality improvements")
                self.logger.info(f"Average data quality score: {df['data_quality_score'].mean():.3f}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error improving fraud data quality: {str(e)}")
            return df
    
    def _clean_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate timestamp data."""
        df_clean = df.copy()
        
        # Convert timestamp columns to datetime with error handling
        timestamp_cols = ['signup_time', 'purchase_time']
        for col in timestamp_cols:
            if col in df_clean.columns:
                # Convert to datetime, invalid dates become NaT
                df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                
                # Log invalid timestamps
                invalid_count = df_clean[col].isnull().sum()
                if invalid_count > 0:
                    self.logger.warning(f"Found {invalid_count} invalid timestamps in {col}")
                
                # Remove rows with invalid timestamps
                df_clean = df_clean.dropna(subset=[col])
        
        # Validate time logic (purchase_time should be after signup_time)
        if 'signup_time' in df_clean.columns and 'purchase_time' in df_clean.columns:
            invalid_time_order = df_clean['purchase_time'] <= df_clean['signup_time']
            invalid_count = invalid_time_order.sum()
            
            if invalid_count > 0:
                self.logger.warning(f"Found {invalid_count} rows where purchase_time <= signup_time")
                df_clean = df_clean[~invalid_time_order]
        
        return df_clean
    
    def _clean_values(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """Clean and validate numerical values."""
        df_clean = df.copy()
        
        if dataset_type == "fraud":
            # Clean purchase_value
            if 'purchase_value' in df_clean.columns:
                # Remove negative or zero values
                invalid_purchase = (df_clean['purchase_value'] <= 0)
                invalid_count = invalid_purchase.sum()
                if invalid_count > 0:
                    self.logger.warning(f"Removing {invalid_count} rows with invalid purchase_value")
                    df_clean = df_clean[~invalid_purchase]
                
                # Remove extreme outliers (top 0.1% and bottom 0.1%)
                q_low = df_clean['purchase_value'].quantile(0.001)
                q_high = df_clean['purchase_value'].quantile(0.999)
                outliers = (df_clean['purchase_value'] < q_low) | (df_clean['purchase_value'] > q_high)
                outlier_count = outliers.sum()
                if outlier_count > 0:
                    self.logger.warning(f"Removing {outlier_count} extreme purchase_value outliers")
                    df_clean = df_clean[~outliers]
            
            # Clean age values
            if 'age' in df_clean.columns:
                # Remove invalid ages (negative or > 120)
                invalid_age = (df_clean['age'] < 0) | (df_clean['age'] > 120)
                invalid_count = invalid_age.sum()
                if invalid_count > 0:
                    self.logger.warning(f"Removing {invalid_count} rows with invalid age")
                    df_clean = df_clean[~invalid_age]
        
        elif dataset_type == "creditcard":
            # Clean Amount column
            if 'Amount' in df_clean.columns:
                # Remove negative amounts
                invalid_amount = df_clean['Amount'] < 0
                invalid_count = invalid_amount.sum()
                if invalid_count > 0:
                    self.logger.warning(f"Removing {invalid_count} rows with negative Amount")
                    df_clean = df_clean[~invalid_amount]
                
                # Remove extreme outliers
                q_low = df_clean['Amount'].quantile(0.001)
                q_high = df_clean['Amount'].quantile(0.999)
                outliers = (df_clean['Amount'] < q_low) | (df_clean['Amount'] > q_high)
                outlier_count = outliers.sum()
                if outlier_count > 0:
                    self.logger.warning(f"Removing {outlier_count} extreme Amount outliers")
                    df_clean = df_clean[~outliers]
        
        return df_clean
    
    def _remove_invalid_rows(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """Remove rows with invalid data patterns."""
        df_clean = df.copy()
        initial_rows = len(df_clean)
        
        if dataset_type == "fraud":
            # Remove rows with missing critical fields
            critical_cols = ['user_id', 'purchase_value', 'class']
            missing_critical = df_clean[critical_cols].isnull().any(axis=1)
            missing_count = missing_critical.sum()
            if missing_count > 0:
                self.logger.warning(f"Removing {missing_count} rows with missing critical fields")
                df_clean = df_clean[~missing_critical]
            
            # Remove duplicate user_id + purchase_time combinations
            if 'user_id' in df_clean.columns and 'purchase_time' in df_clean.columns:
                duplicates = df_clean.duplicated(subset=['user_id', 'purchase_time'], keep='first')
                duplicate_count = duplicates.sum()
                if duplicate_count > 0:
                    self.logger.warning(f"Removing {duplicate_count} duplicate user_id + purchase_time combinations")
                    df_clean = df_clean[~duplicates]
        
        elif dataset_type == "creditcard":
            # Remove rows with missing critical fields
            critical_cols = ['Amount', 'Class']
            missing_critical = df_clean[critical_cols].isnull().any(axis=1)
            missing_count = missing_critical.sum()
            if missing_count > 0:
                self.logger.warning(f"Removing {missing_count} rows with missing critical fields")
                df_clean = df_clean[~missing_critical]
        
        removed_rows = initial_rows - len(df_clean)
        if removed_rows > 0:
            self.logger.info(f"Removed {removed_rows} invalid rows")
        
        return df_clean
    
    def _validate_data_integrity(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """Validate data integrity and consistency."""
        df_clean = df.copy()
        
        if dataset_type == "fraud":
            # Validate class distribution
            if 'class' in df_clean.columns:
                class_counts = df_clean['class'].value_counts()
                fraud_rate = df_clean['class'].mean()
                self.logger.info(f"Class distribution: {class_counts.to_dict()}")
                self.logger.info(f"Fraud rate: {fraud_rate:.3f}")
                
                # Warn if fraud rate is too high or too low
                if fraud_rate > 0.5:
                    self.logger.warning(f"Unusually high fraud rate: {fraud_rate:.3f}")
                elif fraud_rate < 0.01:
                    self.logger.warning(f"Unusually low fraud rate: {fraud_rate:.3f}")
            
            # Validate purchase value distribution
            if 'purchase_value' in df_clean.columns:
                purchase_stats = df_clean['purchase_value'].describe()
                self.logger.info(f"Purchase value statistics: {purchase_stats.to_dict()}")
        
        elif dataset_type == "creditcard":
            # Validate class distribution
            if 'Class' in df_clean.columns:
                class_counts = df_clean['Class'].value_counts()
                fraud_rate = df_clean['Class'].mean()
                self.logger.info(f"Class distribution: {class_counts.to_dict()}")
                self.logger.info(f"Fraud rate: {fraud_rate:.3f}")
            
            # Validate amount distribution
            if 'Amount' in df_clean.columns:
                amount_stats = df_clean['Amount'].describe()
                self.logger.info(f"Amount statistics: {amount_stats.to_dict()}")
        
        return df_clean

    def generate_synthetic_fraud_scenarios(self, df: pd.DataFrame, target_col: str = 'class') -> pd.DataFrame:
        """
        Generate synthetic fraud scenarios to improve training data quality.
        
        Args:
            df: Original DataFrame
            target_col: Target column name
            
        Returns:
            DataFrame with synthetic fraud scenarios added
        """
        try:
            self.logger.info("Generating synthetic fraud scenarios")
            
            # Get existing fraud cases for pattern analysis
            fraud_cases = df[df[target_col] == 1].copy()
            legitimate_cases = df[df[target_col] == 0].copy()
            
            if len(fraud_cases) == 0:
                self.logger.warning("No existing fraud cases found for pattern analysis")
                return df
            
            synthetic_fraud_cases = []
            
            # 1. VELOCITY FRAUD SCENARIOS (Burst transactions)
            self.logger.info("Creating velocity fraud scenarios")
            for _ in range(min(20, len(legitimate_cases))):
                base_case = legitimate_cases.sample(n=1).iloc[0].copy()
                
                # Create burst transaction pattern
                base_case['purchase_value'] = base_case['purchase_value'] * np.random.uniform(1.5, 3.0)
                base_case['purchase_time'] = pd.to_datetime(base_case['purchase_time']) + pd.Timedelta(minutes=np.random.randint(1, 10))
                base_case[target_col] = 1
                
                synthetic_fraud_cases.append(base_case)
            
            # 2. GEOGRAPHIC FRAUD SCENARIOS (Location anomalies)
            self.logger.info("Creating geographic fraud scenarios")
            if 'country' in df.columns:
                for _ in range(min(15, len(legitimate_cases))):
                    base_case = legitimate_cases.sample(n=1).iloc[0].copy()
                    
                    # Change country to create geographic anomaly
                    base_case['country'] = 'Unknown'  # Suspicious location
                    base_case['purchase_value'] = base_case['purchase_value'] * np.random.uniform(2.0, 4.0)
                    base_case[target_col] = 1
                    
                    synthetic_fraud_cases.append(base_case)
            
            # 3. DEVICE FRAUD SCENARIOS (Device sharing)
            self.logger.info("Creating device fraud scenarios")
            if 'device_id' in df.columns:
                # Find devices used by multiple users
                device_users = df.groupby('device_id')['user_id'].nunique()
                shared_devices = device_users[device_users > 1].index.tolist()
                
                for _ in range(min(10, len(legitimate_cases))):
                    base_case = legitimate_cases.sample(n=1).iloc[0].copy()
                    
                    # Use shared device
                    if shared_devices:
                        base_case['device_id'] = np.random.choice(shared_devices)
                    base_case['purchase_value'] = base_case['purchase_value'] * np.random.uniform(1.8, 3.5)
                    base_case[target_col] = 1
                    
                    synthetic_fraud_cases.append(base_case)
            
            # 4. TIME-BASED FRAUD SCENARIOS (Late night, fast fraud)
            self.logger.info("Creating time-based fraud scenarios")
            for _ in range(min(15, len(legitimate_cases))):
                base_case = legitimate_cases.sample(n=1).iloc[0].copy()
                
                # Late night transaction
                purchase_time = pd.to_datetime(base_case['purchase_time'])
                late_night_time = purchase_time.replace(hour=np.random.choice([0, 1, 2, 3, 23]))
                base_case['purchase_time'] = late_night_time
                
                # High value transaction
                base_case['purchase_value'] = base_case['purchase_value'] * np.random.uniform(2.5, 5.0)
                base_case[target_col] = 1
                
                synthetic_fraud_cases.append(base_case)
            
            # 5. AMOUNT-BASED FRAUD SCENARIOS (Suspicious amounts)
            self.logger.info("Creating amount-based fraud scenarios")
            suspicious_amounts = [0.01, 1.00, 10.00, 100.00, 500.00, 1000.00]
            
            for _ in range(min(10, len(legitimate_cases))):
                base_case = legitimate_cases.sample(n=1).iloc[0].copy()
                
                # Use suspicious amount
                base_case['purchase_value'] = np.random.choice(suspicious_amounts)
                base_case[target_col] = 1
                
                synthetic_fraud_cases.append(base_case)
            
            # 6. BEHAVIORAL ANOMALY SCENARIOS (Unusual patterns)
            self.logger.info("Creating behavioral anomaly scenarios")
            for _ in range(min(10, len(legitimate_cases))):
                base_case = legitimate_cases.sample(n=1).iloc[0].copy()
                
                # Multiple anomalies combined
                base_case['purchase_value'] = base_case['purchase_value'] * np.random.uniform(3.0, 6.0)  # Very high amount
                base_case['purchase_time'] = pd.to_datetime(base_case['purchase_time']) + pd.Timedelta(hours=np.random.randint(22, 24))  # Late night
                if 'country' in base_case:
                    base_case['country'] = 'Unknown'  # Suspicious location
                base_case[target_col] = 1
                
                synthetic_fraud_cases.append(base_case)
            
            # Combine original data with synthetic fraud cases
            synthetic_df = pd.DataFrame(synthetic_fraud_cases)
            combined_df = pd.concat([df, synthetic_df], ignore_index=True)
            
            # Shuffle the data
            combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            # Log the results
            original_fraud_count = len(fraud_cases)
            synthetic_fraud_count = len(synthetic_fraud_cases)
            total_fraud_count = len(combined_df[combined_df[target_col] == 1])
            
            self.logger.info(f"Generated {synthetic_fraud_count} synthetic fraud scenarios")
            self.logger.info(f"Original fraud cases: {original_fraud_count}")
            self.logger.info(f"Total fraud cases: {total_fraud_count}")
            self.logger.info(f"New fraud rate: {total_fraud_count / len(combined_df):.3f}")
            
            return combined_df
            
        except Exception as e:
            self.logger.error(f"Error generating synthetic fraud scenarios: {str(e)}")
            return df


 