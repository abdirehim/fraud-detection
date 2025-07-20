"""
Data loading utilities for the fraud detection pipeline.

This module handles loading and validation of financial transaction data
with proper error handling and logging.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import logging
import ipaddress

from .config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from .utils import setup_logging, validate_data_path, get_file_size_mb


class DataLoader:
    """
    Data loader class for handling financial transaction datasets.
    
    This class provides methods to load, validate, and preprocess
    financial data for fraud detection tasks.
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize the DataLoader.
        
        Args:
            data_dir: Directory containing the data files
        """
        self.data_dir = data_dir or RAW_DATA_DIR
        self.logger = setup_logging("data_loader")
        self.logger.info(f"Initialized DataLoader with data directory: {self.data_dir}")
        
    def load_csv_data(
        self, 
        filename: str, 
        encoding: str = "utf-8",
        **kwargs
    ) -> pd.DataFrame:
        """
        Load data from a CSV file with error handling.
        
        Args:
            filename: Name of the CSV file
            encoding: File encoding
            **kwargs: Additional arguments to pass to pd.read_csv
            
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            pd.errors.EmptyDataError: If the file is empty
            pd.errors.ParserError: If there are parsing errors
        """
        filepath = self.data_dir / filename
        
        try:
            # Validate file path
            if not validate_data_path(filepath):
                raise FileNotFoundError(f"Data file not found: {filepath}")
            
            # Log file information
            file_size = get_file_size_mb(filepath)
            self.logger.info(f"Loading CSV file: {filename} (Size: {file_size} MB)")
            
            # Load the data
            df = pd.read_csv(filepath, encoding=encoding, **kwargs)
            
            # Validate loaded data
            if df.empty:
                raise pd.errors.EmptyDataError(f"CSV file is empty: {filename}")
            
            self.logger.info(f"Successfully loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
            self.logger.info(f"Columns: {list(df.columns)}")
            
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
        Load the e-commerce fraud dataset (Fraud_Data.csv).
        
        Returns:
            DataFrame with e-commerce transaction data
        """
        try:
            self.logger.info("Loading e-commerce fraud dataset")
            df = self.load_csv_data("Fraud_Data.csv")
            
            # Apply comprehensive data cleaning
            df = self.clean_raw_data(df, dataset_type="fraud")
            
            # Log class distribution after cleaning
            if 'class' in df.columns:
                fraud_rate = df['class'].mean()
                self.logger.info(f"E-commerce fraud rate after cleaning: {fraud_rate:.3f}")
                self.logger.info(f"Class distribution after cleaning: {df['class'].value_counts().to_dict()}")
            
            # Save cleaned data to processed directory
            cleaned_file_path = PROCESSED_DATA_DIR / "cleaned_fraud_data.csv"
            df.to_csv(cleaned_file_path, index=False)
            self.logger.info(f"Saved cleaned fraud data to: {cleaned_file_path}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading fraud data: {str(e)}")
            raise

    def load_creditcard_data(self) -> pd.DataFrame:
        """
        Load the credit card fraud dataset (creditcard.csv).
        
        Returns:
            DataFrame with credit card transaction data
        """
        try:
            self.logger.info("Loading credit card fraud dataset")
            df = self.load_csv_data("creditcard.csv")
            
            # Apply comprehensive data cleaning
            df = self.clean_raw_data(df, dataset_type="creditcard")
            
            # Log class distribution after cleaning
            if 'Class' in df.columns:
                fraud_rate = df['Class'].mean()
                self.logger.info(f"Credit card fraud rate after cleaning: {fraud_rate:.3f}")
                self.logger.info(f"Class distribution after cleaning: {df['Class'].value_counts().to_dict()}")
            
            # Save cleaned data to processed directory
            cleaned_file_path = PROCESSED_DATA_DIR / "cleaned_creditcard_data.csv"
            df.to_csv(cleaned_file_path, index=False)
            self.logger.info(f"Saved cleaned creditcard data to: {cleaned_file_path}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading credit card data: {str(e)}")
            raise

    def load_ip_country_mapping(self) -> pd.DataFrame:
        """
        Load the IP address to country mapping dataset (IpAddress_to_Country.csv).
        
        Returns:
            DataFrame with IP address ranges and country mappings
        """
        try:
            self.logger.info("Loading IP address to country mapping")
            df = self.load_csv_data("IpAddress_to_Country.csv")
            
            # Convert IP addresses to integers for efficient lookup
            if 'lower_bound_ip_address' in df.columns:
                df['lower_bound_ip_int'] = df['lower_bound_ip_address'].apply(
                    lambda x: int(ipaddress.IPv4Address(x))
                )
            if 'upper_bound_ip_address' in df.columns:
                df['upper_bound_ip_int'] = df['upper_bound_ip_address'].apply(
                    lambda x: int(ipaddress.IPv4Address(x))
                )
            
            self.logger.info(f"Loaded {len(df)} IP address ranges")
            self.logger.info(f"Countries covered: {df['country'].nunique()}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading IP country mapping: {str(e)}")
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
            
            final_shape = df_clean.shape
            removed_rows = initial_shape[0] - final_shape[0]
            
            if removed_rows > 0:
                self.logger.info(f"Removed {removed_rows} invalid rows during cleaning")
            
            self.logger.info(f"Data cleaning completed. Shape: {initial_shape} -> {final_shape}")
            return df_clean
            
        except Exception as e:
            self.logger.error(f"Error during data cleaning: {str(e)}")
            raise
    
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


 