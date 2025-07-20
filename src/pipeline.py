"""
Complete fraud detection pipeline demonstrating the full data flow.

This module shows how all components work together:
1. Data Loading with Cleaning
2. Feature Engineering  
3. Data Transformation
4. Class Imbalance Handling
5. Model Training
6. Model Evaluation
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any
import logging
import os

from .data_loader import DataLoader
from .preprocess import DataPreprocessor
from .models.train import ModelTrainer
from .models.evaluate import ModelEvaluator
from .explainability import SHAPExplainer
from .utils import setup_logging
from .config import PROCESSED_DATA_DIR


class FraudDetectionPipeline:
    """
    Complete fraud detection pipeline that orchestrates all components.
    
    This class demonstrates the proper implementation locations for:
    - Data cleaning and validation
    - Feature engineering
    - Data transformation (scaling, encoding)
    - Class imbalance handling
    - Model training and evaluation
    """
    
    def __init__(self):
        """Initialize the pipeline components."""
        self.logger = setup_logging("fraud_pipeline")
        self.data_loader = DataLoader()
        self.preprocessor = DataPreprocessor()
        self.trainer = ModelTrainer()
        self.evaluator = ModelEvaluator()
        self.explainer = SHAPExplainer()
        
        self.logger.info("Initialized Fraud Detection Pipeline")
    
    def run_complete_pipeline(self, dataset_name: str = "fraud_data_with_geo") -> Dict[str, Any]:
        """
        Run the complete fraud detection pipeline.
        
        Args:
            dataset_name: Name of the dataset to use
            
        Returns:
            Dictionary containing all pipeline results
        """
        try:
            self.logger.info("Starting complete fraud detection pipeline")
            
            # Step 1: Data Loading with Cleaning
            df_cleaned = self._load_and_clean_data(dataset_name)
            
            # Step 2: Feature Engineering
            df_engineered = self._engineer_features(df_cleaned)
            
            # Step 3: Data Transformation
            df_transformed = self._transform_data(df_engineered)
            
            # Step 4: Model Training with Class Imbalance Handling
            models, results = self._train_models(df_transformed)
            
            # Step 5: Model Evaluation
            evaluation_results = self._evaluate_models(models, df_transformed)
            
            # Step 6: Model Explainability
            explainability_results = self._explain_models(models, df_transformed)
            
            # Compile all results
            pipeline_results = {
                "data_cleaning": {
                    "initial_shape": df_cleaned.shape,
                    "cleaning_summary": "Data cleaned with timestamp validation, outlier removal, and integrity checks"
                },
                "feature_engineering": {
                    "final_shape": df_engineered.shape,
                    "features_added": df_engineered.shape[1] - df_cleaned.shape[1],
                    "feature_summary": "Time-based, user behavior, device behavior, and value-based features"
                },
                "data_transformation": {
                    "scaling_applied": "StandardScaler for numerical features",
                    "encoding_applied": "LabelEncoder for categorical features",
                    "feature_selection": "Statistical feature selection applied"
                },
                "class_imbalance": {
                    "method_used": "SMOTE for oversampling",
                    "original_ratio": "Imbalanced (5.6% fraud rate)",
                    "resampled_ratio": "Balanced for training"
                },
                "model_training": results,
                "model_evaluation": evaluation_results,
                "model_explainability": explainability_results
            }
            
            self.logger.info("Pipeline completed successfully")
            return pipeline_results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    def _load_and_clean_data(self, dataset_name: str) -> pd.DataFrame:
        """
        Step 1: Load and clean data.
        
        IMPLEMENTATION LOCATION: src/data_loader.py
        - Timestamp cleaning and validation
        - Value validation and outlier removal
        - Data integrity checks
        """
        self.logger.info("Step 1: Loading and cleaning data")
        
        # Load all datasets
        datasets = self.data_loader.load_all_datasets()
        
        if dataset_name not in datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found. Available: {list(datasets.keys())}")
        
        df = datasets[dataset_name]
        self.logger.info(f"Loaded dataset: {df.shape}")
        
        # Data cleaning is already applied in data_loader.load_fraud_data()
        # and data_loader.load_creditcard_data() methods
        
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 2: Engineer features.
        
        IMPLEMENTATION LOCATION: src/preprocess.py
        - Time-based features (hour, day, month, time differences)
        - User behavior features (purchase patterns, repeat users)
        - Device behavior features (device usage patterns)
        - Value-based features (amount categories, log transformations)
        """
        self.logger.info("Step 2: Engineering features")
        
        # Apply feature engineering
        df_engineered = self.preprocessor.engineer_features(df)
        
        self.logger.info(f"Feature engineering completed: {df.shape[1]} -> {df_engineered.shape[1]} features")
        
        return df_engineered
    
    def _transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 3: Transform data (scaling, encoding, feature selection).
        
        IMPLEMENTATION LOCATION: src/preprocess.py
        - Categorical encoding (LabelEncoder)
        - Feature scaling (StandardScaler)
        - Feature selection (statistical tests)
        """
        self.logger.info("Step 3: Transforming data")
        
        # Determine target column
        target_col = None
        if 'class' in df.columns:
            target_col = 'class'
        elif 'Class' in df.columns:
            target_col = 'Class'
        else:
            raise ValueError("No target column found")
        
        # Apply data transformation pipeline
        df_encoded = self.preprocessor.encode_categorical_features(df)
        df_scaled = self.preprocessor.scale_features(df_encoded)
        df_selected = self.preprocessor.select_features(df_scaled, target_col)
        
        self.logger.info(f"Data transformation completed: {df.shape[1]} -> {df_selected.shape[1]} features")
        
        return df_selected
    
    def _train_models(self, df: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Step 4: Train models with class imbalance handling.
        
        IMPLEMENTATION LOCATION: 
        - Class imbalance: src/preprocess.py (handle_imbalanced_data)
        - Model training: src/models/train.py
        """
        self.logger.info("Step 4: Training models with class imbalance handling")
        
        # Determine target column
        target_col = None
        if 'class' in df.columns:
            target_col = 'class'
        elif 'Class' in df.columns:
            target_col = 'Class'
        
        # Prepare data for training
        X_train, X_test, y_train, y_test = self.trainer.prepare_data(df, target_col)
        
        # Handle class imbalance BEFORE training
        X_train_resampled, y_train_resampled = self.preprocessor.handle_imbalanced_data(
            X_train, y_train, method="smote"
        )
        
        self.logger.info(f"Class imbalance handled: {len(y_train)} -> {len(y_train_resampled)} samples")
        self.logger.info(f"Resampled fraud rate: {y_train_resampled.mean():.3f}")
        
        # Save model-ready data (after resampling)
        model_ready_data = pd.concat([X_train_resampled, y_train_resampled], axis=1)
        model_ready_file_path = PROCESSED_DATA_DIR / "model_ready_fraud_data.csv"
        model_ready_data.to_csv(model_ready_file_path, index=False)
        self.logger.info(f"Saved model-ready data to: {model_ready_file_path}")
        
        # Save resampling metadata
        resampling_metadata = {
            "original_samples": len(y_train),
            "resampled_samples": len(y_train_resampled),
            "original_fraud_rate": y_train.mean(),
            "resampled_fraud_rate": y_train_resampled.mean(),
            "resampling_method": "smote",
            "features_count": X_train_resampled.shape[1],
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        resampling_file = PROCESSED_DATA_DIR / "resampling_metadata.json"
        with open(resampling_file, 'w') as f:
            import json
            json.dump(resampling_metadata, f, indent=2)
        self.logger.info(f"Saved resampling metadata to: {resampling_file}")
        
        # Train models on resampled data
        models = self.trainer.train_all_models(X_train_resampled, y_train_resampled)
        
        # Store test data for evaluation
        self.X_test = X_test
        self.y_test = y_test
        
        return models, {"training_samples": len(y_train_resampled), "test_samples": len(y_test)}
    
    def _evaluate_models(self, models: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """
        Step 5: Evaluate models.
        
        IMPLEMENTATION LOCATION: src/models/evaluate.py
        - Imbalanced metrics (AUC-PR, F1-Score)
        - Confusion matrices
        - Model comparison
        """
        self.logger.info("Step 5: Evaluating models")
        
        # Evaluate all models
        evaluation_results = self.evaluator.evaluate_all_models(
            models, self.X_test, self.y_test
        )
        
        self.logger.info("Model evaluation completed")
        
        return evaluation_results
    
    def _explain_models(self, models: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """
        Step 6: Model explainability.
        
        IMPLEMENTATION LOCATION: src/explainability.py
        - SHAP analysis
        - Feature importance
        - Prediction explanations
        """
        self.logger.info("Step 6: Model explainability")
        
        # Get best model for explanation
        best_model_name = max(models.keys(), key=lambda k: models[k].get('score', 0))
        best_model = models[best_model_name]['model']
        
        # Generate SHAP explanations
        explainability_results = self.explainer.explain_model(
            best_model, self.X_test, model_name=best_model_name
        )
        
        self.logger.info("Model explainability completed")
        
        return explainability_results
    
    def get_implementation_summary(self) -> Dict[str, str]:
        """
        Get a summary of where each component is implemented.
        
        Returns:
            Dictionary mapping component to implementation location
        """
        return {
            "Data Loading": "src/data_loader.py - load_csv_data(), load_fraud_data(), load_creditcard_data()",
            "Data Cleaning": "src/data_loader.py - clean_raw_data(), _clean_timestamps(), _clean_values()",
            "Feature Engineering": "src/preprocess.py - engineer_features(), _engineer_fraud_data_features()",
            "Data Transformation": "src/preprocess.py - encode_categorical_features(), scale_features(), select_features()",
            "Class Imbalance": "src/preprocess.py - handle_imbalanced_data()",
            "Model Training": "src/models/train.py - train_all_models(), prepare_data()",
            "Model Evaluation": "src/models/evaluate.py - evaluate_all_models()",
            "Model Explainability": "src/explainability.py - explain_model(), generate_shap_plots()"
        }


def run_pipeline_demo():
    """Run a demonstration of the complete pipeline."""
    try:
        # Initialize pipeline
        pipeline = FraudDetectionPipeline()
        
        # Get implementation summary
        implementation_summary = pipeline.get_implementation_summary()
        
        print("=== FRAUD DETECTION PIPELINE IMPLEMENTATION SUMMARY ===")
        for component, location in implementation_summary.items():
            print(f"{component}: {location}")
        
        print("\n=== RUNNING COMPLETE PIPELINE ===")
        
        # Run the complete pipeline
        results = pipeline.run_complete_pipeline()
        
        print("\n=== PIPELINE RESULTS SUMMARY ===")
        print(f"Data Cleaning: {results['data_cleaning']['cleaning_summary']}")
        print(f"Feature Engineering: Added {results['feature_engineering']['features_added']} features")
        print(f"Data Transformation: {results['data_transformation']['scaling_applied']}")
        print(f"Class Imbalance: {results['class_imbalance']['method_used']}")
        print(f"Model Training: {results['model_training']['training_samples']} training samples")
        
        return results
        
    except Exception as e:
        print(f"Pipeline demo failed: {str(e)}")
        return None


if __name__ == "__main__":
    run_pipeline_demo() 