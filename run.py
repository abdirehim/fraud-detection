#!/usr/bin/env python3
"""
Main execution script for the fraud detection pipeline.

This script demonstrates the complete pipeline from data loading
to model training and evaluation.
"""

import sys
from pathlib import Path
import argparse
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data_loader import DataLoader
from src.preprocess import DataPreprocessor
from src.models.train import ModelTrainer
from src.models.evaluate import ModelEvaluator
from src.explainability import ModelExplainer
from src.utils import setup_logging, create_experiment_dir, save_model, save_metrics


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Fraud Detection Pipeline")
    parser.add_argument("--data-path", type=str, help="Path to input data file")
    parser.add_argument("--output-dir", type=str, default="experiments", help="Output directory for results")
    parser.add_argument("--resampling", type=str, default="class_weights", 
                       choices=["smote", "undersample", "smoteenn", "adasyn", "borderline_smote", "class_weights", "none"],
                       help="Resampling method for imbalanced data")
    parser.add_argument("--target-col", type=str, default="fraud", help="Target column name")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging("fraud_detection_pipeline", log_level=log_level)
    
    logger.info("Starting Fraud Detection Pipeline")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        # Create experiment directory
        experiment_dir = create_experiment_dir("fraud_detection", Path(args.output_dir))
        logger.info(f"Experiment directory: {experiment_dir}")
        
        # Step 1: Load Data
        logger.info("Step 1: Loading data")
        loader = DataLoader()
        
        if args.data_path:
            # Load from specific file
            if args.data_path.endswith('.csv'):
                df = loader.load_csv_data(args.data_path)
            elif args.data_path.endswith('.parquet'):
                df = loader.load_parquet_data(args.data_path)
            else:
                raise ValueError(f"Unsupported file format: {args.data_path}")
        else:
            # Load all real datasets
            logger.info("Loading all fraud detection datasets")
            datasets = loader.load_all_datasets()
            
            # Use fraud data with geolocation if available, otherwise use regular fraud data
            if 'fraud_data_with_geo' in datasets:
                df = datasets['fraud_data_with_geo']
                logger.info("Using fraud data with geolocation information")
            elif 'fraud_data' in datasets:
                df = datasets['fraud_data']
                logger.info("Using e-commerce fraud data")
            elif 'creditcard_data' in datasets:
                df = datasets['creditcard_data']
                logger.info("Using credit card fraud data")
                args.target_col = 'Class'  # Update target column for credit card data
            else:
                raise ValueError("No fraud detection datasets found in data/raw/")
        
        logger.info(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Determine target column based on dataset
        if args.target_col not in df.columns:
            if 'class' in df.columns:
                args.target_col = 'class'
            elif 'Class' in df.columns:
                args.target_col = 'Class'
            else:
                raise ValueError(f"Target column '{args.target_col}' not found in dataset")
        
        logger.info(f"Using target column: {args.target_col}")
        logger.info(f"Fraud rate: {df[args.target_col].mean():.3f}")
        
        # Step 2: Preprocess Data
        logger.info("Step 2: Preprocessing data")
        preprocessor = DataPreprocessor()
        df_processed = preprocessor.fit_transform(df, args.target_col)
        
        logger.info(f"Processed data: {df_processed.shape[0]} rows, {df_processed.shape[1]} columns")
        
        # Save processed data
        processed_data_path = experiment_dir / "processed_data.parquet"
        df_processed.to_parquet(processed_data_path, index=False)
        logger.info(f"Saved processed data to {processed_data_path}")
        
        # Step 3: Train Models
        logger.info("Step 3: Training models")
        trainer = ModelTrainer()
        
        # Prepare data for training
        X_train, X_test, y_train, y_test = trainer.prepare_data(df_processed, args.target_col)
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        # Train all models
        models = trainer.train_all_models(X_train, y_train, args.resampling)
        logger.info(f"Trained {len(models)} models")
        
        # Save models
        models_dir = experiment_dir / "models"
        trainer.save_models(models_dir)
        
        # Step 4: Evaluate Models
        logger.info("Step 4: Evaluating models")
        
        # Convert test set target to binary format for evaluation
        y_test_binary = trainer.convert_target_to_binary(y_test)
        logger.info(f"Converted test target to binary: {y_test_binary.value_counts().to_dict()}")
        
        evaluator = ModelEvaluator()
        evaluation_results = evaluator.evaluate_all_models(models, X_test, y_test_binary)
        
        # Save evaluation results
        evaluation_dir = experiment_dir / "evaluation"
        evaluator.save_evaluation_results(evaluation_dir)
        
        # Step 5: Model Explainability
        logger.info("Step 5: Model explainability")
        explainer = ModelExplainer()
        
        # Get best model for explainability
        best_model_name, best_model = trainer.get_best_model()
        logger.info(f"Best model: {best_model_name}")
        
        # Fit SHAP explainer
        explainer.fit_shap_explainer(best_model, X_train, model_type="tree")
        
        # Generate explanation report
        explanation_report = explainer.generate_explanation_report(
            best_model, X_test, y_test_binary, 
            output_path=experiment_dir / "explanation_report.json"
        )
        
        # Step 6: Summary
        logger.info("Step 6: Pipeline summary")
        
        # Print best model performance
        best_eval = evaluation_results[best_model_name]
        metrics = best_eval['metrics']
        
        print("\n" + "="*50)
        print("FRAUD DETECTION PIPELINE RESULTS")
        print("="*50)
        print(f"Best Model: {best_model_name}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        if 'roc_auc' in metrics:
            print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        if 'pr_auc' in metrics:
            print(f"PR AUC: {metrics['pr_auc']:.4f}")
        print(f"\nExperiment directory: {experiment_dir}")
        print("="*50)
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main() 