"""
Model evaluation for fraud detection.

This module provides comprehensive evaluation metrics and visualization
tools for assessing fraud detection model performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
from sklearn.model_selection import cross_val_predict
from typing import Dict, Any, Optional, Tuple, List
import logging
from pathlib import Path

from ..config import EVALUATION_METRICS
from ..utils import setup_logging, save_metrics


class ModelEvaluator:
    """
    Model evaluator for fraud detection models.
    
    This class provides comprehensive evaluation metrics and visualization
    tools for assessing model performance on imbalanced datasets.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ModelEvaluator.
        
        Args:
            config: Configuration dictionary for evaluation
        """
        self.config = config or {}
        self.logger = setup_logging("model_evaluator")
        self.evaluation_results = {}
        
        self.logger.info("Initialized ModelEvaluator")
    
    def compute_metrics(
        self, 
        y_true: pd.Series, 
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional)
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            self.logger.info("Computing evaluation metrics")
            
            metrics = {}
            
            # Basic classification metrics
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
            metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
            
            # Probability-based metrics (if available)
            if y_prob is not None:
                try:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
                except ValueError:
                    metrics['roc_auc'] = 0.0
                    self.logger.warning("Could not compute ROC AUC (only one class present)")
                
                try:
                    metrics['pr_auc'] = average_precision_score(y_true, y_prob)
                except ValueError:
                    metrics['pr_auc'] = 0.0
                    self.logger.warning("Could not compute PR AUC (only one class present)")
            
            # Additional metrics for imbalanced datasets
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            metrics['true_negatives'] = int(tn)
            metrics['false_positives'] = int(fp)
            metrics['false_negatives'] = int(fn)
            metrics['true_positives'] = int(tp)
            
            # Derived metrics
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            metrics['balanced_accuracy'] = (metrics['sensitivity'] + metrics['specificity']) / 2
            
            # Log metrics
            for metric, value in metrics.items():
                if isinstance(value, float):
                    self.logger.info(f"{metric}: {value:.4f}")
                else:
                    self.logger.info(f"{metric}: {value}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error computing metrics: {str(e)}")
            raise
    
    def optimize_threshold(
        self, 
        y_true: pd.Series, 
        y_prob: np.ndarray,
        metric: str = "f1"
    ) -> Tuple[float, float]:
        """
        Optimize classification threshold for imbalanced datasets.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            metric: Metric to optimize ('f1', 'precision', 'recall')
            
        Returns:
            Tuple of (optimal_threshold, best_score)
        """
        try:
            thresholds = np.arange(0.1, 0.9, 0.05)
            best_score = 0
            optimal_threshold = 0.5
            
            for threshold in thresholds:
                y_pred = (y_prob >= threshold).astype(int)
                
                if metric == "f1":
                    score = f1_score(y_true, y_pred, zero_division=0)
                elif metric == "precision":
                    score = precision_score(y_true, y_pred, zero_division=0)
                elif metric == "recall":
                    score = recall_score(y_true, y_pred, zero_division=0)
                else:
                    score = f1_score(y_true, y_pred, zero_division=0)
                
                if score > best_score:
                    best_score = score
                    optimal_threshold = threshold
            
            self.logger.info(f"Optimal threshold: {optimal_threshold:.3f} (score: {best_score:.4f})")
            return optimal_threshold, best_score
            
        except Exception as e:
            self.logger.error(f"Error optimizing threshold: {str(e)}")
            return 0.5, 0.0
    
    def evaluate_model(
        self, 
        model: Any, 
        X_test: pd.DataFrame, 
        y_test: pd.Series,
        model_name: str = "model"
    ) -> Dict[str, Any]:
        """
        Evaluate a single model on test data with threshold optimization.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            model_name: Name of the model
            
        Returns:
            Dictionary containing evaluation results
        """
        try:
            self.logger.info(f"Evaluating {model_name}")
            
            # Get probabilities if available
            y_prob = None
            if hasattr(model, 'predict_proba'):
                try:
                    y_prob = model.predict_proba(X_test)[:, 1]  # Probability of positive class
                except Exception as e:
                    self.logger.warning(f"Could not get probabilities for {model_name}: {str(e)}")
            
            # Optimize threshold for fraud detection
            if y_prob is not None:
                optimal_threshold, best_f1 = self.optimize_threshold(y_test, y_prob, "f1")
                y_pred_optimized = (y_prob >= optimal_threshold).astype(int)
                
                # Compute metrics with optimized threshold
                metrics = self.compute_metrics(y_test, y_pred_optimized, y_prob)
                metrics['optimal_threshold'] = optimal_threshold
                metrics['best_f1_score'] = best_f1
                
                # Also compute default threshold metrics for comparison
                y_pred_default = model.predict(X_test)
                metrics_default = self.compute_metrics(y_test, y_pred_default, y_prob)
                metrics['default_threshold_metrics'] = metrics_default
                
            else:
                # Fallback to default predictions
                y_pred_optimized = model.predict(X_test)
                metrics = self.compute_metrics(y_test, y_pred_optimized, y_prob)
                metrics['optimal_threshold'] = 0.5
                metrics['best_f1_score'] = metrics['f1_score']
            
            # Create evaluation result
            evaluation_result = {
                'model_name': model_name,
                'metrics': metrics,
                'predictions': y_pred_optimized,
                'probabilities': y_prob,
                'true_labels': y_test.values,
                'optimal_threshold': metrics.get('optimal_threshold', 0.5)
            }
            
            self.evaluation_results[model_name] = evaluation_result
            self.logger.info(f"{model_name} evaluation completed")
            
            return evaluation_result
            
        except Exception as e:
            self.logger.error(f"Error evaluating {model_name}: {str(e)}")
            raise
    
    def evaluate_all_models(
        self, 
        models: Dict[str, Any], 
        X_test: pd.DataFrame, 
        y_test: pd.Series
    ) -> Dict[str, Any]:
        """
        Evaluate all models on test data.
        
        Args:
            models: Dictionary of trained models
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary containing evaluation results for all models
        """
        try:
            self.logger.info("Evaluating all models")
            
            for model_name, model_info in models.items():
                try:
                    model = model_info['model']
                    self.evaluate_model(model, X_test, y_test, model_name)
                except Exception as e:
                    self.logger.error(f"Failed to evaluate {model_name}: {str(e)}")
                    continue
            
            self.logger.info(f"Evaluation completed for {len(self.evaluation_results)} models")
            return self.evaluation_results
            
        except Exception as e:
            self.logger.error(f"Error in evaluate_all_models: {str(e)}")
            raise
    
    def plot_confusion_matrix(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        model_name: str = "Model",
        figsize: Tuple[int, int] = (8, 6)
    ) -> None:
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model for the plot title
            figsize: Figure size
        """
        try:
            cm = confusion_matrix(y_true, y_pred)
            
            plt.figure(figsize=figsize)
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Legitimate', 'Fraud'],
                yticklabels=['Legitimate', 'Fraud']
            )
            plt.title(f'Confusion Matrix - {model_name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.show()
            
            self.logger.info(f"Plotted confusion matrix for {model_name}")
            
        except Exception as e:
            self.logger.error(f"Error plotting confusion matrix: {str(e)}")
    
    def plot_roc_curve(
        self, 
        y_true: np.ndarray, 
        y_prob: np.ndarray,
        model_name: str = "Model",
        figsize: Tuple[int, int] = (8, 6)
    ) -> None:
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            model_name: Name of the model for the plot title
            figsize: Figure size
        """
        try:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc_score = roc_auc_score(y_true, y_prob)
            
            plt.figure(figsize=figsize)
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {model_name}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
            self.logger.info(f"Plotted ROC curve for {model_name}")
            
        except Exception as e:
            self.logger.error(f"Error plotting ROC curve: {str(e)}")
    
    def plot_precision_recall_curve(
        self, 
        y_true: np.ndarray, 
        y_prob: np.ndarray,
        model_name: str = "Model",
        figsize: Tuple[int, int] = (8, 6)
    ) -> None:
        """
        Plot precision-recall curve.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            model_name: Name of the model for the plot title
            figsize: Figure size
        """
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            pr_auc = average_precision_score(y_true, y_prob)
            
            plt.figure(figsize=figsize)
            plt.plot(recall, precision, label=f'{model_name} (PR-AUC = {pr_auc:.3f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve - {model_name}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
            self.logger.info(f"Plotted precision-recall curve for {model_name}")
            
        except Exception as e:
            self.logger.error(f"Error plotting precision-recall curve: {str(e)}")
    
    def plot_metrics_comparison(
        self, 
        metrics_dict: Dict[str, Dict[str, float]],
        metrics_to_plot: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (12, 8)
    ) -> None:
        """
        Plot comparison of metrics across models.
        
        Args:
            metrics_dict: Dictionary mapping model names to metrics
            metrics_to_plot: List of metrics to plot (default: all)
            figsize: Figure size
        """
        try:
            if not metrics_dict:
                self.logger.warning("No metrics to plot")
                return
            
            # Default metrics to plot
            if metrics_to_plot is None:
                metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc']
            
            # Filter available metrics
            available_metrics = []
            for metric in metrics_to_plot:
                if any(metric in model_metrics for model_metrics in metrics_dict.values()):
                    available_metrics.append(metric)
            
            if not available_metrics:
                self.logger.warning("No available metrics to plot")
                return
            
            # Create comparison DataFrame
            comparison_data = []
            for model_name, metrics in metrics_dict.items():
                row = {'model': model_name}
                for metric in available_metrics:
                    row[metric] = metrics.get(metric, 0.0)
                comparison_data.append(row)
            
            df_comparison = pd.DataFrame(comparison_data)
            
            # Create subplots
            n_metrics = len(available_metrics)
            n_cols = min(3, n_metrics)
            n_rows = (n_metrics + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
            if n_metrics == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes.reshape(1, -1)
            else:
                axes = axes.flatten()
            
            # Plot each metric
            for i, metric in enumerate(available_metrics):
                ax = axes[i]
                df_comparison.plot(x='model', y=metric, kind='bar', ax=ax)
                ax.set_title(f'{metric.replace("_", " ").title()}')
                ax.set_ylabel(metric.replace("_", " ").title())
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
            
            # Hide empty subplots
            for i in range(n_metrics, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.show()
            
            self.logger.info(f"Plotted metrics comparison for {len(available_metrics)} metrics")
            
        except Exception as e:
            self.logger.error(f"Error plotting metrics comparison: {str(e)}")
    
    def generate_evaluation_report(
        self, 
        output_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            output_path: Path to save the report (optional)
            
        Returns:
            Dictionary containing the evaluation report
        """
        try:
            self.logger.info("Generating evaluation report")
            
            report = {
                "summary": {},
                "detailed_results": {},
                "recommendations": []
            }
            
            # Summary statistics
            if self.evaluation_results:
                # Find best model by F1 score
                best_model = max(
                    self.evaluation_results.items(),
                    key=lambda x: x[1]['metrics'].get('f1_score', 0)
                )
                
                report["summary"] = {
                    "total_models_evaluated": len(self.evaluation_results),
                    "best_model": best_model[0],
                    "best_f1_score": best_model[1]['metrics'].get('f1_score', 0),
                    "models_evaluated": list(self.evaluation_results.keys())
                }
                
                # Detailed results
                for model_name, result in self.evaluation_results.items():
                    report["detailed_results"][model_name] = {
                        "metrics": result['metrics'],
                        "model_type": type(result.get('model', 'Unknown')).__name__
                    }
                
                # Recommendations
                recommendations = []
                
                # Check for low recall
                for model_name, result in self.evaluation_results.items():
                    recall = result['metrics'].get('recall', 0)
                    if recall < 0.7:
                        recommendations.append(f"{model_name}: Low recall ({recall:.3f}). Consider adjusting threshold or using different resampling.")
                
                # Check for low precision
                for model_name, result in self.evaluation_results.items():
                    precision = result['metrics'].get('precision', 0)
                    if precision < 0.7:
                        recommendations.append(f"{model_name}: Low precision ({precision:.3f}). Consider feature engineering or different model.")
                
                # Check for overfitting
                for model_name, result in self.evaluation_results.items():
                    accuracy = result['metrics'].get('accuracy', 0)
                    if accuracy > 0.95:
                        recommendations.append(f"{model_name}: Very high accuracy ({accuracy:.3f}). Potential overfitting - check cross-validation.")
                
                report["recommendations"] = recommendations
            
            # Save report if path provided
            if output_path:
                import json
                with open(output_path, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                self.logger.info(f"Evaluation report saved to {output_path}")
            
            self.logger.info("Evaluation report generated successfully")
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating evaluation report: {str(e)}")
            return {"error": str(e)}
    
    def save_evaluation_results(self, output_dir: Path) -> None:
        """
        Save evaluation results to disk.
        
        Args:
            output_dir: Directory to save results
        """
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            for model_name, result in self.evaluation_results.items():
                # Save metrics
                metrics_path = output_dir / f"{model_name}_metrics.json"
                save_metrics(result['metrics'], metrics_path)
                
                # Save predictions
                predictions_df = pd.DataFrame({
                    'true_labels': result['true_labels'],
                    'predictions': result['predictions']
                })
                if result['probabilities'] is not None:
                    predictions_df['probabilities'] = result['probabilities']
                
                predictions_path = output_dir / f"{model_name}_predictions.csv"
                predictions_df.to_csv(predictions_path, index=False)
            
            # Save overall evaluation report
            report_path = output_dir / "evaluation_report.json"
            self.generate_evaluation_report(report_path)
            
            self.logger.info(f"Saved evaluation results to {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving evaluation results: {str(e)}")
            raise 