"""
Evaluation Module

This module provides functions for evaluating machine learning models.
Includes comprehensive metrics, visualizations, and performance reports.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_metrics(y_true, y_pred, y_pred_proba) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities for positive class (1D array)
    
    Returns:
        Dictionary containing all evaluation metrics
    """
    logger.info("Calculating evaluation metrics...")
    
    metrics = {}
    
    # Basic metrics (weighted)
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Binary metrics (positive class)
    metrics['precision_positive'] = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    metrics['recall_positive'] = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    metrics['f1_positive'] = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    
    # ROC-AUC for binary classification
    try:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
    except Exception as e:
        logger.warning(f"Could not calculate ROC-AUC: {e}")
        metrics['roc_auc'] = None
    
    logger.info("Metrics calculated successfully")
    return metrics


def print_classification_report(y_true, y_pred, target_names: Optional[list] = None) -> None:
    """
    Print detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Optional list of class names
    """
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    
    report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
    print(report)


def print_metrics(metrics: Dict[str, float]) -> None:
    """
    Print evaluation metrics in a formatted way.
    
    Args:
        metrics: Dictionary of evaluation metrics
    """
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    
    # Print weighted metrics
    print("\nWeighted Metrics:")
    print("-"*60)
    for metric_name in ['accuracy', 'precision', 'recall', 'f1_score']:
        value = metrics.get(metric_name)
        if value is not None:
            print(f"{metric_name.upper():<15}: {value:.4f}")
    
    # Print positive class metrics
    print("\nPositive Class Metrics:")
    print("-"*60)
    for metric_name in ['precision_positive', 'recall_positive', 'f1_positive']:
        value = metrics.get(metric_name)
        if value is not None:
            print(f"{metric_name.upper():<15}: {value:.4f}")
    
    # Print ROC-AUC
    print("\nROC-AUC:")
    print("-"*60)
    value = metrics.get('roc_auc')
    if value is not None:
        print(f"ROC_AUC{' '*8}: {value:.4f}")
    else:
        print(f"ROC_AUC{' '*8}: N/A")


def plot_confusion_matrix(
    y_true,
    y_pred,
    save_path: str,
    class_names: Optional[list] = None,
    title: str = "Confusion Matrix"
) -> None:
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save the plot
        class_names: Optional list of class names
        title: Plot title
    """
    logger.info("Generating confusion matrix...")
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names if class_names else range(len(cm)),
                yticklabels=class_names if class_names else range(len(cm)),
                cbar_kws={'label': 'Count'})
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    
    # Save plot
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Confusion matrix saved to: {save_path}")


def plot_roc_curve(
    y_true,
    y_pred_proba,
    save_path: str,
    title: str = "ROC Curve"
) -> None:
    """
    Plot and save ROC curve for binary classification.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities for positive class (1D array)
        save_path: Path to save the plot
        title: Plot title
    """
    logger.info("Generating ROC curve...")
    
    plt.figure(figsize=(10, 8))
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
            label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"ROC curve saved to: {save_path}")


def evaluate_model(
    model,
    X_test,
    y_test,
    plots_dir: str,
    model_name: str = "XGBoost"
) -> Dict[str, Any]:
    """
    Complete model evaluation pipeline.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        plots_dir: Directory to save evaluation plots
        model_name: Name of the model for plot titles
    
    Returns:
        Dictionary containing all evaluation results
    """
    logger.info("="*60)
    logger.info("STARTING MODEL EVALUATION")
    logger.info("="*60)
    
    results = {}
    
    # Make predictions
    logger.info("Making predictions on test set...")
    y_pred = model.predict(X_test)
    from sklearn.metrics import accuracy_score

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2%}") 
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Get positive class probability only
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    results['metrics'] = metrics
    
    # Print metrics
    print_metrics(metrics)
    
    # Print classification report
    print_classification_report(y_test, y_pred)
    
    # Plot and save confusion matrix
    cm_path = Path(plots_dir) / f"{model_name.lower()}_confusion_matrix.png"
    plot_confusion_matrix(y_test, y_pred, str(cm_path), 
                         title=f"{model_name} - Confusion Matrix")
    results['confusion_matrix_path'] = str(cm_path)
    
    # Plot and save ROC curve
    roc_path = Path(plots_dir) / f"{model_name.lower()}_roc_curve.png"
    plot_roc_curve(y_test, y_pred_proba, str(roc_path),
                   title=f"{model_name} - ROC Curve")
    results['roc_curve_path'] = str(roc_path)
    
    logger.info("="*60)
    logger.info("MODEL EVALUATION COMPLETED")
    logger.info("="*60)
    
    return results


def save_evaluation_report(results: Dict[str, Any], report_path: str) -> None:
    """
    Save evaluation results to a text file.
    
    Args:
        results: Dictionary containing evaluation results
        report_path: Path to save the report
    """
    logger.info(f"Saving evaluation report to: {report_path}")
    
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("MODEL EVALUATION REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write("WEIGHTED METRICS:\n")
        f.write("-"*60 + "\n")
        for metric_name in ['accuracy', 'precision', 'recall', 'f1_score']:
            value = results['metrics'].get(metric_name)
            if value is not None:
                f.write(f"{metric_name.upper()}: {value:.4f}\n")
        
        f.write("\nPOSITIVE CLASS METRICS:\n")
        f.write("-"*60 + "\n")
        for metric_name in ['precision_positive', 'recall_positive', 'f1_positive']:
            value = results['metrics'].get(metric_name)
            if value is not None:
                f.write(f"{metric_name.upper()}: {value:.4f}\n")
        
        f.write("\nROC-AUC:\n")
        f.write("-"*60 + "\n")
        value = results['metrics'].get('roc_auc')
        if value is not None:
            f.write(f"ROC_AUC: {value:.4f}\n")
        else:
            f.write("ROC_AUC: N/A\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("PLOTS:\n")
        f.write("-"*60 + "\n")
        f.write(f"Confusion Matrix: {results['confusion_matrix_path']}\n")
        f.write(f"ROC Curve: {results['roc_curve_path']}\n")
    
    logger.info("Evaluation report saved successfully")
