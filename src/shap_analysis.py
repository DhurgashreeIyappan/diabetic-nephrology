"""
SHAP Analysis Module

This module provides functions for explainable AI using SHAP (SHapley Additive exPlanations).
SHAP values help interpret machine learning model predictions by showing feature contributions.
"""

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def initialize_shap_explainer(model, X_train):
    """
    Initialize SHAP explainer for the trained model.
    
    For XGBoost, we use TreeExplainer which is optimized for tree-based models.
    
    Args:
        model: Trained XGBoost model
        X_train: Training features (used as background data)
    
    Returns:
        SHAP explainer object
    """
    logger.info("Initializing SHAP TreeExplainer...")
    
    # Use TreeExplainer for XGBoost (faster and more accurate for tree models)
    explainer = shap.TreeExplainer(model)
    
    logger.info("SHAP explainer initialized successfully")
    return explainer


def calculate_shap_values(explainer, X_test):
    """
    Calculate SHAP values for test data.
    
    SHAP values represent the contribution of each feature to the prediction
    for each individual sample.
    
    Args:
        explainer: SHAP explainer object
        X_test: Test features
    
    Returns:
        SHAP values array (for binary classification, returns positive class values)
    """
    logger.info("Calculating SHAP values for test set...")
    
    shap_values = explainer.shap_values(X_test)
    
    # Handle multi-dimensional SHAP values for binary classification
    if isinstance(shap_values, list) and len(shap_values) > 1:
        logger.info("Multi-dimensional SHAP values detected. Using positive class values.")
        shap_values = shap_values[1]  # Use positive class SHAP values
    
    logger.info(f"SHAP values calculated. Shape: {np.array(shap_values).shape}")
    return shap_values


def plot_shap_summary(
    shap_values,
    X_test,
    save_path: str,
    plot_type: str = 'dot',
    max_display: int = 20
) -> None:
    """
    Generate and save SHAP summary plot.
    
    This plot shows the most important features and their impact on the model output.
    
    - **X-axis**: SHAP value (positive = increases prediction, negative = decreases)
    - **Y-axis**: Features ordered by importance
    - **Color**: Feature value (red = high, blue = low)
    
    Interpretation:
    - Features higher on the plot are more important
    - Points to the right increase the prediction
    - Points to the left decrease the prediction
    - Red points indicate high feature values, blue points indicate low values
    
    Args:
        shap_values: Calculated SHAP values
        X_test: Test features DataFrame
        save_path: Path to save the plot
        plot_type: Type of plot ('dot', 'bar', or 'violin')
        max_display: Maximum number of features to display
    """
    logger.info("Generating SHAP summary plot...")
    
    plt.figure(figsize=(12, 8))
    
    # Create summary plot
    shap.summary_plot(
        shap_values, 
        X_test, 
        plot_type=plot_type,
        max_display=max_display,
        show=False
    )
    
    plt.title("SHAP Summary Plot - Feature Impact on Predictions", 
              fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    # Save plot
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"SHAP summary plot saved to: {save_path}")


def plot_global_feature_importance(
    shap_values,
    X_test,
    save_path: str,
    max_display: int = 20
) -> None:
    """
    Generate and save global feature importance plot using SHAP values.
    
    This plot shows the mean absolute SHAP value for each feature,
    representing the overall importance of each feature in the model.
    
    - **X-axis**: Mean |SHAP value| (importance)
    - **Y-axis**: Features ordered by importance
    
    Interpretation:
    - Higher bars indicate more important features
    - Features with higher mean |SHAP| have greater impact on predictions
    - This is a model-agnostic way to understand feature importance
    
    Args:
        shap_values: Calculated SHAP values
        X_test: Test features DataFrame
        save_path: Path to save the plot
        max_display: Maximum number of features to display
    """
    logger.info("Generating global feature importance plot...")
    
    plt.figure(figsize=(12, 8))
    
    # Calculate mean absolute SHAP values
    mean_shap = np.abs(shap_values).mean(axis=0)
    
    # Create DataFrame for plotting
    importance_df = pd.DataFrame({
        'feature': X_test.columns,
        'importance': mean_shap
    }).sort_values('importance', ascending=True)
    
    # Select top features
    importance_df = importance_df.tail(max_display)
    
    # Create bar plot
    plt.barh(importance_df['feature'], importance_df['importance'], 
             color='steelblue', alpha=0.7, edgecolor='black')
    plt.xlabel('Mean |SHAP Value| (Average Impact on Model Output)', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title('Global Feature Importance (SHAP Values)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Global feature importance plot saved to: {save_path}")


def plot_waterfall(
    explainer,
    shap_values,
    X_test,
    sample_idx: int,
    save_path: str,
    max_display: int = 20
) -> None:
    """
    Generate and save waterfall plot for a single prediction.
    
    The waterfall plot shows how each feature contributes to pushing the
    model output from the base value (average prediction) to the final prediction.
    
    - **Base value**: Average model prediction across training data
    - **Red bars**: Features that increase the prediction
    - **Blue bars**: Features that decrease the prediction
    - **Final value**: Actual prediction for this sample
    
    Interpretation:
    - Start from the base value (average prediction)
    - Each feature adds or subtracts from this base
    - The final value is the sum of all contributions
    - Helps understand why the model made a specific prediction
    
    Args:
        explainer: SHAP explainer object
        shap_values: Calculated SHAP values
        X_test: Test features DataFrame
        sample_idx: Index of the sample to explain
        save_path: Path to save the plot
        max_display: Maximum number of features to display
    """
    logger.info(f"Generating waterfall plot for sample {sample_idx}...")
    
    plt.figure(figsize=(12, 8))
    
    # Create waterfall plot
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values[sample_idx],
            base_values=explainer.expected_value,
            data=X_test.iloc[sample_idx],
            feature_names=X_test.columns.tolist()
        ),
        max_display=max_display,
        show=False
    )
    
    plt.title(f"Waterfall Plot - Sample {sample_idx} Prediction Explanation", 
              fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    # Save plot
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Waterfall plot saved to: {save_path}")


def plot_force(
    explainer,
    shap_values,
    X_test,
    sample_idx: int,
    save_path: str
) -> None:
    """
    Generate and save force plot for a single prediction.
    
    The force plot provides an interactive visualization of how each feature
    contributes to the prediction for a specific sample.
    
    - **Base value**: Average prediction (starting point)
    - **Red arrows**: Features pushing prediction higher
    - **Blue arrows**: Features pushing prediction lower
    - **Output**: Final prediction value
    
    Interpretation:
    - Visual representation of feature contributions
    - Length of arrow indicates magnitude of contribution
    - Direction indicates whether it increases or decreases prediction
    - Excellent for explaining individual predictions to clinicians
    
    Args:
        explainer: SHAP explainer object
        shap_values: Calculated SHAP values
        X_test: Test features DataFrame
        sample_idx: Index of the sample to explain
        save_path: Path to save the plot
    """
    logger.info(f"Generating force plot for sample {sample_idx}...")
    
    # Create force plot
    force_plot = shap.force_plot(
        explainer.expected_value,
        shap_values[sample_idx],
        X_test.iloc[sample_idx],
        feature_names=X_test.columns.tolist(),
        matplotlib=True,
        show=False
    )
    
    plt.title(f"Force Plot - Sample {sample_idx} Prediction Explanation", 
              fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    # Save plot
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Force plot saved to: {save_path}")


def plot_dependence(
    shap_values,
    X_test,
    feature_name: str,
    save_path: str,
    interaction_feature: Optional[str] = None
) -> None:
    """
    Generate and save dependence plot for a specific feature.
    
    The dependence plot shows how a feature's value affects its SHAP value
    (impact on prediction). It can also show interaction effects with another feature.
    
    - **X-axis**: Feature value
    - **Y-axis**: SHAP value (impact on prediction)
    - **Color**: Value of interaction feature (if specified)
    
    Interpretation:
    - Points to the right: Feature increases prediction
    - Points to the left: Feature decreases prediction
    - Vertical spread: Indicates interaction with other features
    - Color coding: Shows how interaction feature affects the relationship
    
    Args:
        shap_values: Calculated SHAP values
        X_test: Test features DataFrame
        feature_name: Name of the feature to plot
        save_path: Path to save the plot
        interaction_feature: Optional feature for interaction coloring
    """
    logger.info(f"Generating dependence plot for feature: {feature_name}")
    
    plt.figure(figsize=(12, 8))
    
    # Create dependence plot
    shap.dependence_plot(
        feature_name,
        shap_values,
        X_test,
        interaction_index=interaction_feature if interaction_feature else 'auto',
        show=False
    )
    
    plt.title(f"Dependence Plot - {feature_name}", 
              fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    # Save plot
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Dependence plot saved to: {save_path}")


def explain_individual_prediction(
    explainer,
    shap_values,
    X_test,
    sample_idx: int,
    feature_names: list,
    save_dir: str
) -> Dict[str, Any]:
    """
    Comprehensive explanation for a single patient's prediction.
    
    This function generates multiple visualizations to explain why the model
    made a specific prediction for an individual patient.
    
    Includes:
    1. Waterfall plot - Step-by-step contribution breakdown
    2. Force plot - Visual representation of feature contributions
    3. Feature contribution table - Numerical breakdown
    
    Args:
        explainer: SHAP explainer object
        shap_values: Calculated SHAP values
        X_test: Test features DataFrame
        sample_idx: Index of the patient to explain
        feature_names: List of feature names
        save_dir: Directory to save explanations
    
    Returns:
        Dictionary containing explanation results
    """
    logger.info(f"Generating comprehensive explanation for patient {sample_idx}...")
    
    results = {}
    
    # Get patient data
    patient_data = X_test.iloc[sample_idx]
    patient_shap = shap_values[sample_idx]
    base_value = explainer.expected_value
    prediction = base_value + patient_shap.sum()
    
    # Generate waterfall plot
    waterfall_path = Path(save_dir) / f"patient_{sample_idx}_waterfall.png"
    plot_waterfall(explainer, shap_values, X_test, sample_idx, str(waterfall_path))
    results['waterfall_plot'] = str(waterfall_path)
    
    # Generate force plot
    force_path = Path(save_dir) / f"patient_{sample_idx}_force.png"
    plot_force(explainer, shap_values, X_test, sample_idx, str(force_path))
    results['force_plot'] = str(force_path)
    
    # Create feature contribution table
    contribution_df = pd.DataFrame({
        'Feature': feature_names,
        'Feature Value': patient_data.values,
        'SHAP Value': patient_shap,
        'Contribution Direction': ['Increases' if v > 0 else 'Decreases' for v in patient_shap]
    }).sort_values(by='SHAP Value', key=abs, ascending=False)
    
    results['prediction'] = prediction
    results['base_value'] = base_value
    results['feature_contributions'] = contribution_df
    
    logger.info(f"Patient explanation completed for sample {sample_idx}")
    return results


def run_shap_analysis(
    model,
    X_train,
    X_test,
    plots_dir: str,
    max_display: int = 20,
    patient_idx: int = 0
) -> Dict[str, Any]:
    """
    Complete SHAP analysis pipeline.
    
    This function runs the full SHAP analysis including:
    1. Initialize explainer
    2. Calculate SHAP values
    3. Generate summary plot
    4. Generate global feature importance
    5. Generate waterfall plot for a sample
    6. Generate force plot for a sample
    7. Generate dependence plots for top features
    8. Explain individual patient prediction
    
    Args:
        model: Trained XGBoost model
        X_train: Training features
        X_test: Test features
        plots_dir: Directory to save all plots
        max_display: Maximum number of features to display
        patient_idx: Index of patient for individual explanation
    
    Returns:
        Dictionary containing all SHAP analysis results
    """
    logger.info("="*60)
    logger.info("STARTING SHAP ANALYSIS")
    logger.info("="*60)
    
    results = {}
    
    # Step 1: Initialize explainer
    explainer = initialize_shap_explainer(model, X_train)
    results['explainer_type'] = 'TreeExplainer'
    
    # Step 2: Calculate SHAP values
    shap_values = calculate_shap_values(explainer, X_test)
    results['shap_values_shape'] = np.array(shap_values).shape
    
    # Step 3: Generate summary plot
    summary_path = Path(plots_dir) / 'shap_summary_plot.png'
    plot_shap_summary(shap_values, X_test, str(summary_path), 
                      plot_type='dot', max_display=max_display)
    results['summary_plot'] = str(summary_path)
    
    # Step 4: Generate global feature importance
    importance_path = Path(plots_dir) / 'shap_global_feature_importance.png'
    plot_global_feature_importance(shap_values, X_test, str(importance_path), 
                                   max_display=max_display)
    results['global_importance_plot'] = str(importance_path)
    
    # Step 5: Generate waterfall plot for a sample
    waterfall_path = Path(plots_dir) / 'shap_waterfall_plot.png'
    plot_waterfall(explainer, shap_values, X_test, patient_idx, 
                  str(waterfall_path), max_display=max_display)
    results['waterfall_plot'] = str(waterfall_path)
    
    # Step 6: Generate force plot for a sample
    force_path = Path(plots_dir) / 'shap_force_plot.png'
    plot_force(explainer, shap_values, X_test, patient_idx, str(force_path))
    results['force_plot'] = str(force_path)
    
    # Step 7: Generate dependence plots for top 3 features
    # Get top features by importance
    mean_shap = np.abs(shap_values).mean(axis=0)
    top_features_idx = np.argsort(mean_shap)[-3:][::-1]
    top_features = X_test.columns[top_features_idx].tolist()
    
    results['dependence_plots'] = {}
    for idx, feature in enumerate(top_features):
        dep_path = Path(plots_dir) / f'shap_dependence_{feature}.png'
        plot_dependence(shap_values, X_test, feature, str(dep_path))
        results['dependence_plots'][feature] = str(dep_path)
    
    # Step 8: Explain individual patient prediction
    patient_explanation = explain_individual_prediction(
        explainer, shap_values, X_test, patient_idx,
        X_test.columns.tolist(), plots_dir
    )
    results['patient_explanation'] = patient_explanation
    
    logger.info("="*60)
    logger.info("SHAP ANALYSIS COMPLETED")
    logger.info("="*60)
    
    return results


def save_shap_report(results: Dict[str, Any], report_path: str) -> None:
    """
    Save SHAP analysis results to a text file.
    
    Args:
        results: Dictionary containing SHAP analysis results
        report_path: Path to save the report
    """
    logger.info(f"Saving SHAP analysis report to: {report_path}")
    
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("SHAP ANALYSIS REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write("EXPLAINER TYPE:\n")
        f.write("-"*60 + "\n")
        f.write(f"{results.get('explainer_type', 'N/A')}\n\n")
        
        f.write("SHAP VALUES SHAPE:\n")
        f.write("-"*60 + "\n")
        f.write(f"{results.get('shap_values_shape', 'N/A')}\n\n")
        
        f.write("GENERATED PLOTS:\n")
        f.write("-"*60 + "\n")
        f.write(f"Summary Plot: {results.get('summary_plot', 'N/A')}\n")
        f.write(f"Global Feature Importance: {results.get('global_importance_plot', 'N/A')}\n")
        f.write(f"Waterfall Plot: {results.get('waterfall_plot', 'N/A')}\n")
        f.write(f"Force Plot: {results.get('force_plot', 'N/A')}\n\n")
        
        f.write("DEPENDENCE PLOTS:\n")
        f.write("-"*60 + "\n")
        for feature, path in results.get('dependence_plots', {}).items():
            f.write(f"{feature}: {path}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("PATIENT EXPLANATION\n")
        f.write("="*60 + "\n\n")
        
        patient_exp = results.get('patient_explanation', {})
        f.write(f"Prediction: {patient_exp.get('prediction', 'N/A'):.4f}\n")
        f.write(f"Base Value: {patient_exp.get('base_value', 'N/A'):.4f}\n\n")
        
        f.write("TOP FEATURE CONTRIBUTIONS:\n")
        f.write("-"*60 + "\n")
        if 'feature_contributions' in patient_exp:
            contrib_df = patient_exp['feature_contributions'].head(10)
            for _, row in contrib_df.iterrows():
                f.write(f"{row['Feature']}: {row['SHAP Value']:.4f} ({row['Contribution Direction']})\n")
                f.write(f"  Feature Value: {row['Feature Value']:.4f}\n")
    
    logger.info("SHAP analysis report saved successfully")
