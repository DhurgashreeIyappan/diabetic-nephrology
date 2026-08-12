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


def _get_expected_value(explainer, default_val: float = 0.5) -> float:
    """Safely retrieve expected (base) value from any SHAP explainer instance."""
    if hasattr(explainer, 'expected_value'):
        try:
            val = explainer.expected_value
            if isinstance(val, (list, np.ndarray)):
                val = np.asarray(val).ravel()[-1]
            return float(val)
        except Exception:
            pass
    if hasattr(explainer, 'expected_value_'):
        try:
            return float(explainer.expected_value_)
        except Exception:
            pass
    return float(default_val)


def initialize_shap_explainer(model, X_train):
    """
    Initialize SHAP explainer for the trained model.
    """
    tree_model_names = {'XGBClassifier', 'RandomForestClassifier', 'ExtraTreesClassifier', 'LGBMClassifier',
                        ########## NEW CATBOOST CODE ##########
                        'CatBoostClassifier'
                        ########## NEW CATBOOST CODE ##########
                        }
    if model.__class__.__name__ in tree_model_names:
        logger.info("Initializing SHAP TreeExplainer...")
        explainer = shap.TreeExplainer(model)
    else:
        logger.info("Initializing model-agnostic SHAP explainer...")
        background = shap.sample(X_train, min(100, len(X_train)), random_state=42)
        def positive_probability(values):
            frame = pd.DataFrame(values, columns=X_train.columns)
            return model.predict_proba(frame)[:, 1]
        explainer = shap.Explainer(positive_probability, background)
    
    logger.info("SHAP explainer initialized successfully")
    return explainer


def calculate_shap_values(explainer, X_test):
    """
    Calculate SHAP values for test data.
    """
    logger.info("Calculating SHAP values for test set...")
    
    if hasattr(explainer, 'shap_values'):
        shap_values = explainer.shap_values(X_test)
    else:
        explanation = explainer(X_test)
        shap_values = explanation.values
        try:
            explainer.expected_value_ = float(np.mean(explanation.base_values))
        except Exception:
            pass
    
    if isinstance(shap_values, list) and len(shap_values) > 1:
        logger.info("Multi-dimensional SHAP values detected. Using positive class values.")
        shap_values = shap_values[1]
    elif np.asarray(shap_values).ndim == 3:
        shap_values = np.asarray(shap_values)[:, :, 1]

    if hasattr(explainer, 'expected_value'):
        try:
            expected = np.asarray(explainer.expected_value)
            if expected.ndim and expected.size > 1:
                explainer.expected_value = float(expected.ravel()[-1])
        except Exception:
            pass
    
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
    """
    logger.info("Generating SHAP summary plot...")
    
    plt.figure(figsize=(12, 8))
    
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
    """
    logger.info("Generating global feature importance plot...")
    
    plt.figure(figsize=(12, 8))
    
    mean_shap = np.abs(shap_values).mean(axis=0)
    
    importance_df = pd.DataFrame({
        'feature': X_test.columns,
        'importance': mean_shap
    }).sort_values('importance', ascending=True)
    
    importance_df = importance_df.tail(max_display)
    
    plt.barh(importance_df['feature'], importance_df['importance'], 
             color='steelblue', alpha=0.7, edgecolor='black')
    plt.xlabel('Mean |SHAP Value| (Average Impact on Model Output)', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title('Global Feature Importance (SHAP Values)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
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
    """
    logger.info(f"Generating waterfall plot for sample {sample_idx}...")
    
    plt.figure(figsize=(12, 8))
    
    base_val = _get_expected_value(explainer)
    
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values[sample_idx],
            base_values=base_val,
            data=X_test.iloc[sample_idx],
            feature_names=X_test.columns.tolist()
        ),
        max_display=max_display,
        show=False
    )
    
    plt.title(f"Waterfall Plot - Sample {sample_idx} Prediction Explanation", 
              fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
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
    """
    logger.info(f"Generating force plot for sample {sample_idx}...")
    
    base_val = _get_expected_value(explainer)
    
    force_plot = shap.force_plot(
        base_val,
        shap_values[sample_idx],
        X_test.iloc[sample_idx],
        feature_names=X_test.columns.tolist(),
        matplotlib=True,
        show=False
    )
    
    plt.title(f"Force Plot - Sample {sample_idx} Prediction Explanation", 
              fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
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
    """
    logger.info(f"Generating dependence plot for feature: {feature_name}")
    
    plt.figure(figsize=(12, 8))
    
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
    """
    logger.info(f"Generating comprehensive explanation for patient {sample_idx}...")
    
    results = {}
    
    patient_data = X_test.iloc[sample_idx]
    patient_shap = shap_values[sample_idx]
    base_value = _get_expected_value(explainer)
    prediction = base_value + patient_shap.sum()
    
    waterfall_path = Path(save_dir) / f"patient_{sample_idx}_waterfall.png"
    plot_waterfall(explainer, shap_values, X_test, sample_idx, str(waterfall_path))
    results['waterfall_plot'] = str(waterfall_path)
    
    force_path = Path(save_dir) / f"patient_{sample_idx}_force.png"
    plot_force(explainer, shap_values, X_test, sample_idx, str(force_path))
    results['force_plot'] = str(force_path)
    
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
    """
    logger.info("="*60)
    logger.info("STARTING SHAP ANALYSIS")
    logger.info("="*60)
    
    results = {}
    
    explainer = initialize_shap_explainer(model, X_train)
    results['explainer_type'] = explainer.__class__.__name__
    
    shap_values = calculate_shap_values(explainer, X_test)
    results['shap_values_shape'] = np.array(shap_values).shape
    
    summary_path = Path(plots_dir) / 'shap_summary_plot.png'
    plot_shap_summary(shap_values, X_test, str(summary_path), 
                      plot_type='dot', max_display=max_display)
    results['summary_plot'] = str(summary_path)
    
    importance_path = Path(plots_dir) / 'shap_global_feature_importance.png'
    plot_global_feature_importance(shap_values, X_test, str(importance_path), 
                                   max_display=max_display)
    results['global_importance_plot'] = str(importance_path)
    
    waterfall_path = Path(plots_dir) / 'shap_waterfall_plot.png'
    plot_waterfall(explainer, shap_values, X_test, patient_idx, 
                  str(waterfall_path), max_display=max_display)
    results['waterfall_plot'] = str(waterfall_path)
    
    force_path = Path(plots_dir) / 'shap_force_plot.png'
    plot_force(explainer, shap_values, X_test, patient_idx, str(force_path))
    results['force_plot'] = str(force_path)
    
    mean_shap = np.abs(shap_values).mean(axis=0)
    top_features_idx = np.argsort(mean_shap)[-3:][::-1]
    top_features = X_test.columns[top_features_idx].tolist()
    
    results['dependence_plots'] = {}
    for idx, feature in enumerate(top_features):
        dep_path = Path(plots_dir) / f'shap_dependence_{feature}.png'
        plot_dependence(shap_values, X_test, feature, str(dep_path))
        results['dependence_plots'][feature] = str(dep_path)
    
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
