"""
Diabetic Nephropathy Prediction Package
"""

from .data_loader import load_dataset, display_dataset_info
from .preprocessing import (
    preprocess_pipeline,
    display_data_quality,
    handle_missing_values,
    remove_duplicates,
    encode_categorical_columns,
    handle_class_imbalance,
    split_features_target,
    train_test_split_data,
    apply_feature_scaling
)
from .model import (
    train_xgboost_classifier,
    train_random_forest_classifier,
    train_svm_classifier,
    save_model,
    load_model,
    get_model_feature_importance
)
from .evaluation import (
    calculate_metrics,
    print_classification_report,
    print_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    evaluate_model,
    save_evaluation_report
)
from .shap_analysis import (
    initialize_shap_explainer,
    calculate_shap_values,
    plot_shap_summary,
    plot_global_feature_importance,
    plot_waterfall,
    plot_force,
    plot_dependence,
    explain_individual_prediction,
    run_shap_analysis,
    save_shap_report
)

__all__ = [
    'load_dataset',
    'display_dataset_info',
    'preprocess_pipeline',
    'display_data_quality',
    'handle_missing_values',
    'remove_duplicates',
    'encode_categorical_columns',
    'handle_class_imbalance',
    'split_features_target',
    'train_test_split_data',
    'apply_feature_scaling',
    'train_xgboost_classifier',
    'train_random_forest_classifier',
    'train_svm_classifier',
    'save_model',
    'load_model',
    'get_model_feature_importance',
    'calculate_metrics',
    'print_classification_report',
    'print_metrics',
    'plot_confusion_matrix',
    'plot_roc_curve',
    'evaluate_model',
    'save_evaluation_report',
    'initialize_shap_explainer',
    'calculate_shap_values',
    'plot_shap_summary',
    'plot_global_feature_importance',
    'plot_waterfall',
    'plot_force',
    'plot_dependence',
    'explain_individual_prediction',
    'run_shap_analysis',
    'save_shap_report'
]
