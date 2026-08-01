"""
Main Pipeline Script

This script runs the complete machine learning pipeline for diabetic nephropathy prediction:
1. Data loading
2. Preprocessing
3. Model training (XGBoost)
4. Model evaluation
5. Saving artifacts
"""

import sys
import csv
from pathlib import Path
import logging
import warnings

# Reconfigure stdout/stderr to replace encoding errors on Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(errors='replace')

# Configure logging to write to stdout to avoid scrambled output between print and logging statements
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(levelname)s:%(name)s:%(message)s')

# Suppress warnings from matplotlib regarding unicode glyphs missing from fonts
warnings.filterwarnings("ignore", message=".*Glyph.*missing from font.*")

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src import (
    load_dataset,
    preprocess_pipeline,
    train_xgboost_classifier,
    train_random_forest_classifier,
    train_svm_classifier,
    train_lightgbm_classifier,
    train_extra_trees_classifier,
    tune_classifier,
    build_stacking_classifier,
    save_model,
    load_model,
    evaluate_model,
    save_evaluation_report,
    run_shap_analysis,
    save_shap_report
)


def main():
    """
    Main pipeline execution function.
    """
    print("="*60)
    print("DIABETIC NEPHROPATHY PREDICTION PIPELINE")
    print("="*60)
    
    # Configuration
    DATASET_PATH = 'dataset/Diabetic_Nephropathy_v1.xlsx'
    TARGET_COLUMN = 'Diabetic nephropathy (DN)'  # Target column from dataset
    MODELS_DIR = 'models'
    PLOTS_DIR = 'outputs/plots'
    REPORTS_DIR = 'outputs/reports'
    MODEL_NAME = 'xgboost_diabetic_nephropathy.joblib'
    
    # Step 1: Load dataset
    print("\n[Step 1/5] Loading dataset...")
    df = load_dataset(DATASET_PATH)
    print(f"Dataset loaded: {df.shape}")
    
    # Print target column information
    print(f"\nTarget Column: {TARGET_COLUMN}")
    print(f"Unique Values (Before Encoding): {df[TARGET_COLUMN].unique()}")
    print(f"Value Counts:\n{df[TARGET_COLUMN].value_counts()}")
    
    # Step 2: Preprocess data
    print("\n[Step 2/5] Preprocessing data...")
    preprocessing_artifacts = preprocess_pipeline(
        df=df,
        target_column=TARGET_COLUMN,
        missing_strategy='mean',
        encode_categorical=True,
        encoding_method='label',
        handle_imbalance="oversample",  # Set to 'oversample' or 'undersample' if needed
        apply_scaling=False,  # XGBoost doesn't need scaling
        scaling_method='standard',
        test_size=0.2,
        random_state=42
    )
    
    # Extract processed data
    X_train = preprocessing_artifacts['X_train']
    X_test = preprocessing_artifacts['X_test']
    y_train = preprocessing_artifacts['y_train']
    y_test = preprocessing_artifacts['y_test']
    feature_names = preprocessing_artifacts['feature_names']
    
    print(f"Training set: {X_train.shape}")
    print(f"Testing set: {X_test.shape}")
    print(f"Unique Values (After Encoding): {y_train.unique()}")
    
    # Calculate class imbalance for scale_pos_weight
    class_counts = y_train.value_counts()
    scale_pos_weight = class_counts[0] / class_counts[1] if len(class_counts) == 2 else 1
    print(f"Class distribution: {class_counts.to_dict()}")
    print(f"Scale pos weight: {scale_pos_weight:.2f}")
    
    # Step 3: tune every base model on the same already-preprocessed training set.
    print("\n[Step 3/6] Tuning models with stratified 5-fold ROC-AUC...")
    from xgboost import XGBClassifier
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    from sklearn.svm import SVC
    from lightgbm import LGBMClassifier

    tuned_models = {}
    tuned_models['XGBoost'], xgb_search = tune_classifier(
        XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42, n_jobs=1,
                      scale_pos_weight=scale_pos_weight),
        {'n_estimators': [200, 300, 500], 'max_depth': [3, 4, 6], 'learning_rate': [0.01, 0.03, 0.05],
         'subsample': [0.7, 0.8, 0.9], 'colsample_bytree': [0.7, 0.8, 0.9], 'min_child_weight': [1, 3, 5]},
        X_train, y_train)
    tuned_models['Random Forest'], rf_search = tune_classifier(
        RandomForestClassifier(random_state=42, n_jobs=1),
        {'n_estimators': [200, 300, 500], 'max_depth': [None, 5, 10, 20],
         'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4], 'max_features': ['sqrt', 'log2', None]},
        X_train, y_train)
    tuned_models['Support Vector Machine'], svm_search = tune_classifier(
        SVC(probability=True, random_state=42),
        {'C': [0.1, 1, 10, 100], 'kernel': ['rbf', 'linear'], 'gamma': ['scale', 'auto', 0.01, 0.1]},
        X_train, y_train)
    tuned_models['LightGBM'], lgbm_search = tune_classifier(
        LGBMClassifier(objective='binary', random_state=42, verbosity=-1, n_jobs=1),
        {'n_estimators': [200, 300, 500], 'learning_rate': [0.01, 0.03, 0.05], 'num_leaves': [15, 31, 63],
         'max_depth': [-1, 4, 6, 10], 'subsample': [0.7, 0.8, 0.9], 'colsample_bytree': [0.7, 0.8, 0.9]},
        X_train, y_train)
    tuned_models['Extra Trees'], et_search = tune_classifier(
        ExtraTreesClassifier(random_state=42, n_jobs=1),
        {'n_estimators': [200, 300, 500], 'max_depth': [None, 5, 10, 20],
         'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4], 'max_features': ['sqrt', 'log2', None]},
        X_train, y_train)

    # The final proposed model uses all tuned base estimators and Logistic Regression.
    stacking_model = build_stacking_classifier(
        tuned_models['XGBoost'], tuned_models['Random Forest'], tuned_models['Support Vector Machine'],
        tuned_models['LightGBM'], tuned_models['Extra Trees'])
    stacking_model.fit(X_train, y_train)
    tuned_models['Stacking Classifier'] = stacking_model

    # Step 4: reuse the established evaluator for every candidate.
    print("\n[Step 4/6] Evaluating all models...")
    evaluation_by_model = {}
    for name, candidate in tuned_models.items():
        evaluation_by_model[name] = evaluate_model(candidate, X_test, y_test, PLOTS_DIR, name)

    model_display_names = {
        'XGBoost': 'XGBoost Classifier', 'Random Forest': 'Random Forest Classifier',
        'Support Vector Machine': 'Support Vector Machine', 'LightGBM': 'LightGBM Classifier',
        'Extra Trees': 'Extra Trees Classifier', 'Stacking Classifier': 'Stacking Classifier'
    }
    models_metrics = {
        name: {
            'accuracy': float(result['metrics']['accuracy']), 'precision': float(result['metrics']['precision']),
            'recall': float(result['metrics']['recall']), 'f1': float(result['metrics']['f1_score']),
            'roc_auc': float(result['metrics']['roc_auc'] or 0.0), 'model_name': model_display_names[name]
        } for name, result in evaluation_by_model.items()
    }

    # ROC-AUC alone determines the final deployed model.
    best_model_key = max(models_metrics, key=lambda name: models_metrics[name]['roc_auc'])
    best_model_metrics = models_metrics[best_model_key]
    best_model_obj = tuned_models[best_model_key]
    final_model_path = save_model(best_model_obj, MODELS_DIR, 'final_prediction_model.joblib')

    # Professional console table and the required publication-ready reports.
    model_order = ['XGBoost', 'Random Forest', 'Support Vector Machine', 'LightGBM', 'Extra Trees', 'Stacking Classifier']
    table_border = '-' * 63
    table_lines = [
        table_border,
        f"{'Model':<25} {'Accuracy':<10} {'Precision':<11} {'Recall':<8} {'F1':<5}   {'ROC-AUC'}",
        table_border
    ]
    for name in model_order:
        metrics = models_metrics[name]
        table_lines.append(
            f"{name:<25} {metrics['accuracy']:<10.4f} {metrics['precision']:<11.4f} {metrics['recall']:<8.4f} {metrics['f1']:<5.4f}   {metrics['roc_auc']:.4f}"
        )
    table_lines.extend([
        table_border,
        "",
        "BEST MODEL",
        "",
        best_model_key
    ])
    comparison_output = '\n'.join(table_lines)

    comparison_path = Path(REPORTS_DIR) / 'model_comparison.csv'
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    with open(comparison_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC']
        )
        writer.writeheader()
        for name in model_order:
            metrics = models_metrics[name]
            writer.writerow({
                'Model': name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-score': f"{metrics['f1']:.4f}",
                'ROC-AUC': f"{metrics['roc_auc']:.4f}"
            })

    comparison_report_path = Path(REPORTS_DIR) / 'model_comparison.txt'
    with open(comparison_report_path, 'w', encoding='utf-8') as report_file:
        report_file.write(comparison_output + '\n')

    print('\n' + comparison_output)
    
    # Preserve the established single-model evaluation report for the selected model.
    report_path = Path(REPORTS_DIR) / 'evaluation_report.txt'
    save_evaluation_report(evaluation_by_model[best_model_key], str(report_path))
    
    # Step 6: SHAP Analysis (Explainable AI)
    print("\n[Step 6/6] Running SHAP analysis for model explainability...")
    shap_results = run_shap_analysis(
        model=best_model_obj,
        X_train=X_train,
        X_test=X_test,
        plots_dir=PLOTS_DIR,
        max_display=20,
        patient_idx=0  # Explain first patient in test set
    )
    
    # Save SHAP report
    shap_report_path = Path(REPORTS_DIR) / 'shap_analysis_report.txt'
    save_shap_report(shap_results, str(shap_report_path))
    
    # Save a JSON file with all metrics and metadata for Streamlit to consume dynamically
    metadata_path = Path(REPORTS_DIR) / 'pipeline_metadata.json'
    print(f"\nSaving pipeline metadata and metrics to {metadata_path}...")
    import json
    metadata = {
        'model_name': best_model_metrics['model_name'],
        'dataset_name': Path(DATASET_PATH).name,
        'dataset_size': int(df.shape[0]),
        'num_features': int(X_train.shape[1]),
        'train_samples': int(X_train.shape[0]),
        'test_samples': int(X_test.shape[0]),
        'prediction_classes': int(y_train.nunique()),
        'model_status': 'Trained Successfully',
        'explainability': 'SHAP Enabled',
        'accuracy': float(best_model_metrics['accuracy']),
        'cv_accuracy': float(max(search.best_score_ for search in [xgb_search, rf_search, svm_search, lgbm_search, et_search])),
        'precision': float(best_model_metrics['precision']),
        'recall': float(best_model_metrics['recall']),
        'f1_score': float(best_model_metrics['f1']),
        'roc_auc': float(best_model_metrics['roc_auc']),
        'comparison': models_metrics,
        'best_model': {
            'name': best_model_metrics['model_name'],
            'key': best_model_key,
            'accuracy': float(best_model_metrics['accuracy']),
            'roc_auc': float(best_model_metrics['roc_auc']),
            'f1_score': float(best_model_metrics['f1'])
        }
    }
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"\nFinal model saved to: {final_model_path}")
    print(f"Evaluation report saved to: {report_path}")
    print(f"SHAP analysis report saved to: {shap_report_path}")
    print(f"Pipeline metadata saved to: {metadata_path}")
    print(f"Plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
