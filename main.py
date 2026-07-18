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
    
    # Step 3: Train XGBoost model
    print("\n[Step 3/5] Training XGBoost model...")
    
    # Optional: Customize XGBoost parameters
    xgb_params = {
    'n_estimators': 500,
    'max_depth': 6,
    'learning_rate': 0.03,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'scale_pos_weight': scale_pos_weight,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'random_state': 42
    }
    
    model = train_xgboost_classifier(
        X_train=X_train,
        y_train=y_train,
        params=xgb_params,
        random_state=42
    )
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    print("\nPerforming 5-Fold Cross Validation...")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy"
    )

    print(f"Cross Validation Accuracy: {scores.mean():.2%}")
    print(f"Fold Scores: {scores}")
    
    # Train Random Forest
    print("\nTraining Random Forest model...")
    rf_model = train_random_forest_classifier(
        X_train=X_train,
        y_train=y_train,
        random_state=42
    )
    
    # Train SVM
    print("\nTraining SVM model...")
    svm_model = train_svm_classifier(
        X_train=X_train,
        y_train=y_train,
        random_state=42
    )

    # Step 4: Save trained models
    print("\n[Step 4/5] Saving trained models...")
    model_path = save_model(
        model=model,
        model_path=MODELS_DIR,
        model_name=MODEL_NAME
    )
    
    rf_model_path = save_model(
        model=rf_model,
        model_path=MODELS_DIR,
        model_name='random_forest_diabetic_nephropathy.joblib'
    )
    
    svm_model_path = save_model(
        model=svm_model,
        model_path=MODELS_DIR,
        model_name='svm_diabetic_nephropathy.joblib'
    )
    
    # Step 5: Evaluate models
    print("\n[Step 5/5] Evaluating models...")
    evaluation_results = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        plots_dir=PLOTS_DIR,
        model_name="XGBoost"
    )
    
    rf_evaluation_results = evaluate_model(
        model=rf_model,
        X_test=X_test,
        y_test=y_test,
        plots_dir=PLOTS_DIR,
        model_name="Random Forest"
    )
    
    svm_evaluation_results = evaluate_model(
        model=svm_model,
        X_test=X_test,
        y_test=y_test,
        plots_dir=PLOTS_DIR,
        model_name="SVM"
    )
    
    # Compare all three models
    models_metrics = {
        'XGBoost': {
            'accuracy': float(evaluation_results['metrics']['accuracy']),
            'precision': float(evaluation_results['metrics']['precision']),
            'recall': float(evaluation_results['metrics']['recall']),
            'f1': float(evaluation_results['metrics']['f1_score']),
            'roc_auc': float(evaluation_results['metrics']['roc_auc'] or 0.0),
            'model_name': 'XGBoost Classifier'
        },
        'Random Forest': {
            'accuracy': float(rf_evaluation_results['metrics']['accuracy']),
            'precision': float(rf_evaluation_results['metrics']['precision']),
            'recall': float(rf_evaluation_results['metrics']['recall']),
            'f1': float(rf_evaluation_results['metrics']['f1_score']),
            'roc_auc': float(rf_evaluation_results['metrics']['roc_auc'] or 0.0),
            'model_name': 'Random Forest Classifier'
        },
        'Support Vector Machine': {
            'accuracy': float(svm_evaluation_results['metrics']['accuracy']),
            'precision': float(svm_evaluation_results['metrics']['precision']),
            'recall': float(svm_evaluation_results['metrics']['recall']),
            'f1': float(svm_evaluation_results['metrics']['f1_score']),
            'roc_auc': float(svm_evaluation_results['metrics']['roc_auc'] or 0.0),
            'model_name': 'Support Vector Machine'
        }
    }
    
    # Sort models by Accuracy (descending), ROC-AUC (descending), and F1 Score (descending)
    sorted_models = sorted(
        models_metrics.keys(),
        key=lambda k: (models_metrics[k]['accuracy'], models_metrics[k]['roc_auc'], models_metrics[k]['f1']),
        reverse=True
    )
    best_model_key = sorted_models[0]
    best_model_metrics = models_metrics[best_model_key]
    
    # Retrieve the best model object
    best_model_obj = None
    if best_model_key == 'XGBoost':
        best_model_obj = model
    elif best_model_key == 'Random Forest':
        best_model_obj = rf_model
    else:
        best_model_obj = svm_model
        
    # Save the selected best model as the final prediction model.
    final_model_path = save_model(
        model=best_model_obj,
        model_path=MODELS_DIR,
        model_name='final_prediction_model.joblib'
    )

    # Create one professional comparison table for the console and report.
    model_order = ['XGBoost', 'Random Forest', 'Support Vector Machine']
    table_border = '+' + '-' * 26 + '+' + '-' * 10 + '+' + '-' * 11 + '+' + '-' * 8 + '+' + '-' * 10 + '+' + '-' * 10 + '+'
    table_lines = [
        '=' * 62,
        '      MODEL COMPARISON AFTER FEATURE SELECTION',
        '=' * 62,
        '',
        table_border,
        '| Model                    | Accuracy | Precision | Recall | F1 Score | ROC-AUC  |',
        table_border
    ]
    for name in model_order:
        metrics = models_metrics[name]
        table_lines.append(
            f"| {name:<24} | {metrics['accuracy']:<8.4f} | "
            f"{metrics['precision']:<9.4f} | {metrics['recall']:<6.4f} | "
            f"{metrics['f1']:<8.4f} | {metrics['roc_auc']:<8.4f} |"
        )
    table_lines.extend([
        table_border,
        '',
        '=' * 62,
        'BEST MODEL',
        '=' * 62,
        '',
        best_model_key
    ])
    comparison_output = '\n'.join(table_lines)

    # Save the comparison CSV with its established column format.
    comparison_path = Path(REPORTS_DIR) / 'model_comparison_feature_selection.csv'
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    with open(comparison_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=['Model', 'Accuracy', 'Precision', 'Recall', 'F1_score', 'Roc_auc']
        )
        writer.writeheader()
        for name in model_order:
            metrics = models_metrics[name]
            writer.writerow({
                'Model': name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1_score': f"{metrics['f1']:.4f}",
                'Roc_auc': f"{metrics['roc_auc']:.4f}"
            })

    comparison_report_path = Path(REPORTS_DIR) / 'model_comparison_feature_selection_report.txt'
    with open(comparison_report_path, 'w', encoding='utf-8') as report_file:
        report_file.write(comparison_output + '\n')

    print('\n' + comparison_output)
    
    # Save evaluation report
    report_path = Path(REPORTS_DIR) / 'evaluation_report.txt'
    save_evaluation_report(evaluation_results, str(report_path))
    
    # Step 6: SHAP Analysis (Explainable AI)
    print("\n[Step 6/6] Running SHAP analysis for model explainability...")
    shap_results = run_shap_analysis(
        model=model,
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
        'model_name': 'XGBoost Classifier',
        'dataset_name': Path(DATASET_PATH).name,
        'dataset_size': int(df.shape[0]),
        'num_features': int(X_train.shape[1]),
        'train_samples': int(X_train.shape[0]),
        'test_samples': int(X_test.shape[0]),
        'prediction_classes': int(y_train.nunique()),
        'model_status': 'Trained Successfully',
        'explainability': 'SHAP Enabled',
        'accuracy': float(evaluation_results['metrics']['accuracy']),
        'cv_accuracy': float(scores.mean()),
        'precision': float(evaluation_results['metrics']['precision']),
        'recall': float(evaluation_results['metrics']['recall']),
        'f1_score': float(evaluation_results['metrics']['f1_score']),
        'roc_auc': float(evaluation_results['metrics']['roc_auc']),
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
    print(f"\nModel saved to: {model_path}")
    print(f"Evaluation report saved to: {report_path}")
    print(f"SHAP analysis report saved to: {shap_report_path}")
    print(f"Pipeline metadata saved to: {metadata_path}")
    print(f"Plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
