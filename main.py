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
    
    # Step 4: Save trained model
    print("\n[Step 4/5] Saving trained model...")
    model_path = save_model(
        model=model,
        model_path=MODELS_DIR,
        model_name=MODEL_NAME
    )
    
    # Step 5: Evaluate model
    print("\n[Step 5/5] Evaluating model...")
    evaluation_results = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        plots_dir=PLOTS_DIR,
        model_name="XGBoost"
    )
    
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
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"\nModel saved to: {model_path}")
    print(f"Evaluation report saved to: {report_path}")
    print(f"SHAP analysis report saved to: {shap_report_path}")
    print(f"Plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
