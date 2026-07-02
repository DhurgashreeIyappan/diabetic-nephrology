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
    TARGET_COLUMN = 'diabetic_nephropathy'  # Update with your target column
    MODELS_DIR = 'models'
    PLOTS_DIR = 'outputs/plots'
    REPORTS_DIR = 'outputs/reports'
    MODEL_NAME = 'xgboost_diabetic_nephropathy.joblib'
    
    # Step 1: Load dataset
    print("\n[Step 1/5] Loading dataset...")
    df = load_dataset(DATASET_PATH)
    print(f"Dataset loaded: {df.shape}")
    
    # Step 2: Preprocess data
    print("\n[Step 2/5] Preprocessing data...")
    preprocessing_artifacts = preprocess_pipeline(
        df=df,
        target_column=TARGET_COLUMN,
        missing_strategy='mean',
        encode_categorical=True,
        encoding_method='label',
        handle_imbalance=None,  # Set to 'oversample' or 'undersample' if needed
        apply_scaling=True,
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
    
    # Step 3: Train XGBoost model
    print("\n[Step 3/5] Training XGBoost model...")
    
    # Optional: Customize XGBoost parameters
    xgb_params = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }
    
    model = train_xgboost_classifier(
        X_train=X_train,
        y_train=y_train,
        params=xgb_params,
        random_state=42
    )
    
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
