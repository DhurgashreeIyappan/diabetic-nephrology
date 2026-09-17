"""
Evaluation Graphs Generator

This script generates ROC curves, accuracy comparison bar chart, and confusion matrices
with 'Non-DN' and 'DN' class labels for all six trained models:
1. XGBoost
2. Random Forest
3. LightGBM
4. Extra Trees
5. CatBoost
6. Stacking Classifier

Graphs are displayed via Matplotlib and saved to outputs/plots/evaluation_graphs/
without modifying any existing source code or Streamlit dashboard.
"""

import sys
import os
from pathlib import Path

# Reconfigure stdout/stderr for Windows console unicode support
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Add src to sys.path
project_root = Path(__file__).parent
sys.path.append(str(project_root / 'src'))

from src.data_loader import load_dataset
from src.preprocessing import preprocess_pipeline
from src.model import tune_classifier, build_stacking_classifier

def main():
    print("=" * 60)
    print("GENERATING MODEL EVALUATION GRAPHS (VS CODE WORKFLOW)")
    print("=" * 60)

    # 1. Configuration
    DATASET_PATH = Path('dataset/Diabetic_Nephropathy_v1.xlsx').resolve()
    TARGET_COLUMN = 'Diabetic nephropathy (DN)'
    OUTPUT_DIR = Path('outputs/plots/evaluation_graphs').resolve()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Set overall matplotlib style
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    sns.set_theme(style="whitegrid")

    # 2. Load & Preprocess Data
    print("\n[1/4] Loading and preprocessing dataset...")
    df = load_dataset(str(DATASET_PATH))
    
    preprocessing_artifacts = preprocess_pipeline(
        df=df,
        target_column=TARGET_COLUMN,
        missing_strategy='mean',
        encode_categorical=True,
        encoding_method='label',
        handle_imbalance="oversample",
        apply_scaling=False,
        test_size=0.2,
        random_state=42
    )

    X_train = preprocessing_artifacts['X_train']
    X_test = preprocessing_artifacts['X_test']
    y_train = preprocessing_artifacts['y_train']
    y_test = preprocessing_artifacts['y_test']

    # Class distribution and pos weight
    class_counts = y_train.value_counts()
    scale_pos_weight = class_counts[0] / class_counts[1] if len(class_counts) == 2 else 1

    # 3. Train Base Models & Stacking Ensemble (Matching main.py setup)
    print("\n[2/4] Training and evaluating all six models...")
    base_models = {
        'XGBoost': (
            XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42, n_jobs=1, scale_pos_weight=scale_pos_weight),
            {'n_estimators': [200, 300, 500], 'max_depth': [3, 4, 6], 'learning_rate': [0.01, 0.03, 0.05],
             'subsample': [0.7, 0.8, 0.9], 'colsample_bytree': [0.7, 0.8, 0.9], 'min_child_weight': [1, 3, 5]}
        ),
        'Random Forest': (
            RandomForestClassifier(random_state=42, n_jobs=1),
            {'n_estimators': [200, 300, 500], 'max_depth': [None, 5, 10, 20],
             'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4], 'max_features': ['sqrt', 'log2', None]}
        ),
        'LightGBM': (
            LGBMClassifier(objective='binary', random_state=42, verbosity=-1, n_jobs=1),
            {'n_estimators': [200, 300, 500], 'learning_rate': [0.01, 0.03, 0.05], 'num_leaves': [15, 31, 63],
             'max_depth': [-1, 4, 6, 10], 'subsample': [0.7, 0.8, 0.9], 'colsample_bytree': [0.7, 0.8, 0.9]}
        ),
        'Extra Trees': (
            ExtraTreesClassifier(random_state=42, n_jobs=1),
            {'n_estimators': [200, 300, 500], 'max_depth': [None, 5, 10, 20],
             'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4], 'max_features': ['sqrt', 'log2', None]}
        ),
        'CatBoost': (
            CatBoostClassifier(loss_function='Logloss', eval_metric='AUC', random_seed=42, verbose=False),
            {'iterations': [200, 300, 500], 'depth': [4, 6, 8], 'learning_rate': [0.01, 0.03, 0.05]}
        )
    }

    models = {}
    for name, (estimator, grid) in base_models.items():
        print(f"  Tuning {name}...")
        model, _ = tune_classifier(estimator, grid, X_train, y_train)
        models[name] = model

    print("  Fitting Stacking Classifier...")
    stacking_model = build_stacking_classifier(
        models['XGBoost'], models['Random Forest'],
        models['LightGBM'], models['Extra Trees']
    )
    stacking_model.fit(X_train, y_train)
    models['Stacking Classifier'] = stacking_model

    # Class names as required
    class_names = ['Non-DN', 'DN']

    model_results = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc_val = auc(fpr, tpr)
        cm = confusion_matrix(y_test, y_pred)

        model_results[name] = {
            'y_pred': y_pred,
            'y_proba': y_proba,
            'accuracy': acc,
            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': roc_auc_val,
            'cm': cm
        }

    print("\n[3/4] Generating requested plots...")

    # =========================================================================
    # A. Confusion Matrix for Every Model
    # =========================================================================
    print("  Generating individual Confusion Matrices...")
    for name, res in model_results.items():
        fig, ax = plt.subplots(figsize=(7, 6))
        cm = res['cm']
        
        # Annotations format showing TN, FP, FN, TP with values
        tn, fp, fn, tp = cm.ravel()
        labels = np.array([
            [f"True Neg (TN)\n{tn}", f"False Pos (FP)\n{fp}"],
            [f"False Neg (FN)\n{fn}", f"True Pos (TP)\n{tp}"]
        ])

        sns.heatmap(
            cm, annot=labels, fmt='', cmap='Blues', cbar=True,
            xticklabels=class_names, yticklabels=class_names, ax=ax,
            annot_kws={"size": 13, "weight": "bold"}
        )

        ax.set_title(f"{name} - Confusion Matrix", fontsize=15, fontweight='bold', pad=15)
        ax.set_xlabel('Predicted Class', fontsize=12, labelpad=10)
        ax.set_ylabel('True Class', fontsize=12, labelpad=10)
        plt.tight_layout()

        save_path = OUTPUT_DIR / f"confusion_matrix_{name.lower().replace(' ', '_')}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"    Saved: {save_path.name}")
        plt.close(fig)

    # Combined Confusion Matrix Grid for VS Code visualization
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Confusion Matrices for All 6 Models', fontsize=18, fontweight='bold', y=0.98)

    for i, (name, res) in enumerate(model_results.items()):
        ax = axes[i // 3, i % 3]
        cm = res['cm']
        tn, fp, fn, tp = cm.ravel()
        labels = np.array([
            [f"TN: {tn}", f"FP: {fp}"],
            [f"FN: {fn}", f"TP: {tp}"]
        ])
        sns.heatmap(
            cm, annot=labels, fmt='', cmap='Blues', cbar=False,
            xticklabels=class_names, yticklabels=class_names, ax=ax,
            annot_kws={"size": 11, "weight": "bold"}
        )
        ax.set_title(f"{name}", fontsize=13, fontweight='bold')
        ax.set_xlabel('Predicted Class', fontsize=10)
        ax.set_ylabel('True Class', fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    combined_cm_path = OUTPUT_DIR / "all_models_confusion_matrices.png"
    plt.savefig(combined_cm_path, dpi=300, bbox_inches='tight')
    print(f"    Saved: {combined_cm_path.name}")
    plt.close(fig)

    # =========================================================================
    # B. ROC Curve for Every Model
    # =========================================================================
    print("  Generating individual ROC Curves...")
    colors = {
        'XGBoost': '#1f77b4',
        'Random Forest': '#ff7f0e',
        'LightGBM': '#2ca02c',
        'Extra Trees': '#d62728',
        'CatBoost': '#9467bd',
        'Stacking Classifier': '#8c564b'
    }

    for name, res in model_results.items():
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(res['fpr'], res['tpr'], color=colors[name], lw=2.5,
                label=f"{name} (AUC = {res['roc_auc']:.4f})")
        ax.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Random Classifier (AUC = 0.50)')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate (FPR)', fontsize=12, labelpad=10)
        ax.set_ylabel('True Positive Rate (TPR)', fontsize=12, labelpad=10)
        ax.set_title(f"{name} - Receiver Operating Characteristic (ROC)", fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc="lower right", fontsize=11, frameon=True)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = OUTPUT_DIR / f"roc_curve_{name.lower().replace(' ', '_')}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"    Saved: {save_path.name}")
        plt.close(fig)

    # Combined ROC Curves Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    for name, res in model_results.items():
        ax.plot(res['fpr'], res['tpr'], color=colors[name], lw=2,
                label=f"{name} (AUC = {res['roc_auc']:.4f})")
    ax.plot([0, 1], [0, 1], color='black', lw=1.5, linestyle='--', label='Random Chance')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=12, labelpad=10)
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=12, labelpad=10)
    ax.set_title('ROC Curves Comparison - All Models', fontsize=16, fontweight='bold', pad=15)
    ax.legend(loc="lower right", fontsize=10, frameon=True)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    combined_roc_path = OUTPUT_DIR / "all_models_roc_curves.png"
    plt.savefig(combined_roc_path, dpi=300, bbox_inches='tight')
    print(f"    Saved: {combined_roc_path.name}")
    plt.close(fig)

    # =========================================================================
    # C. Model Comparison - Accuracy Bar Chart
    # =========================================================================
    print("  Generating Accuracy Comparison Bar Chart...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    model_names = list(model_results.keys())
    accuracies = [model_results[m]['accuracy'] for m in model_names]
    bar_colors = [colors[m] for m in model_names]

    bars = ax.bar(model_names, accuracies, color=bar_colors, width=0.55, edgecolor='black', linewidth=0.8)

    ax.set_ylim([0.8, 1.0])
    ax.set_ylabel('Accuracy Score', fontsize=12, labelpad=10)
    ax.set_xlabel('Model', fontsize=12, labelpad=10)
    ax.set_title('Model Comparison - Accuracy', fontsize=16, fontweight='bold', pad=15)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Annotate accuracy percentage above each bar
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2%}\n({height:.4f})',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    accuracy_chart_path = OUTPUT_DIR / "model_accuracy_comparison.png"
    plt.savefig(accuracy_chart_path, dpi=300, bbox_inches='tight')
    print(f"    Saved: {accuracy_chart_path.name}")
    plt.close(fig)

    print("\n[4/4] Graph generation completed successfully!")
    print(f"All graphs saved to: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == '__main__':
    main()
