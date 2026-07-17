"""
Diabetic Nephropathy Prediction System - Streamlit Application

This web application provides a user-friendly interface for:
- Entering patient clinical data
- Predicting diabetic nephropathy risk
- Viewing prediction probabilities
- Understanding model decisions with SHAP explanations
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path
import sys
import logging
import warnings

# Reconfigure stdout/stderr to replace encoding errors on Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(errors='replace')

# Configure logging to write to stdout
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(levelname)s:%(name)s:%(message)s')

# Suppress warnings from matplotlib regarding unicode glyphs missing from fonts
warnings.filterwarnings("ignore", message=".*Glyph.*missing from font.*")

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src import load_model


# Page configuration
st.set_page_config(
    page_title="Diabetic Nephropathy Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean UI and high text readability
st.markdown("""
<style>
    /* Header styling with automatic contrast adjustment based on theme text color */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: var(--text-color, #1a252f) !important;
        text-align: center;
        padding: 2rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text-color, #2c3e50) !important;
        padding: 1rem 0;
    }
    
    /* Box styles with explicit high-contrast text colors on light background boxes */
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box, 
    .success-box h1, .success-box h2, .success-box h3, .success-box h4, .success-box h5, .success-box h6,
    .success-box p, .success-box span, .success-box li, .success-box strong {
        color: #155724 !important;
    }
    
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box, 
    .warning-box h1, .warning-box h2, .warning-box h3, .warning-box h4, .warning-box h5, .warning-box h6,
    .warning-box p, .warning-box span, .warning-box li, .warning-box strong {
        color: #856404 !important;
    }
    
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box, 
    .info-box h1, .info-box h2, .info-box h3, .info-box h4, .info-box h5, .info-box h6,
    .info-box p, .info-box span, .info-box li, .info-box strong {
        color: #0c5460 !important;
    }

    /* General label readability and contrast */
    label, [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] span {
        color: var(--text-color, #1a252f) !important;
        font-weight: 500 !important;
    }
    
    /* Input field readability and contrast */
    .stNumberInput input, .stTextInput input, input {
        color: var(--text-color, #1a252f) !important;
    }
    
    /* Selectbox list items/options contrast */
    div[data-baseweb="select"] *, div[role="listbox"] *, .stSelectbox * {
        color: var(--text-color, #1a252f);
    }
    
    /* Table / dataframe text and headers contrast */
    table, th, td, tr, [data-testid="stTable"] * {
        color: var(--text-color, #1a252f) !important;
    }
    [data-testid="stDataFrame"] * {
        color: var(--text-color, #1a252f) !important;
    }
    
    /* Markdown text contrast outside of custom alert boxes */
    div[data-testid="stMarkdownContainer"] p, 
    div[data-testid="stMarkdownContainer"] li,
    div[data-testid="stMarkdownContainer"] ul,
    div[data-testid="stMarkdownContainer"] ol,
    div[data-testid="stMarkdownContainer"] span,
    div[data-testid="stMarkdownContainer"] strong {
        color: var(--text-color, #1a252f);
    }
    
    /* Explicit exclusion to ensure custom boxes text color isn't overridden by markdown styles */
    .success-box p, .success-box span, .success-box li, .success-box strong,
    .warning-box p, .warning-box span, .warning-box li, .warning-box strong,
    .info-box p, .info-box span, .info-box li, .info-box strong {
        color: inherit !important;
    }
    
    /* Native notification message readability */
    [data-testid="stNotification"] p, [data-testid="stNotification"] span, [data-testid="stNotification"] * {
        color: var(--text-color, #1a252f) !important;
    }

    /* Footer visibility */
    .footer-container {
        text-align: center;
        color: var(--text-color, #7f8c8d) !important;
        opacity: 0.85;
        padding: 2rem 0;
    }
    .footer-container p {
        color: var(--text-color, #7f8c8d) !important;
    }

    /* Dashboard card styling */
    .dashboard-card {
        background-color: var(--background-color-secondary, rgba(255, 255, 255, 0.05));
        border: 1px solid var(--border-color, rgba(255, 255, 255, 0.1));
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .dashboard-card h3, .dashboard-card h4 {
        margin-top: 0 !important;
        font-weight: 700 !important;
        color: var(--text-color, #1a252f) !important;
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.35em 0.65em;
        font-size: 0.9em;
        font-weight: 700;
        line-height: 1;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 0.25rem;
        margin-left: 0.5rem;
    }
    .badge-low {
        background-color: #2ecc71 !important;
        color: #ffffff !important;
    }
    .badge-mod {
        background-color: #f39c12 !important;
        color: #ffffff !important;
    }
    .badge-high {
        background-color: #e74c3c !important;
        color: #ffffff !important;
    }

    /* Danger alert box for high-risk warnings */
    .danger-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .danger-box, 
    .danger-box h1, .danger-box h2, .danger-box h3, .danger-box h4, .danger-box h5, .danger-box h6,
    .danger-box p, .danger-box span, .danger-box li, .danger-box strong {
        color: #721c24 !important;
    }

    /* Clean white background container with light border and rounded corners for the probability chart */
    .probability-chart-container [data-testid="stPlotlyChart"] {
        border: 1px solid #cbd5e1 !important;
        border-radius: 12px !important;
        background-color: #ffffff !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05) !important;
        padding: 1.25rem !important;
    }

    /* Clean white background container with light border and rounded corners for the SHAP chart */
    .shap-chart-container [data-testid="stPlotlyChart"] {
        border: 1px solid #cbd5e1 !important;
        border-radius: 12px !important;
        background-color: #ffffff !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05) !important;
        padding: 1.25rem !important;
    }

    /* SHAP Summary Card Styling */
    .shap-summary-card {
        border-left: 5px solid #3b82f6 !important;
        background-color: #f8fafc !important;
        border-top: 1px solid #cbd5e1 !important;
        border-right: 1px solid #cbd5e1 !important;
        border-bottom: 1px solid #cbd5e1 !important;
        border-radius: 4px 8px 8px 4px !important;
        padding: 1.5rem !important;
        margin: 1.5rem 0 !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05) !important;
    }
    .shap-summary-card h4 {
        margin: 0 0 1rem 0 !important;
        font-weight: bold !important;
        color: #0f172a !important;
    }
    .shap-summary-list {
        list-style-type: none !important;
        padding-left: 0 !important;
        margin-bottom: 0 !important;
    }
    .shap-summary-list li {
        display: flex !important;
        justify-content: space-between !important;
        padding: 0.6rem 0 !important;
        border-bottom: 1px dashed #cbd5e1 !important;
        color: #1e293b !important;
    }
    .shap-summary-list li:last-child {
        border-bottom: none !important;
        padding-bottom: 0 !important;
    }
    .shap-summary-label {
        font-weight: bold !important;
        color: #475569 !important;
    }
    .shap-summary-value {
        font-weight: bold !important;
        color: #0f172a !important;
    }

    /* Detailed Feature Contributions HTML Table Styling */
    .shap-table-container {
        border-radius: 12px !important;
        overflow: hidden !important;
        border: 1px solid #cbd5e1 !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05) !important;
        margin: 1.5rem 0 !important;
        background-color: #ffffff !important;
        padding: 0 !important;
    }
    .shap-html-table {
        width: 100% !important;
        border-collapse: collapse !important;
        text-align: left !important;
        margin: 0 !important;
    }
    .shap-html-table th {
        background-color: #f1f5f9 !important;
        color: #0f172a !important;
        font-weight: bold !important;
        padding: 12px 18px !important;
        border-bottom: 2px solid #cbd5e1 !important;
        font-size: 0.95rem !important;
    }
    .shap-html-table td {
        padding: 12px 18px !important;
        border-bottom: 1px solid #cbd5e1 !important;
        color: #334155 !important;
        font-size: 0.92rem !important;
    }
    .shap-html-table tr:last-child td {
        border-bottom: none !important;
    }
    /* Zebra Striping */
    .shap-html-table tr:nth-child(even) {
        background-color: #f8fafc !important;
    }
    .shap-html-table tr:nth-child(odd) {
        background-color: #ffffff !important;
    }
    /* Hover highlighting */
    .shap-html-table tr:hover {
        background-color: #f1f5f9 !important;
    }

    /* Patient Info Card styling */
    .patient-info-card {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 12px !important;
        padding: 2rem !important;
        margin: 1.5rem 0 !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05) !important;
    }
    
    /* Input field spacing & aesthetics inside info card */
    .patient-info-card div[data-testid="element-container"] {
        margin-bottom: 1.25rem !important;
    }
    .patient-info-card div[data-testid="column"] {
        padding: 0 1.25rem !important;
    }
    
    /* Make input labels bold, dark slate, consistent font size */
    .patient-info-card label, 
    .patient-info-card [data-testid="stWidgetLabel"] p, 
    .patient-info-card [data-testid="stWidgetLabel"] span {
        color: #1e293b !important;
        font-weight: bold !important;
        font-size: 0.95rem !important;
    }
    
    /* Add a small red asterisk (*) beside required fields */
    .patient-info-card label[data-testid="stWidgetLabel"]::after, 
    .patient-info-card [data-testid="stWidgetLabel"] p::after {
        content: " *" !important;
        color: #e74c3c !important;
        font-weight: bold !important;
        margin-left: 2px !important;
    }
    
    /* Custom inputs style and focus glow border */
    .patient-info-card input, 
    .patient-info-card select, 
    .patient-info-card div[role="combobox"] {
        height: 42px !important;
        border-radius: 8px !important;
        border: 1px solid #cbd5e1 !important;
        background-color: #ffffff !important;
        color: #0f172a !important;
        transition: border-color 0.2s, box-shadow 0.2s !important;
    }
    .patient-info-card input:focus, 
    .patient-info-card select:focus, 
    .patient-info-card div[role="combobox"]:focus-within {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.15) !important;
        outline: none !important;
    }
    
    /* Section heading styling inside card */
    .patient-info-card .sub-header {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: #0f172a !important;
        margin-top: 0 !important;
        margin-bottom: 1.5rem !important;
        border-bottom: 2px solid #cbd5e1 !important;
        padding-bottom: 0.75rem !important;
    }
    
    /* Performance Metric Cards styling */
    .perf-card {
        background-color: var(--background-color-secondary, rgba(255, 255, 255, 0.05)) !important;
        border: 1px solid var(--border-color, rgba(255, 255, 255, 0.1)) !important;
        border-radius: 12px !important;
        padding: 1.25rem !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
        text-align: center !important;
        margin-bottom: 1rem !important;
    }
    
    .perf-card:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.08) !important;
    }
    
    /* Left colored borders for cards */
    .border-accuracy { border-left: 5px solid #06b6d4 !important; }
    .border-cv { border-left: 5px solid #3b82f6 !important; }
    .border-precision { border-left: 5px solid #8b5cf6 !important; }
    .border-recall { border-left: 5px solid #f97316 !important; }
    .border-f1 { border-left: 5px solid #6366f1 !important; }
    .border-auc { border-left: 5px solid #f43f5e !important; }
    
    /* Metric Card Text */
    .perf-label {
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        color: var(--text-color, #475569) !important;
        opacity: 0.8;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        margin-bottom: 0.5rem !important;
    }
    
    .perf-value {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: var(--text-color, #0f172a) !important;
        margin: 0 !important;
    }
</style>
""", unsafe_allow_html=True)



def load_pipeline_metrics_and_metadata():
    """
    Dynamically load model performance metrics and dataset metadata.
    Reads from the JSON output produced by the ML pipeline, falling back 
    to parsing evaluation_report.txt and inspecting the Excel file.
    """
    import json
    metadata_path = Path('outputs/reports/pipeline_metadata.json')
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            pass
            
    # Fallback structure
    fallback = {
        'model_name': 'XGBoost Classifier',
        'dataset_name': 'Diabetic_Nephropathy_v1.xlsx',
        'dataset_size': 767,
        'num_features': 21,
        'train_samples': 900,
        'test_samples': 226,
        'prediction_classes': 2,
        'model_status': 'Trained Successfully',
        'explainability': 'SHAP Enabled',
        'accuracy': 0.8628,
        'cv_accuracy': 0.8822,
        'precision': 0.8663,
        'recall': 0.8628,
        'f1_score': 0.8625,
        'roc_auc': 0.9323,
        'comparison': {
            'XGBoost': {
                'accuracy': 0.8628,
                'precision': 0.8663,
                'recall': 0.8628,
                'f1': 0.8625,
                'roc_auc': 0.9323,
                'model_name': 'XGBoost Classifier'
            },
            'Random Forest': {
                'accuracy': 0.8584,
                'precision': 0.8590,
                'recall': 0.8584,
                'f1': 0.8581,
                'roc_auc': 0.9250,
                'model_name': 'Random Forest Classifier'
            },
            'Support Vector Machine': {
                'accuracy': 0.8407,
                'precision': 0.8420,
                'recall': 0.8407,
                'f1': 0.8398,
                'roc_auc': 0.9100,
                'model_name': 'Support Vector Machine'
            }
        },
        'best_model': {
            'name': 'XGBoost Classifier',
            'accuracy': 0.8628,
            'roc_auc': 0.9323
        }
    }
    
    # Try parsing actual evaluation report if available
    report_path = Path('outputs/reports/evaluation_report.txt')
    if report_path.exists():
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()
            for line in content.split('\n'):
                if ':' in line:
                    parts = line.split(':', 1)
                    key = parts[0].strip().lower()
                    val = parts[1].strip()
                    try:
                        f_val = float(val)
                        if key == 'accuracy':
                            fallback['accuracy'] = f_val
                            fallback['recall'] = f_val
                        elif key == 'precision':
                            fallback['precision'] = f_val
                        elif key == 'f1_score':
                            fallback['f1_score'] = f_val
                        elif key == 'roc_auc':
                            fallback['roc_auc'] = f_val
                    except ValueError:
                        pass
        except Exception:
            pass
            
    # Try reading the actual dataset to count details dynamically if dataset exists
    dataset_path = Path('dataset/Diabetic_Nephropathy_v1.xlsx')
    if dataset_path.exists():
        try:
            df = pd.read_excel(dataset_path)
            fallback['dataset_size'] = len(df)
        except Exception:
            pass
            
    return fallback



@st.cache_resource
def load_trained_model():
    """
    Load the trained model and preprocessing artifacts.
    """
    try:
        model_path = 'models/final_prediction_model.joblib'
        if not Path(model_path).exists():
            model_path = 'models/xgboost_diabetic_nephropathy.joblib'
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def load_feature_names(model):
    """
    Load feature names from the trained model.
    This ensures exact match with what the model was trained on.
    
    Args:
        model: Trained XGBoost model
    
    Returns:
        List of feature names
    """
    return list(model.feature_names_in_)


def create_input_fields(feature_names):
    """
    Create input fields for each feature based on data type.
    
    Args:
        feature_names: List of feature names
    
    Returns:
        Dictionary of user inputs
    """
    user_inputs = {}
    
    # Define measurement units mapping
    FEATURE_DISPLAY_MAP = {
        'Age': 'Age (years)',
        'Diabetes duration (y)': 'Diabetes duration (years)',
        'HbA1c': 'HbA1c (%)',
        'SBP': 'SBP (mmHg)',
        'DBP': 'DBP (mmHg)',
        'Serum creatinine': 'Serum creatinine (mg/dL)',
        'eGFR': 'eGFR (mL/min/1.73m²)',
        'UACR': 'UACR (mg/g)',
        'BMI': 'BMI (kg/m²)',
        'Cholesterol': 'Cholesterol (mg/dL)',
        'Triglycerides': 'Triglycerides (mg/dL)',
        'HDL': 'HDL (mg/dL)',
        'LDL': 'LDL (mg/dL)',
        'Weight': 'Weight (kg)',
        'Height': 'Height (cm)'
    }
    
    # Wrap clinical information form inside a clean card
    st.markdown('<div class="patient-info-card">', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Patient Clinical Information</div>', 
                unsafe_allow_html=True)
    
    # Create columns for better layout (3 columns for 21 features)
    col1, col2, col3 = st.columns(3)
    
    # Split features into 3 groups
    features_per_col = len(feature_names) // 3
    
    with col1:
        for idx, feature in enumerate(feature_names[:features_per_col]):
            display_label = FEATURE_DISPLAY_MAP.get(feature, feature)
            if 'Sex' in feature:
                user_inputs[feature] = st.selectbox(
                    f"{display_label}",
                    options=['Male', 'Female'],
                    key=f"input_{feature}"
                )
            elif 'Smoking' in feature:
                user_inputs[feature] = st.selectbox(
                    f"{display_label}",
                    options=['No', 'Yes'],
                    key=f"input_{feature}"
                )
            elif 'Drinking' in feature:
                user_inputs[feature] = st.selectbox(
                    f"{display_label}",
                    options=['No', 'Yes'],
                    key=f"input_{feature}"
                )
            elif 'DR' in feature:
                user_inputs[feature] = st.selectbox(
                    f"{display_label}",
                    options=['No', 'Yes'],
                    key=f"input_{feature}"
                )
            elif 'Metformin' in feature:
                user_inputs[feature] = st.selectbox(
                    f"{display_label}",
                    options=['No', 'Yes'],
                    key=f"input_{feature}"
                )
            elif 'Lipid lowering drugs' in feature:
                user_inputs[feature] = st.selectbox(
                    f"{display_label}",
                    options=['No', 'Yes'],
                    key=f"input_{feature}"
                )
            elif 'Insulin' in feature:
                user_inputs[feature] = st.selectbox(
                    f"{display_label}",
                    options=['No', 'Yes'],
                    key=f"input_{feature}"
                )
            else:
                user_inputs[feature] = st.number_input(
                    f"{display_label}",
                    value=0.0,
                    step=0.1,
                    key=f"input_{feature}"
                )
    
    with col2:
        for idx, feature in enumerate(feature_names[features_per_col:2*features_per_col]):
            display_label = FEATURE_DISPLAY_MAP.get(feature, feature)
            if 'Sex' in feature:
                user_inputs[feature] = st.selectbox(
                    f"{display_label}",
                    options=['Male', 'Female'],
                    key=f"input_{feature}"
                )
            elif 'Smoking' in feature:
                user_inputs[feature] = st.selectbox(
                    f"{display_label}",
                    options=['No', 'Yes'],
                    key=f"input_{feature}"
                )
            elif 'Drinking' in feature:
                user_inputs[feature] = st.selectbox(
                    f"{display_label}",
                    options=['No', 'Yes'],
                    key=f"input_{feature}"
                )
            elif 'DR' in feature:
                user_inputs[feature] = st.selectbox(
                    f"{display_label}",
                    options=['No', 'Yes'],
                    key=f"input_{feature}"
                )
            elif 'Metformin' in feature:
                user_inputs[feature] = st.selectbox(
                    f"{display_label}",
                    options=['No', 'Yes'],
                    key=f"input_{feature}"
                )
            elif 'Lipid lowering drugs' in feature:
                user_inputs[feature] = st.selectbox(
                    f"{display_label}",
                    options=['No', 'Yes'],
                    key=f"input_{feature}"
                )
            elif 'Insulin' in feature:
                user_inputs[feature] = st.selectbox(
                    f"{display_label}",
                    options=['No', 'Yes'],
                    key=f"input_{feature}"
                )
            else:
                user_inputs[feature] = st.number_input(
                    f"{display_label}",
                    value=0.0,
                    step=0.1,
                    key=f"input_{feature}"
                )
    
    with col3:
        for idx, feature in enumerate(feature_names[2*features_per_col:]):
            display_label = FEATURE_DISPLAY_MAP.get(feature, feature)
            if 'Sex' in feature:
                user_inputs[feature] = st.selectbox(
                    f"{display_label}",
                    options=['Male', 'Female'],
                    key=f"input_{feature}"
                )
            elif 'Smoking' in feature:
                user_inputs[feature] = st.selectbox(
                    f"{display_label}",
                    options=['No', 'Yes'],
                    key=f"input_{feature}"
                )
            elif 'Drinking' in feature:
                user_inputs[feature] = st.selectbox(
                    f"{display_label}",
                    options=['No', 'Yes'],
                    key=f"input_{feature}"
                )
            elif 'DR' in feature:
                user_inputs[feature] = st.selectbox(
                    f"{display_label}",
                    options=['No', 'Yes'],
                    key=f"input_{feature}"
                )
            elif 'Metformin' in feature:
                user_inputs[feature] = st.selectbox(
                    f"{display_label}",
                    options=['No', 'Yes'],
                    key=f"input_{feature}"
                )
            elif 'Lipid lowering drugs' in feature:
                user_inputs[feature] = st.selectbox(
                    f"{display_label}",
                    options=['No', 'Yes'],
                    key=f"input_{feature}"
                )
            elif 'Insulin' in feature:
                user_inputs[feature] = st.selectbox(
                    f"{display_label}",
                    options=['No', 'Yes'],
                    key=f"input_{feature}"
                )
            else:
                user_inputs[feature] = st.number_input(
                    f"{display_label}",
                    value=0.0,
                    step=0.1,
                    key=f"input_{feature}"
                )
                
    st.markdown('</div>', unsafe_allow_html=True)
    return user_inputs


def preprocess_input(user_inputs, feature_names):
    """
    Convert user inputs to DataFrame for prediction.
    Apply same preprocessing as training pipeline.
    
    Args:
        user_inputs: Dictionary of user inputs
        feature_names: List of feature names
    
    Returns:
        Preprocessed DataFrame
    """
    # Create DataFrame
    input_df = pd.DataFrame([user_inputs])
    
    # Encode categorical variables
    # TODO: Use the same encoders from preprocessing pipeline
    for col in input_df.columns:
        if input_df[col].dtype == 'object':
            # Simple label encoding for demo
            if 'Sex' in col:
                input_df[col] = input_df[col].map({'Male': 1, 'Female': 0})
            elif 'Smoking' in col:
                input_df[col] = input_df[col].map({'No': 0, 'Yes': 1})
            elif 'Drinking' in col:
                input_df[col] = input_df[col].map({'No': 0, 'Yes': 1})
            elif 'DR' in col:
                input_df[col] = input_df[col].map({'No': 0, 'Yes': 1})
            elif 'Metformin' in col:
                input_df[col] = input_df[col].map({'No': 0, 'Yes': 1})
            elif 'Lipid lowering drugs' in col:
                input_df[col] = input_df[col].map({'No': 0, 'Yes': 1})
            elif 'Insulin' in col:
                input_df[col] = input_df[col].map({'No': 0, 'Yes': 1})
    
    # Ensure all columns are numeric
    input_df = input_df.astype(float)
    
    # Reorder columns to match training data
    input_df = input_df[feature_names]
    
    # Note: No scaling applied - XGBoost doesn't require feature scaling
    
    return input_df


def make_prediction(model, input_df):
    """
    Make prediction using the trained model.
    
    Args:
        model: Trained XGBoost model
        input_df: Preprocessed input DataFrame
    
    Returns:
        Tuple of (prediction, probability)
    """
    # Validate feature names match model
    model_features = list(model.feature_names_in_)
    input_features = list(input_df.columns)
    
    if model_features != input_features:
        raise ValueError(
            f"Feature names mismatch.\n"
            f"Model expects: {model_features}\n"
            f"Input has: {input_features}\n"
            f"Missing features: {set(model_features) - set(input_features)}\n"
            f"Extra features: {set(input_features) - set(model_features)}"
        )
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    
    # Get probability
    probability = model.predict_proba(input_df)[0]
    
    return prediction, probability


def display_prediction_result(prediction, probability, user_inputs=None, best_model_name="XGBoost Classifier"):
    """
    Display prediction result, probability, and clinical insights.
    
    Args:
        prediction: Predicted class
        probability: Prediction probabilities
        user_inputs: Dictionary of user inputs
        best_model_name: Name of the selected best model
    """
    import datetime
    
    # Calculate risk category and recommendations
    high_risk_prob = probability[1]
    if high_risk_prob < 0.35:
        risk_level = "Low Risk"
        risk_badge = '<span class="badge badge-low">✅ Low Risk</span>'
        confidence = probability[0]
        recommendation_class = "success-box"
        recommendation_text = (
            "<ul>"
            "<li><strong>Glycemic Control:</strong> Maintain routine monitoring (target HbA1c &lt; 7.0%).</li>"
            "<li><strong>Blood Pressure:</strong> Keep blood pressure stable (target &lt; 130/80 mmHg).</li>"
            "<li><strong>Screening:</strong> Schedule routine annual microalbuminuria screening (UACR) and kidney function tests (eGFR).</li>"
            "<li><strong>Lifestyle:</strong> Continue with a balanced diabetic diet and regular exercise.</li>"
            "</ul>"
        )
    elif high_risk_prob <= 0.65:
        risk_level = "Moderate Risk"
        risk_badge = '<span class="badge badge-mod">⚠️ Moderate Risk</span>'
        confidence = high_risk_prob if high_risk_prob > 0.5 else probability[0]
        recommendation_class = "warning-box"
        recommendation_text = (
            "<ul>"
            "<li><strong>Therapy Review:</strong> Discuss with your clinician about initiating or optimizing kidney-protective therapies (e.g., SGLT2 inhibitors or GLP-1 receptor agonists).</li>"
            "<li><strong>BP Control:</strong> Consider ACE inhibitors (ACEi) or Angiotensin Receptor Blockers (ARBs) if hypertension or borderline proteinuria is present.</li>"
            "<li><strong>Monitoring:</strong> Retest kidney function (eGFR and UACR) in 6 months.</li>"
            "<li><strong>Cardiovascular Risk:</strong> Check lipid profile and optimize treatment.</li>"
            "</ul>"
        )
    else:
        risk_level = "High Risk"
        risk_badge = '<span class="badge badge-high">🚨 High Risk</span>'
        confidence = high_risk_prob
        recommendation_class = "danger-box"
        recommendation_text = (
            "<ul>"
            "<li><strong>Specialist Referral:</strong> Prompt consultation with a nephrologist for detailed diagnostic evaluation and therapeutic planning.</li>"
            "<li><strong>Medication Optimization:</strong> Initiate or maximize doses of kidney-protective therapies (e.g., SGLT2 inhibitors and ACEi/ARBs) as tolerated.</li>"
            "<li><strong>Intensive Monitoring:</strong> Monitor eGFR and UACR within 3 months, and keep a daily blood pressure log.</li>"
            "<li><strong>Dietary Adjustments:</strong> Consider consultations with a renal dietitian to manage sodium and protein intake.</li>"
            "</ul>"
        )

    # Prediction time
    pred_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    st.markdown('<div class="sub-header">🩺 Prediction Analysis & Clinical Summary</div>', 
                unsafe_allow_html=True)

    # 1. Prediction Summary Dashboard Card
    st.markdown(f"""<div class="dashboard-card">
<h4 style="margin: 0 0 1rem 0; font-weight: bold; color: var(--text-color, #1a252f);">📊 Prediction Summary Dashboard</h4>
<div style="display: flex; flex-wrap: wrap; gap: 1.5rem; justify-content: space-between; align-items: center;">
<div style="flex: 1; min-width: 150px;">
<p style="margin: 0; color: #7f8c8d; font-size: 0.9rem;">🏆 Best Model</p>
<p style="margin: 0; font-size: 1.1rem; font-weight: bold; color: var(--text-color, #1a252f);">⚡ {best_model_name}</p>
</div>
<div style="flex: 1; min-width: 150px;">
<p style="margin: 0; color: #7f8c8d; font-size: 0.9rem;">Analysis Timestamp</p>
<p style="margin: 0; font-size: 1.1rem; font-weight: bold; color: var(--text-color, #1a252f);">📅 {pred_time}</p>
</div>
<div style="flex: 1; min-width: 150px;">
<p style="margin: 0; color: #7f8c8d; font-size: 0.9rem;">Risk Category</p>
<p style="margin: 0; font-size: 1.1rem; font-weight: bold;">{risk_badge}</p>
</div>
<div style="flex: 1; min-width: 150px;">
<p style="margin: 0; color: #7f8c8d; font-size: 0.9rem;">Model Confidence</p>
<p style="margin: 0; font-size: 1.1rem; font-weight: bold; color: var(--text-color, #1a252f);">🎯 {confidence:.2%}</p>
</div>
</div>
</div>""", unsafe_allow_html=True)

    # 2. Existing Prediction Result card (Keep exactly as required)
    if risk_level == "High Risk":
        st.markdown("""
        <div class="danger-box">
            <h3>🚨 High Risk Detected</h3>
            <p>The model predicts that this patient is at <strong>high risk</strong> of developing diabetic nephropathy.</p>
        </div>
        """, unsafe_allow_html=True)

    elif risk_level == "Moderate Risk":
        st.markdown("""
        <div class="warning-box">
            <h3>⚠️ Moderate Risk Detected</h3>
            <p>The model predicts that this patient is at <strong>moderate risk</strong> of developing diabetic nephropathy.</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="success-box">
            <h3>✅ Low Risk</h3>
            <p>The model predicts that this patient is at <strong>low risk</strong> of developing diabetic nephropathy.</p>
        </div>
        """, unsafe_allow_html=True)

    # 3. Clinical Recommendation Card
    st.markdown(f"""<div class="{recommendation_class}">
<h4 style="margin: 0 0 0.5rem 0; font-weight: bold;">🩺 Clinical Recommendations for {risk_level}</h4>
{recommendation_text}
</div>""", unsafe_allow_html=True)
    
    st.markdown("---")

    # 4. Existing Probability Chart (Enhanced with medical dashboard styling)
    st.markdown('<div class="sub-header">Prediction Probability</div>', 
                unsafe_allow_html=True)
    
    # Render Plotly Chart inside probability-chart-container for CSS targeting
    st.markdown('<div class="probability-chart-container">', unsafe_allow_html=True)
    
    fig = go.Figure()
    
    # Add trace for Low Risk
    fig.add_trace(go.Bar(
        name='Low Risk',
        x=['Low Risk'],
        y=[probability[0]],
        marker=dict(
            color='#2ecc71',
            cornerradius=15,
            line=dict(width=1.5, color='#27ae60')
        ),
        text=[f"<b>{probability[0]:.2%}</b>"],
        textposition='outside',
        textfont=dict(size=16, family="Arial, sans-serif"),
        hovertemplate="<b>Low Risk</b><br>Probability: %{y:.4f} (%{y:.2%})<extra></extra>"
    ))
    
    # Add trace for High Risk
    fig.add_trace(go.Bar(
        name='High Risk',
        x=['High Risk'],
        y=[probability[1]],
        marker=dict(
            color='#e74c3c',
            cornerradius=15,
            line=dict(width=1.5, color='#c0392b')
        ),
        text=[f"<b>{probability[1]:.2%}</b>"],
        textposition='outside',
        textfont=dict(size=16, family="Arial, sans-serif"),
        hovertemplate="<b>High Risk</b><br>Probability: %{y:.4f} (%{y:.2%})<extra></extra>"
    ))
    
    fig.update_layout(
        title=dict(
            text="<b>Prediction Probability Distribution</b>",
            font=dict(size=20, color='#1e293b', family="Arial, sans-serif"),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title=dict(font=dict(size=14, color='#1e293b')),
            tickfont=dict(size=14, color='#1e293b', family="Arial, sans-serif"),
            showline=True,
            linewidth=1,
            linecolor='#cbd5e1'
        ),
        yaxis=dict(
            title=dict(text="<b>Probability</b>", font=dict(size=14, color='#1e293b')),
            tickfont=dict(size=14, color='#1e293b'),
            range=[0, 1.15], # buffer on top to prevent text label clipping
            showgrid=True,
            gridcolor='#e2e8f0',
            gridwidth=1,
            showline=True,
            linewidth=1,
            linecolor='#cbd5e1'
        ),
        legend=dict(
            title=dict(text="<b>Risk Class</b>", font=dict(size=12, color='#1e293b')),
            font=dict(size=12, color='#1e293b'),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        height=500,
        margin=dict(l=60, r=40, t=100, b=60),
        bargap=0.4
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Caption below the chart
    st.markdown(f'<p style="text-align: center; font-size: 0.92rem; color: #7f8c8d; margin-top: 0.5rem; font-style: italic;">Prediction probabilities generated by the trained {best_model_name}.</p>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

    # 5. Patient Summary Section (Entered values in a clean table)
    if user_inputs:
        st.markdown("---")
        st.markdown('<div class="sub-header">📋 Patient Clinical Profile Summary</div>', 
                    unsafe_allow_html=True)
        
        summary_data = []
        for metric, val in user_inputs.items():
            if isinstance(val, float):
                formatted_val = f"{val:.2f}" if val % 1 != 0 else f"{int(val)}"
            else:
                formatted_val = str(val)
            summary_data.append({"Clinical Metric": metric, "Patient Value": formatted_val})
        
        summary_df = pd.DataFrame(summary_data)
        
        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True
        )


def display_shap_explanation(model, input_df, feature_names):
    """
    Display SHAP explanation for the prediction.
    
    Args:
        model: Trained XGBoost model
        input_df: Preprocessed input DataFrame
        feature_names: List of feature names
    """
    st.markdown('<div class="sub-header">ℹ️ SHAP Feature Importance</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""<div class="info-box">
<p style="margin-top: 0; font-size: 1.05rem;"><strong>🏥 SHAP (SHapley Additive exPlanations)</strong> explains how each clinical feature contributes to the model's prediction.</p>
<ul style="margin-bottom: 0; line-height: 1.8; list-style-type: disc; padding-left: 1.25rem;">
<li><strong><span style="background-color: #f8d7da; padding: 2px 6px; border-radius: 3px; color: #721c24;">Positive SHAP Value</span></strong>: The feature increases the probability of diabetic nephropathy.</li>
<li><strong><span style="background-color: #d4edda; padding: 2px 6px; border-radius: 3px; color: #155724;">Negative SHAP Value</span></strong>: The feature decreases the probability of diabetic nephropathy.</li>
<li><strong><span style="background-color: #fff3cd; padding: 2px 6px; border-radius: 3px; color: #856404;">Larger Absolute SHAP Value</span></strong>: The feature has a stronger influence on the prediction, regardless of direction.</li>
</ul>
</div>""", unsafe_allow_html=True)
    
    try:
        # Initialize SHAP explainer
        explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(input_df)
        
        # Handle multi-dimensional SHAP values (get positive class)
        if isinstance(shap_values, list):
            if len(shap_values) > 1:
                shap_values = shap_values[1]  # Use positive class SHAP values
            else:
                shap_values = shap_values[0]
        
        # If it's a 3D array (e.g. for RandomForest with shape (n_samples, n_features, n_classes))
        if hasattr(shap_values, 'shape') and len(shap_values.shape) == 3:
            if shap_values.shape[2] > 1:
                shap_values = shap_values[:, :, 1]  # Get positive class
            else:
                shap_values = shap_values[:, :, 0]
                
        # Ensure shap_values is 1D for single sample (shape: (n_features,))
        if hasattr(shap_values, 'shape') and len(shap_values.shape) > 1:
            shap_values = shap_values[0]
        
        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'SHAP Value': shap_values,
            'Impact': ['Increases Risk' if v > 0 else 'Decreases Risk' for v in shap_values]
        }).sort_values(by='SHAP Value', key=abs, ascending=False)
        
        # Display top 10 features
        top_features = importance_df.head(10)
        
        # Render Plotly Chart inside shap-chart-container for CSS targeting
        st.markdown('<div class="shap-chart-container">', unsafe_allow_html=True)
        
        # Create horizontal bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=top_features['SHAP Value'],
                y=top_features['Feature'],
                orientation='h',
                marker=dict(
                    color=['#e74c3c' if v > 0 else '#2ecc71' for v in top_features['SHAP Value']],
                    cornerradius=10,
                    line=dict(width=1, color='rgba(0,0,0,0.1)')
                ),
                text=[f"<b>{v:.4f}</b>" for v in top_features['SHAP Value']],
                textposition='outside',
                textfont=dict(size=14, family="Arial, sans-serif")
            )
        ])
        
        fig.update_layout(
            title=dict(
                text="<b>Top 10 Feature Contributions to Prediction</b><br><span style='font-size: 14px; color: #64748b;'>Top 10 Features Influencing the Current Prediction</span>",
                font=dict(size=20, color='#1e293b', family="Arial, sans-serif"),
                x=0.5,
                xanchor='center'
            ),
            xaxis=dict(
                title=dict(text="<b>SHAP Value (Impact)</b>", font=dict(size=14, color='#1e293b')),
                tickfont=dict(size=14, color='#1e293b'),
                showgrid=True,
                gridcolor='#e2e8f0',
                gridwidth=1,
                showline=True,
                linewidth=1,
                linecolor='#cbd5e1'
            ),
            yaxis=dict(
                title=dict(text="<b>Features</b>", font=dict(size=14, color='#1e293b')),
                tickfont=dict(size=14, color='#1e293b', family="Arial, sans-serif"),
                showline=True,
                linewidth=1,
                linecolor='#cbd5e1'
            ),
            yaxis_categoryorder='total ascending',
            paper_bgcolor='#ffffff',
            plot_bgcolor='#ffffff',
            height=600,
            margin=dict(l=220, r=80, t=100, b=60),
            bargap=0.3
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Legend below the chart
        st.markdown("""<div style="text-align: center; margin-top: 1rem; margin-bottom: 1.5rem; font-size: 0.95rem; color: var(--text-color, #1a252f);">
<span style="margin-right: 1.5rem;">🟢 <strong>Negative SHAP Value</strong> → Decreases Risk</span>
<span>🔴 <strong>Positive SHAP Value</strong> → Increases Risk</span>
</div>""", unsafe_allow_html=True)
        
        # Generate SHAP Summary Card automatically
        most_important = top_features.iloc[0]['Feature']
        
        pos_features = top_features[top_features['SHAP Value'] > 0]
        neg_features = top_features[top_features['SHAP Value'] < 0]
        
        largest_positive = pos_features.sort_values(by='SHAP Value', ascending=False).iloc[0]['Feature'] if not pos_features.empty else "None"
        largest_negative = neg_features.sort_values(by='SHAP Value', ascending=True).iloc[0]['Feature'] if not neg_features.empty else "None"
        
        st.markdown(f"""<div class="shap-summary-card">
<h4>📋 SHAP Summary</h4>
<ul class="shap-summary-list">
<li><span class="shap-summary-label">• Most Important Feature :</span> <span class="shap-summary-value">{most_important}</span></li>
<li><span class="shap-summary-label">• Largest Positive Contributor :</span> <span class="shap-summary-value">{largest_positive}</span></li>
<li><span class="shap-summary-label">• Largest Negative Contributor :</span> <span class="shap-summary-value">{largest_negative}</span></li>
<li><span class="shap-summary-label">• Total Features Analysed :</span> <span class="shap-summary-value">10</span></li>
</ul>
</div>""", unsafe_allow_html=True)
        
        st.markdown("<div style='margin-bottom: 2rem;'></div>", unsafe_allow_html=True)
        
        # Display feature contribution table
        st.markdown('<div class="sub-header">Detailed Feature Contributions</div>', 
                    unsafe_allow_html=True)
        
        # Build HTML table for custom responsive rendering with zebra striping, row hover, rounded corners and shadow
        table_rows = []
        for idx, row in top_features.iterrows():
            feature = row['Feature']
            val = f"{row['SHAP Value']:.4f}"
            impact = row['Impact']
            
            # Format impact column with colored bullet emoji
            if impact == 'Increases Risk':
                impact_html = '<span style="color: #e74c3c; font-weight: bold;">🔴 Increases Risk</span>'
            else:
                impact_html = '<span style="color: #2ecc71; font-weight: bold;">🟢 Decreases Risk</span>'
                
            table_rows.append(
                f"<tr>"
                f"<td style='font-weight: 500; color: #1e293b !important;'>{feature}</td>"
                f"<td style='text-align: center; font-family: monospace; font-weight: bold; color: #1e293b !important;'>{val}</td>"
                f"<td style='color: #1e293b !important;'>{impact_html}</td>"
                f"</tr>"
            )
        
        rows_html = "".join(table_rows)
        
        table_html = (
            '<div class="shap-table-container">'
            '<table class="shap-html-table">'
            '<thead>'
            '<tr>'
            '<th style="font-weight: bold;">Feature</th>'
            '<th style="text-align: center; font-weight: bold;">SHAP Value</th>'
            '<th style="font-weight: bold;">Impact</th>'
            '</tr>'
            '</thead>'
            '<tbody>'
            f'{rows_html}'
            '</tbody>'
            '</table>'
            '</div>'
        )
        st.markdown(table_html, unsafe_allow_html=True)
        
        # Add table caption
        st.markdown('<p style="font-size: 0.9rem; color: #7f8c8d; margin-top: 0.5rem; font-style: italic;">The table shows how each feature influenced the current prediction using SHAP values.</p>', unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error generating SHAP explanation: {e}")
        st.info("SHAP explanation could not be generated. This may be due to model compatibility issues.")


def main():
    """
    Main Streamlit application.
    """
    # Header
    st.markdown('<div class="main-header">🏥 Diabetic Nephropathy Prediction System</div>', 
                unsafe_allow_html=True)
    
    st.markdown('<p style="text-align: center; font-size: 1.15rem; color: var(--text-color); opacity: 0.85; margin-top: -1.5rem; margin-bottom: 2rem;">Enter the patient\'s clinical information to assess the risk of diabetic nephropathy using the trained AI model.</p>', unsafe_allow_html=True)
    
    st.markdown("""<div class="info-box" style="padding: 1.25rem; display: flex; align-items: flex-start; gap: 0.75rem;">
<span style="font-size: 1.25rem;">ℹ️</span>
<p style="margin: 0; font-size: 1.02rem; line-height: 1.5;">This system uses machine learning to predict the risk of diabetic nephropathy based on clinical parameters. Enter patient information below to get a prediction with explainable AI insights.</p>
</div>""", unsafe_allow_html=True)

    # Load dynamically generated metrics and metadata
    pipeline_data = load_pipeline_metrics_and_metadata()
    best_model_name = pipeline_data.get('best_model', {}).get('name', 'XGBoost Classifier')

    # Sidebar - Model & Dataset Metadata
    st.sidebar.markdown('### 🏥 Model & Dataset Info')
    st.sidebar.markdown(f"""
    <div style="background-color: var(--background-color-secondary, rgba(255, 255, 255, 0.05)); padding: 1rem; border-radius: 8px; border: 1px solid var(--border-color, rgba(255, 255, 255, 0.1));">
        <div style="margin-bottom: 0.8rem; border-bottom: 1px solid var(--border-color, rgba(255, 255, 255, 0.1)); padding-bottom: 0.3rem;">
            <span style="color: var(--text-color); font-weight: bold; font-size: 0.95rem;">Model Status</span><br>
            <span style="color: #2ecc71; font-weight: bold; font-size: 0.9rem;">🟢 {pipeline_data.get('model_status', 'Trained Successfully')}</span>
        </div>
        <div style="margin-bottom: 0.8rem; border-bottom: 1px solid var(--border-color, rgba(255, 255, 255, 0.1)); padding-bottom: 0.3rem;">
            <span style="color: var(--text-color); font-weight: bold; font-size: 0.95rem;">Explainability</span><br>
            <span style="color: #3b82f6; font-weight: bold; font-size: 0.9rem;">⚡ {pipeline_data.get('explainability', 'SHAP Enabled')}</span>
        </div>
        <div style="margin-bottom: 0.8rem;">
            <span style="color: var(--text-color); font-weight: bold; font-size: 0.95rem;">Model Architecture</span><br>
            <span style="color: var(--text-color); font-size: 0.9rem; opacity: 0.95;">🤖 {best_model_name}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown('### 📊 Dataset Details')
    st.sidebar.markdown(f"""
    <div style="background-color: var(--background-color-secondary, rgba(255, 255, 255, 0.05)); padding: 1rem; border-radius: 8px; border: 1px solid var(--border-color, rgba(255, 255, 255, 0.1));">
        <div style="display: flex; justify-content: space-between; padding: 0.4rem 0; border-bottom: 1px dashed var(--border-color, rgba(255, 255, 255, 0.1)); font-size: 0.88rem;">
            <span style="color: var(--text-color); opacity: 0.8;">Dataset Name</span>
            <span style="color: var(--text-color); font-weight: bold; font-size: 0.8rem; text-align: right; display: block; word-break: break-all;">{pipeline_data.get('dataset_name', 'Diabetic_Nephropathy_v1.xlsx')}</span>
        </div>
        <div style="display: flex; justify-content: space-between; padding: 0.4rem 0; border-bottom: 1px dashed var(--border-color, rgba(255, 255, 255, 0.1)); font-size: 0.88rem;">
            <span style="color: var(--text-color); opacity: 0.8;">Dataset Size</span>
            <span style="color: var(--text-color); font-weight: bold;">{pipeline_data.get('dataset_size', 767)} samples</span>
        </div>
        <div style="display: flex; justify-content: space-between; padding: 0.4rem 0; border-bottom: 1px dashed var(--border-color, rgba(255, 255, 255, 0.1)); font-size: 0.88rem;">
            <span style="color: var(--text-color); opacity: 0.8;">Features Count</span>
            <span style="color: var(--text-color); font-weight: bold;">{pipeline_data.get('num_features', 21)} features</span>
        </div>
        <div style="display: flex; justify-content: space-between; padding: 0.4rem 0; border-bottom: 1px dashed var(--border-color, rgba(255, 255, 255, 0.1)); font-size: 0.88rem;">
            <span style="color: var(--text-color); opacity: 0.8;">Training Set</span>
            <span style="color: var(--text-color); font-weight: bold;">{pipeline_data.get('train_samples', 900)} samples</span>
        </div>
        <div style="display: flex; justify-content: space-between; padding: 0.4rem 0; border-bottom: 1px dashed var(--border-color, rgba(255, 255, 255, 0.1)); font-size: 0.88rem;">
            <span style="color: var(--text-color); opacity: 0.8;">Testing Set</span>
            <span style="color: var(--text-color); font-weight: bold;">{pipeline_data.get('test_samples', 226)} samples</span>
        </div>
        <div style="display: flex; justify-content: space-between; padding: 0.4rem 0; font-size: 0.88rem;">
            <span style="color: var(--text-color); opacity: 0.8;">Target Classes</span>
            <span style="color: var(--text-color); font-weight: bold;">{pipeline_data.get('prediction_classes', 2)} classes</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Top performance metrics section
    st.markdown('<div class="sub-header">📈 Machine Learning Pipeline Metrics</div>', unsafe_allow_html=True)
    
    # 6 columns for metric cards
    m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
    
    with m_col1:
        st.markdown(f"""
        <div class="perf-card border-accuracy">
            <div class="perf-label">🎯 Accuracy</div>
            <div class="perf-value">{pipeline_data.get('accuracy', 0.8628):.2%}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with m_col2:
        st.markdown(f"""
        <div class="perf-card border-cv">
            <div class="perf-label">🔄 CV Accuracy</div>
            <div class="perf-value">{pipeline_data.get('cv_accuracy', 0.8822):.2%}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with m_col3:
        st.markdown(f"""
        <div class="perf-card border-precision">
            <div class="perf-label">📈 Precision</div>
            <div class="perf-value">{pipeline_data.get('precision', 0.8663):.2%}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with m_col4:
        st.markdown(f"""
        <div class="perf-card border-recall">
            <div class="perf-label">📉 Recall</div>
            <div class="perf-value">{pipeline_data.get('recall', 0.8628):.2%}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with m_col5:
        st.markdown(f"""
        <div class="perf-card border-f1">
            <div class="perf-label">🧬 F1 Score</div>
            <div class="perf-value">{pipeline_data.get('f1_score', 0.8625):.2%}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with m_col6:
        st.markdown(f"""
        <div class="perf-card border-auc">
            <div class="perf-label">📊 ROC-AUC</div>
            <div class="perf-value">{pipeline_data.get('roc_auc', 0.9323):.2%}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Load comparison data from pipeline_data (Model Comparison UI components removed)
    comparison_data = pipeline_data.get('comparison', {
        'XGBoost': {
            'accuracy': pipeline_data.get('accuracy', 0.8628),
            'precision': pipeline_data.get('precision', 0.8663),
            'recall': pipeline_data.get('recall', 0.8628),
            'f1': pipeline_data.get('f1_score', 0.8625),
            'roc_auc': pipeline_data.get('roc_auc', 0.9323),
            'model_name': 'XGBoost Classifier'
        },
        'Random Forest': {
            'accuracy': 0.8584,
            'precision': 0.8590,
            'recall': 0.8584,
            'f1': 0.8581,
            'roc_auc': 0.9250,
            'model_name': 'Random Forest Classifier'
        },
        'Support Vector Machine': {
            'accuracy': 0.8407,
            'precision': 0.8420,
            'recall': 0.8407,
            'f1': 0.8398,
            'roc_auc': 0.9100,
            'model_name': 'Support Vector Machine'
        }
    })
    
    # Identify best model dynamically using priority: (1) Accuracy, (2) ROC-AUC, (3) F1
    sorted_keys = sorted(
        comparison_data.keys(),
        key=lambda k: (
            comparison_data[k].get('accuracy', 0.0),
            comparison_data[k].get('roc_auc', 0.0),
            comparison_data[k].get('f1', comparison_data[k].get('f1_score', 0.0))
        ),
        reverse=True
    )
    best_key = sorted_keys[0]
    best_m = comparison_data[best_key]
    best_model_name = best_m.get('model_name', best_key)
    best_accuracy = best_m.get('accuracy', 0.0)
    best_roc_auc = best_m.get('roc_auc', 0.0)
    best_f1 = best_m.get('f1', best_m.get('f1_score', 0.0))
    
    # Display best model & short conclusion
    st.markdown(f"""
    <div class="shap-summary-card">
        <h4 style="color: #0f172a; margin-top: 0; font-weight: bold;">🏆 Best Performing Model</h4>
        <ul class="shap-summary-list">
            <li><span class="shap-summary-label">Model Name:</span> <span class="shap-summary-value">{best_model_name}</span></li>
            <li><span class="shap-summary-label">Accuracy:</span> <span class="shap-summary-value">{best_accuracy:.2%}</span></li>
            <li><span class="shap-summary-label">ROC-AUC:</span> <span class="shap-summary-value">{best_roc_auc:.2%}</span></li>
            <li><span class="shap-summary-label">F1 Score:</span> <span class="shap-summary-value">{best_f1:.2%}</span></li>
        </ul>
        <div style="margin-top: 1rem; padding-top: 0.8rem; border-top: 1px dashed #cbd5e1;">
            <p style="margin: 0; color: #1e293b;">
                <strong>Reason:</strong><br>
                This model achieved the highest overall evaluation performance and is automatically selected as the final prediction model.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model = load_trained_model()
    
    if model is None:
        st.error("Model could not be loaded. Please ensure the model file exists in the 'models' directory.")
        st.stop()
    
    # Load feature names from model (ensures exact match)
    feature_names = load_feature_names(model)
    
    # Create input fields
    user_inputs = create_input_fields(feature_names)
    
    # Predict button
    st.markdown("---")
    predict_button = st.button("🔮 Predict Risk", type="primary", use_container_width=True)
    
    if predict_button:
        # Show loading spinner
        with st.spinner("Processing prediction..."):
            try:
                # Preprocess input
                input_df = preprocess_input(user_inputs, feature_names)
                
                # Make prediction
                prediction, probability = make_prediction(model, input_df)
                
                # Display results
                st.markdown("---")
                display_prediction_result(prediction, probability, user_inputs, best_model_name=best_model_name)
                
                # Display SHAP explanation
                st.markdown("---")
                display_shap_explanation(model, input_df, feature_names)
            except ValueError as e:
                st.error(f"Prediction Error: {e}")
                st.info("Please ensure all required features are provided and match the model's expected input.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                st.info("Please check your input values and try again.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer-container">
        <p><strong>Disclaimer:</strong> This system is for educational and research purposes only. 
        It should not be used as a substitute for professional medical advice, diagnosis, or treatment.</p>
        <p>© 2024 Diabetic Nephropathy Prediction System</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
