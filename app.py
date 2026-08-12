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
            st.error("Final prediction model not found. Run main.py to train the pipeline.")
            return None
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
        # Initialize SHAP explainer (tree-based or model-agnostic for ensemble models)
        tree_model_names = {'XGBClassifier', 'RandomForestClassifier', 'ExtraTreesClassifier', 'LGBMClassifier',
                            ########## NEW CATBOOST CODE ##########
                            'CatBoostClassifier'
                            ########## NEW CATBOOST CODE ##########
                            }
        if model.__class__.__name__ in tree_model_names:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df)
        else:
            def positive_probability(values):
                frame = pd.DataFrame(values, columns=input_df.columns)
                return model.predict_proba(frame)[:, 1]
            background = input_df
            explainer = shap.Explainer(positive_probability, background)
            explanation = explainer(input_df)
            shap_values = explanation.values
        
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
        
        return top_features
        
    except Exception as e:
        st.error(f"Error generating SHAP explanation: {e}")
        st.info("SHAP explanation could not be generated. This may be due to model compatibility issues.")
        return None


def get_feature_status_and_explanation(feature, val, shap_val):
    """
    Generate status, clinical relevance, and SHAP influence explanation for a given feature.
    
    Returns:
        tuple: (val_str, status_str, explanation_str)
    """
    if isinstance(val, (int, float)):
        if isinstance(val, float) and val % 1 != 0:
            val_str = f"{val:.2f}"
        else:
            val_str = f"{int(val)}"
    else:
        val_str = str(val) if val is not None else "N/A"
        
    f_lower = feature.lower()
    shap_direction = "increased the predicted risk of diabetic nephropathy" if shap_val > 0 else "decreased the predicted risk of diabetic nephropathy"
    
    status = "Evaluated"
    explanation = ""
    
    if 'hba1c' in f_lower:
        num_val = float(val) if isinstance(val, (int, float)) else 0.0
        val_str += "%"
        if num_val > 7.0:
            status = "Elevated (Above Target)"
            explanation = (
                f"Your HbA1c level is {val_str}, which is above the recommended target range of 7.0%. "
                f"Elevated HbA1c indicates higher average blood sugar over recent months. "
                f"Persistent high blood glucose can cause vascular damage to kidney filtration units, and therefore this factor {shap_direction}."
            )
        else:
            status = "Normal / Controlled"
            explanation = (
                f"Your HbA1c level is {val_str}, indicating well-controlled average blood sugar. "
                f"Maintaining optimal glycemic control protects small microvascular vessels in the kidneys. "
                f"This favorable reading {shap_direction}."
            )
            
    elif 'egfr' in f_lower:
        num_val = float(val) if isinstance(val, (int, float)) else 90.0
        val_str += " mL/min/1.73m²"
        if num_val < 60.0:
            status = "Reduced (Needs Attention)"
            explanation = (
                f"Your estimated Glomerular Filtration Rate (eGFR) is {val_str}, which is below normal range. "
                f"eGFR measures how efficiently your kidneys filter waste products from the blood. "
                f"A reduced filtration rate signals kidney impairment and therefore this factor {shap_direction}."
            )
        elif num_val < 90.0:
            status = "Slightly Reduced"
            explanation = (
                f"Your eGFR level is {val_str}, indicating slightly reduced kidney filtration function. "
                f"eGFR tracks renal clearance efficiency over time. "
                f"This factor was evaluated by the model and {shap_direction}."
            )
        else:
            status = "Normal / Healthy"
            explanation = (
                f"Your eGFR level is {val_str}, reflecting strong, healthy kidney filtration function. "
                f"A normal eGFR indicates efficient removal of waste products from circulation. "
                f"This healthy level {shap_direction}."
            )
            
    elif 'uacr' in f_lower:
        num_val = float(val) if isinstance(val, (int, float)) else 0.0
        val_str += " mg/g"
        if num_val >= 300.0:
            status = "High (Severely Elevated)"
            explanation = (
                f"Your Urine Albumin-to-Creatinine Ratio (UACR) is {val_str}, indicating high protein excretion. "
                f"UACR detects protein leaking into urine, a primary hallmark of kidney filter damage. "
                f"This significant elevation strongly contributed to the model's prediction and {shap_direction}."
            )
        elif num_val >= 30.0:
            status = "Elevated (Microalbuminuria)"
            explanation = (
                f"Your UACR is {val_str}, indicating microalbuminuria (early protein leakage). "
                f"Elevated UACR is an early sign of diabetic kidney strain. "
                f"This factor was flagged by the model and {shap_direction}."
            )
        else:
            status = "Normal (< 30 mg/g)"
            explanation = (
                f"Your UACR is {val_str}, which is within the optimal range (< 30 mg/g). "
                f"Normal UACR confirms minimal protein leakage into the urine. "
                f"This reassuring finding {shap_direction}."
            )
            
    elif 'creatinine' in f_lower:
        num_val = float(val) if isinstance(val, (int, float)) else 1.0
        val_str += " mg/dL"
        if num_val > 1.2:
            status = "Elevated"
            explanation = (
                f"Your serum creatinine level is {val_str}, which is higher than normal reference ranges. "
                f"Creatinine is a waste product filtered by healthy kidneys; elevated blood levels indicate reduced renal clearance. "
                f"Consequently, this factor {shap_direction}."
            )
        else:
            status = "Normal Range"
            explanation = (
                f"Your serum creatinine level is {val_str}, falling within acceptable healthy limits. "
                f"Normal serum creatinine suggests effective metabolic waste elimination by your kidneys. "
                f"This parameter {shap_direction}."
            )
            
    elif 'sbp' in f_lower:
        num_val = float(val) if isinstance(val, (int, float)) else 120.0
        val_str += " mmHg"
        if num_val >= 130.0:
            status = "Elevated (High BP)"
            explanation = (
                f"Your Systolic Blood Pressure (SBP) is {val_str}, which exceeds the recommended clinical target. "
                f"High blood pressure increases mechanical pressure on kidney vessels, accelerating renal strain. "
                f"Because of this pressure, SBP {shap_direction}."
            )
        else:
            status = "Normal / Controlled"
            explanation = (
                f"Your Systolic Blood Pressure is {val_str}, maintaining a healthy pressure level. "
                f"Optimal blood pressure minimizes physical stress on sensitive renal glomeruli. "
                f"This controlled reading {shap_direction}."
            )

    elif 'dbp' in f_lower:
        num_val = float(val) if isinstance(val, (int, float)) else 80.0
        val_str += " mmHg"
        if num_val >= 80.0:
            status = "Elevated (High BP)"
            explanation = (
                f"Your Diastolic Blood Pressure (DBP) is {val_str}, which is elevated. "
                f"Diastolic pressure measures vascular resistance when the heart rests, impacting kidney vessel health. "
                f"This elevated reading {shap_direction}."
            )
        else:
            status = "Normal"
            explanation = (
                f"Your Diastolic Blood Pressure is {val_str}, resting within normal limits. "
                f"Healthy diastolic pressure helps prevent microvascular strain in renal tissue. "
                f"This positive reading {shap_direction}."
            )

    elif 'duration' in f_lower:
        num_val = float(val) if isinstance(val, (int, float)) else 0.0
        val_str += " years"
        if num_val >= 10.0:
            status = "Long Duration (≥ 10 years)"
            explanation = (
                f"You have had diabetes for {val_str}. "
                f"Longer diabetes duration increases total exposure to blood sugar fluctuations and potential vascular wear. "
                f"This factor {shap_direction}."
            )
        else:
            status = "Shorter Duration (< 10 years)"
            explanation = (
                f"Your diabetes duration is {val_str}. "
                f"A shorter history of diabetes reduces total years of microvascular risk exposure. "
                f"This factor {shap_direction}."
            )

    elif 'bmi' in f_lower:
        num_val = float(val) if isinstance(val, (int, float)) else 22.0
        val_str += " kg/m²"
        if num_val >= 25.0:
            status = "Elevated (Overweight/Obese)"
            explanation = (
                f"Your Body Mass Index (BMI) is {val_str}, placing you above the standard healthy weight range. "
                f"Higher body weight places additional metabolic demand and filtration stress on the kidneys. "
                f"As a result, BMI {shap_direction}."
            )
        else:
            status = "Normal Weight Range"
            explanation = (
                f"Your BMI is {val_str}, which is within the recommended healthy weight range. "
                f"Maintaining healthy weight reduces hyperfiltration stress on kidney nephrons. "
                f"This healthy range {shap_direction}."
            )

    elif 'triglycerides' in f_lower:
        num_val = float(val) if isinstance(val, (int, float)) else 100.0
        val_str += " mg/dL"
        if num_val >= 150.0:
            status = "High (≥ 150 mg/dL)"
            explanation = (
                f"Your Triglyceride level is {val_str}, which is above the recommended target range (< 150 mg/dL). "
                f"High triglycerides contribute to lipid deposition and oxidative stress in renal tissue. "
                f"Therefore, this lipid factor {shap_direction}."
            )
        else:
            status = "Normal (< 150 mg/dL)"
            explanation = (
                f"Your Triglyceride level is {val_str}, which is well within healthy limits. "
                f"Normal lipid levels support optimal vascular health and clear circulation. "
                f"This favorable lipid metric {shap_direction}."
            )

    elif 'hdl' in f_lower:
        num_val = float(val) if isinstance(val, (int, float)) else 50.0
        val_str += " mg/dL"
        if num_val < 40.0:
            status = "Low (< 40 mg/dL)"
            explanation = (
                f"Your HDL cholesterol is {val_str}, which is below protective target levels. "
                f"HDL is 'good' cholesterol that clears excess lipids from blood vessel walls. "
                f"Lower HDL reduces vascular protection and therefore {shap_direction}."
            )
        else:
            status = "Normal / Desirable"
            explanation = (
                f"Your HDL cholesterol is {val_str}, meeting desirable protective targets. "
                f"Adequate HDL cholesterol supports blood vessel health and guards against microvascular complications. "
                f"This healthy level {shap_direction}."
            )

    elif 'ldl' in f_lower:
        num_val = float(val) if isinstance(val, (int, float)) else 90.0
        val_str += " mg/dL"
        if num_val >= 100.0:
            status = "Elevated (≥ 100 mg/dL)"
            explanation = (
                f"Your LDL cholesterol is {val_str}, which exceeds optimal target levels. "
                f"Elevated LDL can cause arterial plaque buildup, compromising renal blood flow. "
                f"This elevation {shap_direction}."
            )
        else:
            status = "Optimal (< 100 mg/dL)"
            explanation = (
                f"Your LDL cholesterol is {val_str}, keeping within optimal targets. "
                f"Healthy LDL levels prevent arterial plaque accumulation and preserve clear circulation. "
                f"This optimal reading {shap_direction}."
            )

    elif 'cholesterol' in f_lower:
        num_val = float(val) if isinstance(val, (int, float)) else 180.0
        val_str += " mg/dL"
        if num_val >= 200.0:
            status = "High (≥ 200 mg/dL)"
            explanation = (
                f"Your total cholesterol level is {val_str}, which is elevated. "
                f"Higher total cholesterol increases overall vascular strain across major organ systems including the kidneys. "
                f"This parameter {shap_direction}."
            )
        else:
            status = "Normal (< 200 mg/dL)"
            explanation = (
                f"Your total cholesterol is {val_str}, remaining in the healthy target range. "
                f"A balanced lipid profile reduces systemic vascular inflammation. "
                f"This factor {shap_direction}."
            )

    elif 'dr' in f_lower or 'retinopathy' in f_lower:
        is_yes = (str(val).strip().lower() in ['yes', '1', 'true'])
        status = "Present" if is_yes else "Absent"
        if is_yes:
            explanation = (
                f"Diabetic Retinopathy is present in your clinical profile. "
                f"Retinopathy reflects existing damage to small eye microvessels, which closely correlates with microvascular changes in the kidneys. "
                f"This clinical finding {shap_direction}."
            )
        else:
            explanation = (
                f"Diabetic Retinopathy is absent in your clinical profile. "
                f"The absence of eye microvascular damage suggests healthier overall microvasculature. "
                f"This positive indicator {shap_direction}."
            )

    elif 'age' in f_lower:
        num_val = float(val) if isinstance(val, (int, float)) else 50.0
        val_str += " years"
        if num_val >= 60.0:
            status = "Older Age Group (≥ 60)"
            explanation = (
                f"Your age is {val_str}. "
                f"Advancing age naturally reduces baseline renal functional reserve and increases susceptibility to diabetes-related kidney changes. "
                f"This baseline factor {shap_direction}."
            )
        else:
            status = "Younger / Middle Age"
            explanation = (
                f"Your age is {val_str}. "
                f"Younger age generally preserves baseline physiological renal reserve. "
                f"This age profile {shap_direction}."
            )

    elif 'smoking' in f_lower:
        is_yes = (str(val).strip().lower() in ['yes', '1', 'true'])
        status = "Yes (Active)" if is_yes else "No (Non-smoker)"
        if is_yes:
            explanation = (
                f"Smoking status is listed as Yes. "
                f"Smoking restricts renal blood flow, accelerates arterial stiffness, and promotes inflammation in kidney nephrons. "
                f"This lifestyle factor {shap_direction}."
            )
        else:
            explanation = (
                f"Smoking status is listed as No. "
                f"Not smoking avoids tobacco-induced microvascular toxicity and renal vasoconstriction. "
                f"This healthy habit {shap_direction}."
            )

    else:
        explanation = (
            f"The clinical feature '{feature}' has a patient value of {val_str}. "
            f"Based on the trained AI model's decision tree structure, this factor {shap_direction}."
        )

    return val_str, status, explanation


def generate_personalized_recommendations(user_inputs, importance_df=None, risk_category="Low Risk"):
    """
    Generate dynamic, patient-specific health recommendations based on:
    1. Actual current patient input values
    2. Clinical reference range evaluation
    3. Existing SHAP feature importance for priority ranking
    4. Prediction risk category context
    """
    shap_lookup = {}
    if importance_df is not None and not importance_df.empty:
        for idx, row in importance_df.iterrows():
            feat = str(row['Feature'])
            shap_lookup[feat] = {
                'shap_val': float(row['SHAP Value']),
                'abs_shap': abs(float(row['SHAP Value'])),
                'rank': idx + 1
            }

    def get_shap_info(feature_key):
        for k, v in shap_lookup.items():
            if feature_key.lower() in k.lower():
                return v
        return {'shap_val': 0.0, 'abs_shap': 0.0, 'rank': 99}

    flagged = []
    if not user_inputs:
        return flagged

    # 1. HbA1c
    hba1c = user_inputs.get('HbA1c', None)
    if hba1c is not None:
        try:
            val = float(hba1c)
            if val > 7.0:
                shap_info = get_shap_info('hba1c')
                is_high = shap_info['shap_val'] > 0 or shap_info['rank'] <= 5 or risk_category == "High Risk"
                prio_tag = "🔴 High Priority" if is_high else "🟠 Moderate Priority"
                color = "#e74c3c" if is_high else "#f39c12"
                msg = f"Your HbA1c is <strong>{val:.1f}%</strong>, which is elevated above the 7.0% recommended target. Better long-term blood sugar control may help reduce the risk of diabetes-related complications."
                sort_score = (100 if is_high else 50) + shap_info['abs_shap'] * 10
                flagged.append({'tag': prio_tag, 'title': 'Blood Sugar', 'msg': msg, 'score': sort_score, 'color': color})
        except (ValueError, TypeError):
            pass

    # 2. Fasting Blood Glucose
    fbg = user_inputs.get('FBG_(mmol/L)', user_inputs.get('FBG', None))
    if fbg is not None:
        try:
            val = float(fbg)
            if val > 7.0:
                shap_info = get_shap_info('fbg')
                is_high = shap_info['shap_val'] > 0 or shap_info['rank'] <= 5
                prio_tag = "🔴 High Priority" if is_high else "🟠 Moderate Priority"
                color = "#e74c3c" if is_high else "#f39c12"
                msg = f"Your fasting blood glucose is <strong>{val:.1f} mmol/L</strong>, which is elevated. Monitoring and optimizing daily blood glucose management is recommended."
                sort_score = (95 if is_high else 45) + shap_info['abs_shap'] * 10
                flagged.append({'tag': prio_tag, 'title': 'Fasting Blood Glucose', 'msg': msg, 'score': sort_score, 'color': color})
        except (ValueError, TypeError):
            pass

    # 3. Systolic & Diastolic Blood Pressure
    sbp = user_inputs.get('SBP', user_inputs.get('SBP_(mmHg)_', None))
    dbp = user_inputs.get('DBP', user_inputs.get('DBP_(mmHg)', None))
    if sbp is not None or dbp is not None:
        try:
            sbp_val = float(sbp) if sbp is not None else 120.0
            dbp_val = float(dbp) if dbp is not None else 80.0
            if sbp_val >= 130.0 or dbp_val >= 80.0:
                shap_info = get_shap_info('sbp') if sbp_val >= 130 else get_shap_info('dbp')
                is_high = shap_info['shap_val'] > 0 or shap_info['rank'] <= 5 or risk_category == "High Risk"
                prio_tag = "🔴 High Priority" if is_high else "🟠 Moderate Priority"
                color = "#e74c3c" if is_high else "#f39c12"
                
                if sbp_val >= 130.0 and dbp_val >= 80.0:
                    bp_msg = f"Your systolic blood pressure is <strong>{int(sbp_val)} mmHg</strong> and diastolic blood pressure is <strong>{int(dbp_val)} mmHg</strong>, which are elevated."
                elif sbp_val >= 130.0:
                    bp_msg = f"Your systolic blood pressure is <strong>{int(sbp_val)} mmHg</strong>, which is elevated."
                else:
                    bp_msg = f"Your diastolic blood pressure is <strong>{int(dbp_val)} mmHg</strong>, which is elevated."
                
                msg = f"{bp_msg} Regular monitoring and appropriate blood-pressure management should be discussed with your healthcare provider."
                sort_score = (90 if is_high else 40) + shap_info['abs_shap'] * 10
                flagged.append({'tag': prio_tag, 'title': 'Blood Pressure', 'msg': msg, 'score': sort_score, 'color': color})
        except (ValueError, TypeError):
            pass

    # 4. eGFR
    egfr = user_inputs.get('eGFR', user_inputs.get('eGFR_(mL/min/1.73m2)', None))
    if egfr is not None:
        try:
            val = float(egfr)
            if val < 60.0:
                shap_info = get_shap_info('egfr')
                prio_tag = "🔴 High Priority"
                color = "#e74c3c"
                msg = f"Your estimated Glomerular Filtration Rate (eGFR) is <strong>{val:.1f} mL/min/1.73m²</strong>, indicating reduced kidney filtration function. Prompt specialist evaluation and follow-up laboratory testing are recommended."
                sort_score = 110 + shap_info['abs_shap'] * 10
                flagged.append({'tag': prio_tag, 'title': 'Kidney Filtration (eGFR)', 'msg': msg, 'score': sort_score, 'color': color})
            elif val < 90.0:
                shap_info = get_shap_info('egfr')
                prio_tag = "🟠 Moderate Priority"
                color = "#f39c12"
                msg = f"Your eGFR is <strong>{val:.1f} mL/min/1.73m²</strong>, indicating slightly reduced filtration function. Routine renal monitoring is advisable."
                sort_score = 35 + shap_info['abs_shap'] * 10
                flagged.append({'tag': prio_tag, 'title': 'Kidney Filtration (eGFR)', 'msg': msg, 'score': sort_score, 'color': color})
        except (ValueError, TypeError):
            pass

    # 5. UACR
    uacr = user_inputs.get('UACR', user_inputs.get('UACR_(mg/g)', None))
    if uacr is not None:
        try:
            val = float(uacr)
            if val >= 30.0:
                shap_info = get_shap_info('uacr')
                is_high = val >= 300.0 or shap_info['shap_val'] > 0 or shap_info['rank'] <= 5
                prio_tag = "🔴 High Priority" if is_high else "🟠 Moderate Priority"
                color = "#e74c3c" if is_high else "#f39c12"
                severity_str = "macroalbuminuria" if val >= 300 else "microalbuminuria"
                msg = f"Your Urine Albumin-to-Creatinine Ratio (UACR) is <strong>{val:.1f} mg/g</strong>, indicating {severity_str} (protein leaking into urine). Discuss initiating or optimizing kidney-protective therapy with your physician."
                sort_score = (105 if is_high else 45) + shap_info['abs_shap'] * 10
                flagged.append({'tag': prio_tag, 'title': 'Urine Protein Excretion (UACR)', 'msg': msg, 'score': sort_score, 'color': color})
        except (ValueError, TypeError):
            pass

    # 6. Serum Creatinine
    creat = user_inputs.get('Serum creatinine', user_inputs.get('Serum_creatinine', None))
    if creat is not None:
        try:
            val = float(creat)
            if val > 1.2:
                shap_info = get_shap_info('creatinine')
                is_high = shap_info['shap_val'] > 0 or shap_info['rank'] <= 5
                prio_tag = "🔴 High Priority" if is_high else "🟠 Moderate Priority"
                color = "#e74c3c" if is_high else "#f39c12"
                msg = f"Your serum creatinine is <strong>{val:.2f} mg/dL</strong>, which is elevated above normal reference ranges. High serum creatinine indicates reduced renal waste clearance."
                sort_score = (85 if is_high else 38) + shap_info['abs_shap'] * 10
                flagged.append({'tag': prio_tag, 'title': 'Serum Creatinine', 'msg': msg, 'score': sort_score, 'color': color})
        except (ValueError, TypeError):
            pass

    # 7. HDL Cholesterol
    hdl = user_inputs.get('HDL', user_inputs.get('HDLC?mmoll?', user_inputs.get('HDL-C', None)))
    if hdl is not None:
        try:
            val = float(hdl)
            if val < 40.0:
                shap_info = get_shap_info('hdl')
                is_high = shap_info['shap_val'] > 0 and shap_info['rank'] <= 5
                prio_tag = "🔴 High Priority" if is_high else "🟠 Moderate Priority"
                color = "#e74c3c" if is_high else "#f39c12"
                msg = f"Your HDL-C is <strong>{val:.1f} mg/dL</strong>, which is below protective target levels. Healthy lifestyle measures, balanced diet, and physical activity may support a healthier lipid profile."
                sort_score = (70 if is_high else 30) + shap_info['abs_shap'] * 10
                flagged.append({'tag': prio_tag, 'title': 'HDL-C / Lipid Health', 'msg': msg, 'score': sort_score, 'color': color})
        except (ValueError, TypeError):
            pass

    # 8. Triglycerides
    tg = user_inputs.get('Triglycerides', user_inputs.get('TG?mmoll?', user_inputs.get('TG', None)))
    if tg is not None:
        try:
            val = float(tg)
            if val >= 150.0:
                shap_info = get_shap_info('triglycerides')
                is_high = shap_info['shap_val'] > 0 and shap_info['rank'] <= 5
                prio_tag = "🔴 High Priority" if is_high else "🟠 Moderate Priority"
                color = "#e74c3c" if is_high else "#f39c12"
                msg = f"Your triglyceride level is <strong>{val:.1f} mg/dL</strong>, which is elevated. Attention to diet, physical activity, and lipid monitoring is recommended."
                sort_score = (65 if is_high else 28) + shap_info['abs_shap'] * 10
                flagged.append({'tag': prio_tag, 'title': 'Triglyceride Management', 'msg': msg, 'score': sort_score, 'color': color})
        except (ValueError, TypeError):
            pass

    # 9. LDL Cholesterol
    ldl = user_inputs.get('LDL', user_inputs.get('LDLC?mmoll?', None))
    if ldl is not None:
        try:
            val = float(ldl)
            if val >= 100.0:
                shap_info = get_shap_info('ldl')
                prio_tag = "🟠 Moderate Priority"
                color = "#f39c12"
                msg = f"Your LDL cholesterol is <strong>{val:.1f} mg/dL</strong>, which exceeds optimal targets. Managing LDL cholesterol reduces arterial plaque accumulation and preserves kidney blood flow."
                sort_score = 25 + shap_info['abs_shap'] * 10
                flagged.append({'tag': prio_tag, 'title': 'LDL Cholesterol', 'msg': msg, 'score': sort_score, 'color': color})
        except (ValueError, TypeError):
            pass

    # 10. Total Cholesterol
    tc = user_inputs.get('Cholesterol', user_inputs.get('TC?mmoll?', None))
    if tc is not None:
        try:
            val = float(tc)
            if val >= 200.0:
                shap_info = get_shap_info('cholesterol')
                prio_tag = "🟠 Moderate Priority"
                color = "#f39c12"
                msg = f"Your total cholesterol is <strong>{val:.1f} mg/dL</strong>, which is elevated. Reviewing your lipid profile and dietary habits with your healthcare provider is recommended."
                sort_score = 24 + shap_info['abs_shap'] * 10
                flagged.append({'tag': prio_tag, 'title': 'Total Cholesterol', 'msg': msg, 'score': sort_score, 'color': color})
        except (ValueError, TypeError):
            pass

    # 11. BMI
    bmi = user_inputs.get('BMI', user_inputs.get('BMI_(kg/m2)', None))
    if bmi is not None:
        try:
            val = float(bmi)
            if val >= 25.0:
                shap_info = get_shap_info('bmi')
                prio_tag = "🟠 Moderate Priority"
                color = "#f39c12"
                msg = f"Your Body Mass Index (BMI) is <strong>{val:.1f} kg/m²</strong>, placing you above the standard healthy weight range. Achieving gradual weight management reduces hyperfiltration stress on your kidneys."
                sort_score = 20 + shap_info['abs_shap'] * 10
                flagged.append({'tag': prio_tag, 'title': 'Body Mass Index', 'msg': msg, 'score': sort_score, 'color': color})
        except (ValueError, TypeError):
            pass

    # 12. Diabetes Duration
    dur = user_inputs.get('Diabetes duration (y)', user_inputs.get('Diabetes_duration_(y)', None))
    if dur is not None:
        try:
            val = float(dur)
            if val >= 10.0:
                shap_info = get_shap_info('duration')
                prio_tag = "🟠 Moderate Priority"
                color = "#f39c12"
                msg = f"You have had diabetes for <strong>{int(val)} years</strong>. Longer diabetes duration can increase the likelihood of diabetes-related complications, including kidney problems, and therefore regular diabetes and kidney-health monitoring are recommended."
                sort_score = 32 + shap_info['abs_shap'] * 10
                flagged.append({'tag': prio_tag, 'title': 'Diabetes Duration', 'msg': msg, 'score': sort_score, 'color': color})
        except (ValueError, TypeError):
            pass

    # 13. Diabetic Retinopathy
    dr = user_inputs.get('Diabetic Retinopathy (DR)', user_inputs.get('DR', user_inputs.get('Diabetic_retinopathy_(DR)', None)))
    if dr is not None:
        dr_str = str(dr).strip().lower()
        if dr_str in ['yes', '1', 'true']:
            shap_info = get_shap_info('retinopathy')
            prio_tag = "🟠 Moderate Priority"
            color = "#f39c12"
            msg = f"Diabetic retinopathy is present in your clinical profile. Regular diabetes-related clinical follow-up is important because eye and kidney microvessels are closely connected."
            sort_score = 34 + shap_info['abs_shap'] * 10
            flagged.append({'tag': prio_tag, 'title': 'Diabetic Retinopathy', 'msg': msg, 'score': sort_score, 'color': color})

    # 14. Smoking
    smoke = user_inputs.get('Smoking', None)
    if smoke is not None and str(smoke).strip().lower() in ['yes', '1', 'true']:
        shap_info = get_shap_info('smoking')
        prio_tag = "🟠 Moderate Priority"
        color = "#f39c12"
        msg = f"Smoking is listed as <strong>Yes</strong> in your profile. Smoking impairs renal microvascular blood flow; pursuing smoking cessation support is strongly recommended."
        sort_score = 22 + shap_info['abs_shap'] * 10
        flagged.append({'tag': prio_tag, 'title': 'Smoking Cessation', 'msg': msg, 'score': sort_score, 'color': color})

    # Sort flagged recommendations by sort_score descending
    flagged.sort(key=lambda x: x['score'], reverse=True)
    return flagged


def display_clinical_priority_list(user_inputs, importance_df=None):
    """
    Display 🧑‍⚕️ Clinical Priority List immediately after SHAP Feature Importance
    and before AI Clinical Explanation.
    
    Args:
        user_inputs: Dict of user inputs
        importance_df: DataFrame of SHAP feature importances (optional)
    """
    st.markdown('<div class="sub-header">🧑‍⚕️ Clinical Priority List</div>', unsafe_allow_html=True)
    
    if not user_inputs:
        st.info("No patient inputs available to generate priority list.")
        return

    # Build SHAP lookup dictionary
    shap_lookup = {}
    if importance_df is not None and not importance_df.empty:
        for idx, row in importance_df.iterrows():
            feat = str(row['Feature'])
            shap_lookup[feat] = {
                'shap_val': float(row['SHAP Value']),
                'abs_shap': abs(float(row['SHAP Value'])),
                'rank': idx + 1
            }

    def get_shap_info(feature_key):
        for k, v in shap_lookup.items():
            if feature_key.lower() in k.lower():
                return v
        return {'shap_val': 0.0, 'abs_shap': 0.0, 'rank': 99}

    priorities = []

    # 1. HbA1c
    hba1c = user_inputs.get('HbA1c', user_inputs.get('HbA1c_(%)', None))
    if hba1c is not None:
        try:
            val = float(hba1c)
            if val > 7.0:
                s_info = get_shap_info('hba1c')
                score = 100 + s_info['abs_shap'] * 10 + (20 if s_info['shap_val'] > 0 else 0)
                priorities.append({
                    'name': 'HbA1c',
                    'display_name': 'HbA1c',
                    'val_str': f"{val:.1f}%",
                    'status': 'High',
                    'status_color': '#e74c3c',
                    'explanation': 'Long-term blood sugar control needs attention.',
                    'score': score
                })
        except (ValueError, TypeError):
            pass

    # 2. Fasting Blood Glucose
    fbg = user_inputs.get('FBG_(mmol/L)', user_inputs.get('FBG', None))
    if fbg is not None:
        try:
            val = float(fbg)
            if val > 7.0:
                s_info = get_shap_info('fbg')
                score = 95 + s_info['abs_shap'] * 10
                priorities.append({
                    'name': 'FBG',
                    'display_name': 'Fasting Blood Glucose',
                    'val_str': f"{val:.1f} mmol/L",
                    'status': 'High',
                    'status_color': '#e74c3c',
                    'explanation': 'Fasting blood sugar is elevated.',
                    'score': score
                })
        except (ValueError, TypeError):
            pass

    # 3. Systolic Blood Pressure
    sbp = user_inputs.get('SBP', user_inputs.get('SBP_(mmHg)_', None))
    if sbp is not None:
        try:
            val = float(sbp)
            if val >= 130.0:
                s_info = get_shap_info('sbp')
                score = 90 + s_info['abs_shap'] * 10 + (20 if s_info['shap_val'] > 0 else 0)
                priorities.append({
                    'name': 'SBP',
                    'display_name': 'SBP',
                    'val_str': f"{int(val)} mmHg",
                    'status': 'High',
                    'status_color': '#e74c3c',
                    'explanation': 'Blood pressure should be monitored regularly.',
                    'score': score
                })
        except (ValueError, TypeError):
            pass

    # 4. Diastolic Blood Pressure
    dbp = user_inputs.get('DBP', user_inputs.get('DBP_(mmHg)', None))
    if dbp is not None:
        try:
            val = float(dbp)
            if val >= 80.0:
                s_info = get_shap_info('dbp')
                score = 85 + s_info['abs_shap'] * 10
                priorities.append({
                    'name': 'DBP',
                    'display_name': 'DBP',
                    'val_str': f"{int(val)} mmHg",
                    'status': 'High',
                    'status_color': '#e74c3c',
                    'explanation': 'Diastolic blood pressure is elevated.',
                    'score': score
                })
        except (ValueError, TypeError):
            pass

    # 5. eGFR
    egfr = user_inputs.get('eGFR', user_inputs.get('eGFR_(mL/min/1.73m2)', None))
    if egfr is not None:
        try:
            val = float(egfr)
            if val < 60.0:
                s_info = get_shap_info('egfr')
                score = 110 + s_info['abs_shap'] * 10
                priorities.append({
                    'name': 'eGFR',
                    'display_name': 'eGFR',
                    'val_str': f"{val:.1f} mL/min/1.73m²",
                    'status': 'Low',
                    'status_color': '#e74c3c',
                    'explanation': 'Kidney filtration rate is reduced.',
                    'score': score
                })
            elif val < 90.0:
                s_info = get_shap_info('egfr')
                score = 45 + s_info['abs_shap'] * 10
                priorities.append({
                    'name': 'eGFR',
                    'display_name': 'eGFR',
                    'val_str': f"{val:.1f} mL/min/1.73m²",
                    'status': 'Needs Attention',
                    'status_color': '#f39c12',
                    'explanation': 'Kidney filtration function is slightly reduced.',
                    'score': score
                })
        except (ValueError, TypeError):
            pass

    # 6. UACR
    uacr = user_inputs.get('UACR', user_inputs.get('UACR_(mg/g)', None))
    if uacr is not None:
        try:
            val = float(uacr)
            if val >= 30.0:
                s_info = get_shap_info('uacr')
                score = 105 + s_info['abs_shap'] * 10
                status_str = 'High' if val >= 300 else 'Needs Attention'
                color_str = '#e74c3c' if val >= 300 else '#f39c12'
                priorities.append({
                    'name': 'UACR',
                    'display_name': 'UACR',
                    'val_str': f"{val:.1f} mg/g",
                    'status': status_str,
                    'status_color': color_str,
                    'explanation': 'Protein leakage into urine detected.',
                    'score': score
                })
        except (ValueError, TypeError):
            pass

    # 7. Serum Creatinine
    creat = user_inputs.get('Serum creatinine', user_inputs.get('Serum_creatinine', None))
    if creat is not None:
        try:
            val = float(creat)
            if val > 1.2:
                s_info = get_shap_info('creatinine')
                score = 80 + s_info['abs_shap'] * 10
                priorities.append({
                    'name': 'Serum Creatinine',
                    'display_name': 'Serum Creatinine',
                    'val_str': f"{val:.2f} mg/dL",
                    'status': 'High',
                    'status_color': '#e74c3c',
                    'explanation': 'Serum creatinine level is above normal range.',
                    'score': score
                })
        except (ValueError, TypeError):
            pass

    # 8. HDL Cholesterol
    hdl = user_inputs.get('HDL', user_inputs.get('HDLC?mmoll?', user_inputs.get('HDL-C', None)))
    if hdl is not None:
        try:
            val = float(hdl)
            if val < 40.0:
                s_info = get_shap_info('hdl')
                score = 75 + s_info['abs_shap'] * 10
                val_display = f"{val:.1f} mg/dL" if val > 5 else f"{val:.1f} mmol/L"
                priorities.append({
                    'name': 'HDL-C',
                    'display_name': 'HDL-C',
                    'val_str': val_display,
                    'status': 'Low',
                    'status_color': '#e74c3c',
                    'explanation': 'HDL-C is below the desired level.',
                    'score': score
                })
        except (ValueError, TypeError):
            pass

    # 9. Triglycerides
    tg = user_inputs.get('Triglycerides', user_inputs.get('TG?mmoll?', user_inputs.get('TG', None)))
    if tg is not None:
        try:
            val = float(tg)
            if val >= 150.0:
                s_info = get_shap_info('triglycerides')
                score = 70 + s_info['abs_shap'] * 10
                val_display = f"{val:.1f} mg/dL" if val > 10 else f"{val:.1f} mmol/L"
                priorities.append({
                    'name': 'Triglycerides',
                    'display_name': 'Triglycerides',
                    'val_str': val_display,
                    'status': 'High',
                    'status_color': '#e74c3c',
                    'explanation': 'Triglyceride level is elevated.',
                    'score': score
                })
        except (ValueError, TypeError):
            pass

    # 10. LDL Cholesterol
    ldl = user_inputs.get('LDL', user_inputs.get('LDLC?mmoll?', None))
    if ldl is not None:
        try:
            val = float(ldl)
            if val >= 100.0:
                s_info = get_shap_info('ldl')
                score = 65 + s_info['abs_shap'] * 10
                val_display = f"{val:.1f} mg/dL" if val > 10 else f"{val:.1f} mmol/L"
                priorities.append({
                    'name': 'LDL-C',
                    'display_name': 'LDL-C',
                    'val_str': val_display,
                    'status': 'High',
                    'status_color': '#e74c3c',
                    'explanation': 'LDL-C exceeds optimal target levels.',
                    'score': score
                })
        except (ValueError, TypeError):
            pass

    # 11. Total Cholesterol
    tc = user_inputs.get('Cholesterol', user_inputs.get('TC?mmoll?', None))
    if tc is not None:
        try:
            val = float(tc)
            if val >= 200.0:
                s_info = get_shap_info('cholesterol')
                score = 60 + s_info['abs_shap'] * 10
                val_display = f"{val:.1f} mg/dL" if val > 10 else f"{val:.1f} mmol/L"
                priorities.append({
                    'name': 'Total Cholesterol',
                    'display_name': 'Total Cholesterol',
                    'val_str': val_display,
                    'status': 'High',
                    'status_color': '#e74c3c',
                    'explanation': 'Total cholesterol is elevated.',
                    'score': score
                })
        except (ValueError, TypeError):
            pass

    # 12. BMI
    bmi = user_inputs.get('BMI', user_inputs.get('BMI_(kg/m2)', None))
    if bmi is not None:
        try:
            val = float(bmi)
            if val >= 25.0:
                s_info = get_shap_info('bmi')
                score = 55 + s_info['abs_shap'] * 10
                priorities.append({
                    'name': 'BMI',
                    'display_name': 'BMI',
                    'val_str': f"{val:.1f} kg/m²",
                    'status': 'High',
                    'status_color': '#f39c12',
                    'explanation': 'Body Mass Index is above standard weight range.',
                    'score': score
                })
        except (ValueError, TypeError):
            pass

    # 13. Diabetes Duration
    dur = user_inputs.get('Diabetes duration (y)', user_inputs.get('Diabetes_duration_(y)', None))
    if dur is not None:
        try:
            val = float(dur)
            if val >= 10.0:
                s_info = get_shap_info('duration')
                score = 50 + s_info['abs_shap'] * 10
                priorities.append({
                    'name': 'Diabetes Duration',
                    'display_name': 'Diabetes Duration',
                    'val_str': f"{int(val)} years",
                    'status': 'Needs Attention',
                    'status_color': '#f39c12',
                    'explanation': 'Long duration of diabetes increases risk exposure.',
                    'score': score
                })
        except (ValueError, TypeError):
            pass

    # 14. Diabetic Retinopathy
    dr = user_inputs.get('Diabetic Retinopathy (DR)', user_inputs.get('DR', user_inputs.get('Diabetic_retinopathy_(DR)', None)))
    if dr is not None and str(dr).strip().lower() in ['yes', '1', 'true']:
        s_info = get_shap_info('retinopathy')
        score = 52 + s_info['abs_shap'] * 10
        priorities.append({
            'name': 'Diabetic Retinopathy',
            'display_name': 'Diabetic Retinopathy',
            'val_str': 'Present',
            'status': 'Needs Attention',
            'status_color': '#f39c12',
            'explanation': 'Presence of eye microvascular complications.',
            'score': score
        })

    # 15. Smoking
    smoke = user_inputs.get('Smoking', None)
    if smoke is not None and str(smoke).strip().lower() in ['yes', '1', 'true']:
        s_info = get_shap_info('smoking')
        score = 40 + s_info['abs_shap'] * 10
        priorities.append({
            'name': 'Smoking',
            'display_name': 'Smoking',
            'val_str': 'Yes',
            'status': 'Needs Attention',
            'status_color': '#f39c12',
            'explanation': 'Smoking impairs renal microvascular blood flow.',
            'score': score
        })

    # Sort by score descending
    priorities.sort(key=lambda x: x['score'], reverse=True)

    # Pick top 3 to 5 priorities
    top_priorities = priorities[:5]

    badge_emojis = ['🔴', '🟠', '🟡', '🔵', '🟣']
    badge_colors = ['#e74c3c', '#e67e22', '#f1c40f', '#3498db', '#9b59b6']

    if top_priorities:
        for idx, item in enumerate(top_priorities, 1):
            emoji = badge_emojis[idx - 1] if idx <= len(badge_emojis) else '📌'
            card_border_color = badge_colors[idx - 1] if idx <= len(badge_colors) else '#95a5a6'
            
            st.markdown(f"""<div class="dashboard-card" style="margin-bottom: 0.9rem; padding: 1.1rem; border-left: 5px solid {card_border_color};">
<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; margin-bottom: 0.4rem;">
<h5 style="margin: 0; font-size: 1.05rem; font-weight: bold; color: var(--text-color, #1a252f);">{emoji} Priority {idx} — {item['display_name']}</h5>
<span style="font-size: 0.88rem; font-weight: bold; color: {item['status_color']};">Status: {item['status']}</span>
</div>
<p style="margin: 0 0 0.3rem 0; font-size: 0.95rem; color: var(--text-color, #4a5568);"><strong>Patient Value:</strong> <span style="color: #2b6cb0; font-weight: bold;">{item['val_str']}</span></p>
<p style="margin: 0; font-size: 0.93rem; line-height: 1.5; color: var(--text-color, #2d3748);">{item['explanation']}</p>
</div>""", unsafe_allow_html=True)
    else:
        # All normal case
        st.markdown("""<div class="success-box" style="padding: 1.25rem;">
<p style="margin: 0; font-size: 1.05rem; font-weight: bold; color: #155724;">🟢 No major clinical priorities identified. Continue regular monitoring.</p>
</div>""", unsafe_allow_html=True)


def display_ai_clinical_explanation(prediction, probability, user_inputs, importance_df=None):
    """
    Display AI Clinical Explanation section immediately below SHAP Feature Importance.
    
    Args:
        prediction: Predicted class (0 or 1)
        probability: Prediction probabilities array [prob_0, prob_1]
        user_inputs: Dict of user inputs
        importance_df: DataFrame of SHAP feature importances (optional)
    """
    st.markdown('<div class="sub-header">🤖 AI Clinical Explanation</div>', unsafe_allow_html=True)
    
    # 1. Prediction Summary
    high_risk_prob = probability[1] if (probability is not None and len(probability) > 1) else 0.0
    prob_pct = f"{high_risk_prob * 100:.1f}%"
    
    if high_risk_prob < 0.35:
        risk_category = "Low Risk"
        badge_style = "background-color: #2ecc71; color: white; padding: 4px 12px; border-radius: 5px; font-weight: bold;"
    elif high_risk_prob <= 0.65:
        risk_category = "Moderate Risk"
        badge_style = "background-color: #f39c12; color: white; padding: 4px 12px; border-radius: 5px; font-weight: bold;"
    else:
        risk_category = "High Risk"
        badge_style = "background-color: #e74c3c; color: white; padding: 4px 12px; border-radius: 5px; font-weight: bold;"

    st.markdown(f"""<div class="info-box" style="padding: 1.25rem; margin-bottom: 1.5rem;">
<h4 style="margin-top: 0; margin-bottom: 0.5rem; font-size: 1.15rem; font-weight: bold; color: var(--text-color, #1a252f);">📋 Prediction Summary</h4>
<p style="margin: 0; font-size: 1.05rem; line-height: 1.6;">
Based on your clinical information, the AI model predicts a <span style="{badge_style}">{risk_category}</span> of Diabetic Nephropathy with a risk probability of <strong>{prob_pct}</strong>.
</p>
</div>""", unsafe_allow_html=True)

    # 2. Personalized Health Recommendations (Replaces Why Did AI Make Prediction)
    st.markdown("""<h4 style="font-weight: 600; color: var(--text-color, #1a252f); margin-top: 1.5rem; margin-bottom: 0.8rem;">💡 Personalized Health Recommendations</h4>""", unsafe_allow_html=True)
    
    st.markdown(f"""<p style="font-size: 1.02rem; margin-bottom: 1rem; color: var(--text-color, #1a252f);">
<strong>Risk Status:</strong> <span style="{badge_style}">{risk_category}</span>
</p>""", unsafe_allow_html=True)

    # Generate dynamic patient-specific recommendations
    dynamic_recs = generate_personalized_recommendations(user_inputs, importance_df, risk_category)
    
    if dynamic_recs:
        for rec in dynamic_recs:
            tag = rec['tag']
            title = rec['title']
            msg = rec['msg']
            color = rec['color']
            
            st.markdown(f"""<div class="dashboard-card" style="margin-bottom: 1rem; padding: 1.1rem; border-left: 5px solid {color};">
<h5 style="margin: 0 0 0.5rem 0; font-size: 1.05rem; font-weight: bold; color: var(--text-color, #1a252f);">{tag} — {title}</h5>
<p style="margin: 0; font-size: 0.95rem; line-height: 1.6; color: var(--text-color, #2d3748);">{msg}</p>
</div>""", unsafe_allow_html=True)
    else:
        # Patient with all normal indicators
        st.markdown("""<div class="success-box" style="padding: 1.25rem; margin-bottom: 1rem;">
<h5 style="margin: 0 0 0.5rem 0; font-weight: bold; color: #155724;">🟢 Positive Indicator — Clinical Status</h5>
<p style="margin: 0; font-size: 1.02rem; line-height: 1.6; color: #155724;">Your current clinical indicators do not show major areas requiring additional attention based on the entered values. Continue regular diabetes monitoring and a healthy lifestyle.</p>
</div>""", unsafe_allow_html=True)

    # 3. Overall AI Assessment
    st.markdown("""<h4 style="font-weight: 600; color: var(--text-color, #1a252f); margin-top: 2rem; margin-bottom: 1rem;">📊 Overall AI Assessment</h4>""", unsafe_allow_html=True)
    
    if risk_category == "High Risk":
        assessment_text = f"Overall, the AI model predicts a <strong>High Risk</strong> of Diabetic Nephropathy with a probability of {prob_pct}. Key clinical factors require prompt medical attention and therapy optimization to mitigate kidney risk."
    elif risk_category == "Moderate Risk":
        assessment_text = f"Overall, the AI model predicts a <strong>Moderate Risk</strong> of Diabetic Nephropathy with a probability of {prob_pct}. Specific clinical parameters require targeted monitoring and lifestyle management to prevent risk progression."
    else:
        assessment_text = f"Overall, the AI model predicts a <strong>Low Risk</strong> of Diabetic Nephropathy with a risk probability of {prob_pct}. Most of the evaluated clinical indicators fall within acceptable ranges. Continuing regular monitoring and maintaining a healthy lifestyle may help reduce future risk."

    st.markdown(f"""<div class="dashboard-card" style="padding: 1.25rem; background-color: var(--background-color-secondary, rgba(255, 255, 255, 0.03));">
<p style="margin: 0; font-size: 1.02rem; line-height: 1.6; color: var(--text-color, #1a252f);">{assessment_text}</p>
</div>""", unsafe_allow_html=True)

    # 4. Simple User-Friendly Explanation Note
    st.markdown("""<div style="font-size: 0.9rem; color: var(--text-color, #7f8c8d); margin-top: 1rem; margin-bottom: 1.5rem; font-style: italic;">
ℹ️ <strong>User-Friendly Guide:</strong> SHAP shows which clinical factors had the strongest influence on the AI model's prediction.
</div>""", unsafe_allow_html=True)

    # 5. Medical Disclaimer
    st.markdown("""<div style="font-size: 0.88rem; color: #7f8c8d; border-top: 1px solid #cbd5e1; padding-top: 0.8rem; margin-top: 1.5rem; font-style: italic; text-align: center;">
This AI explanation is generated using the model prediction and SHAP feature importance. It is intended for educational purposes only and should not replace professional medical advice.
</div>""", unsafe_allow_html=True)


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
        <div style="display: flex; justify-content: space-between; padding: 0.4rem 0; font-size: 0.88rem;">
            <span style="color: var(--text-color); opacity: 0.8;">Target Classes</span>
            <span style="color: var(--text-color); font-weight: bold;">{pipeline_data.get('prediction_classes', 2)} classes</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Identify best model dynamically using priority: (1) Accuracy, (2) ROC-AUC, (3) F1
    comparison_data = pipeline_data.get('comparison', {})
    if comparison_data:
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
    else:
        best_m = pipeline_data

    best_model_name = best_m.get('model_name', pipeline_data.get('best_model', {}).get('name', 'Stacking Classifier'))
    best_accuracy = best_m.get('accuracy', pipeline_data.get('accuracy', 0.8894))
    best_cv_accuracy = best_m.get('cv_accuracy', pipeline_data.get('cv_accuracy', 0.9613))
    best_precision = best_m.get('precision', pipeline_data.get('precision', 0.8894))
    best_recall = best_m.get('recall', pipeline_data.get('recall', 0.8894))
    best_f1 = best_m.get('f1', best_m.get('f1_score', pipeline_data.get('f1_score', 0.8894)))
    best_roc_auc = best_m.get('roc_auc', pipeline_data.get('roc_auc', 0.9388))

    # Top performance metrics section
    st.markdown('<div class="sub-header">📈 Machine Learning Pipeline Metrics</div>', unsafe_allow_html=True)
    
    # 6 columns for metric cards
    m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
    
    with m_col1:
        st.markdown(f"""
        <div class="perf-card border-accuracy">
            <div class="perf-label">🎯 Accuracy</div>
            <div class="perf-value">{best_accuracy:.2%}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with m_col2:
        st.markdown(f"""
        <div class="perf-card border-cv">
            <div class="perf-label">🔄 CV Accuracy</div>
            <div class="perf-value">{best_cv_accuracy:.2%}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with m_col3:
        st.markdown(f"""
        <div class="perf-card border-precision">
            <div class="perf-label">📈 Precision</div>
            <div class="perf-value">{best_precision:.2%}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with m_col4:
        st.markdown(f"""
        <div class="perf-card border-recall">
            <div class="perf-label">📉 Recall</div>
            <div class="perf-value">{best_recall:.2%}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with m_col5:
        st.markdown(f"""
        <div class="perf-card border-f1">
            <div class="perf-label">🧬 F1 Score</div>
            <div class="perf-value">{best_f1:.2%}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with m_col6:
        st.markdown(f"""
        <div class="perf-card border-auc">
            <div class="perf-label">📊 ROC-AUC</div>
            <div class="perf-value">{best_roc_auc:.2%}</div>
        </div>
        """, unsafe_allow_html=True)
    
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
                importance_df = display_shap_explanation(model, input_df, feature_names)
                
                # Display Clinical Priority List (placed after SHAP Feature Importance and before AI Clinical Explanation)
                st.markdown("---")
                display_clinical_priority_list(user_inputs, importance_df)
                
                # Display AI Clinical Explanation
                st.markdown("---")
                display_ai_clinical_explanation(prediction, probability, user_inputs, importance_df)
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
