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

# Custom CSS for clean UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        padding: 2rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #34495e;
        padding: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_trained_model():
    """
    Load the trained XGBoost model and preprocessing artifacts.
    """
    try:
        model_path = 'models/xgboost_diabetic_nephropathy.joblib'
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_resource
def load_feature_names():
    """
    Load feature names from the preprocessing artifacts.
    This is a placeholder - in production, load from saved artifacts.
    """
    # Actual feature names from the dataset
    actual_features = [
        'Sex',
        'Age',
        'Diabetes duration (y)',
        'Diabetic retinopathy (DR)',
        'Smoking',
        'Drinking',
        'Height(cm)',
        'Weight(kg)',
        'BMI (kg/m2)',
        'SBP (mmHg) ',
        'DBP (mmHg)',
        'HbA1c (%)',
        'FBG (mmol/L)'
    ]
    return actual_features


def create_input_fields(feature_names):
    """
    Create input fields for each feature based on data type.
    
    Args:
        feature_names: List of feature names
    
    Returns:
        Dictionary of user inputs
    """
    user_inputs = {}
    
    st.markdown('<div class="sub-header">Patient Clinical Information</div>', 
                unsafe_allow_html=True)
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        for idx, feature in enumerate(feature_names[:len(feature_names)//2]):
            # Determine input type based on feature name
            if 'Sex' in feature:
                user_inputs[feature] = st.selectbox(
                    f"{feature}",
                    options=['Male', 'Female'],
                    key=f"input_{feature}"
                )
            elif 'Smoking' in feature:
                user_inputs[feature] = st.selectbox(
                    f"{feature}",
                    options=['No', 'Yes'],
                    key=f"input_{feature}"
                )
            elif 'Drinking' in feature:
                user_inputs[feature] = st.selectbox(
                    f"{feature}",
                    options=['No', 'Yes'],
                    key=f"input_{feature}"
                )
            elif 'DR' in feature:
                user_inputs[feature] = st.selectbox(
                    f"{feature}",
                    options=['No', 'Yes'],
                    key=f"input_{feature}"
                )
            else:
                # Default to numeric input
                user_inputs[feature] = st.number_input(
                    f"{feature}",
                    value=0.0,
                    step=0.1,
                    key=f"input_{feature}"
                )
    
    with col2:
        for idx, feature in enumerate(feature_names[len(feature_names)//2:]):
            if 'Sex' in feature:
                user_inputs[feature] = st.selectbox(
                    f"{feature}",
                    options=['Male', 'Female'],
                    key=f"input_{feature}"
                )
            elif 'Smoking' in feature:
                user_inputs[feature] = st.selectbox(
                    f"{feature}",
                    options=['No', 'Yes'],
                    key=f"input_{feature}"
                )
            elif 'Drinking' in feature:
                user_inputs[feature] = st.selectbox(
                    f"{feature}",
                    options=['No', 'Yes'],
                    key=f"input_{feature}"
                )
            elif 'DR' in feature:
                user_inputs[feature] = st.selectbox(
                    f"{feature}",
                    options=['No', 'Yes'],
                    key=f"input_{feature}"
                )
            else:
                user_inputs[feature] = st.number_input(
                    f"{feature}",
                    value=0.0,
                    step=0.1,
                    key=f"input_{feature}"
                )
    
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
    # Make prediction
    prediction = model.predict(input_df)[0]
    
    # Get probability
    probability = model.predict_proba(input_df)[0]
    
    return prediction, probability


def display_prediction_result(prediction, probability):
    """
    Display prediction result and probability.
    
    Args:
        prediction: Predicted class
        probability: Prediction probabilities
    """
    st.markdown('<div class="sub-header">Prediction Result</div>', 
                unsafe_allow_html=True)
    
    # Display prediction
    if prediction == 1:
        st.markdown("""
        <div class="warning-box">
            <h3 style="color: #856404;">⚠️ High Risk Detected</h3>
            <p>The model predicts that this patient is at <strong>high risk</strong> of developing diabetic nephropathy.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="success-box">
            <h3 style="color: #155724;">✅ Low Risk</h3>
            <p>The model predicts that this patient is at <strong>low risk</strong> of developing diabetic nephropathy.</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Display probability
    st.markdown('<div class="sub-header">Prediction Probability</div>', 
                unsafe_allow_html=True)
    
    # Create probability bar chart
    prob_df = pd.DataFrame({
        'Class': ['Low Risk', 'High Risk'],
        'Probability': probability
    })
    
    fig = go.Figure(data=[
        go.Bar(
            x=prob_df['Class'],
            y=prob_df['Probability'],
            marker_color=['#2ecc71', '#e74c3c'],
            text=[f"{p:.2%}" for p in prob_df['Probability']],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Prediction Probability Distribution",
        yaxis_title="Probability",
        yaxis_range=[0, 1],
        showlegend=False,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def display_shap_explanation(model, input_df, feature_names):
    """
    Display SHAP explanation for the prediction.
    
    Args:
        model: Trained XGBoost model
        input_df: Preprocessed input DataFrame
        feature_names: List of feature names
    """
    st.markdown('<div class="sub-header">SHAP Feature Importance</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <p><strong>SHAP (SHapley Additive exPlanations)</strong> values show how each feature 
    contributed to this specific prediction.</p>
    <ul>
        <li><strong>Positive values</strong>: Feature increases the risk prediction</li>
        <li><strong>Negative values</strong>: Feature decreases the risk prediction</li>
        <li><strong>Higher absolute values</strong>: Stronger impact on prediction</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        # Initialize SHAP explainer
        explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(input_df)
        
        # Handle multi-dimensional SHAP values (get positive class)
        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap_values = shap_values[1]  # Use positive class SHAP values
        
        # Ensure shap_values is 1D for single sample
        if len(shap_values.shape) > 1:
            shap_values = shap_values[0]
        
        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'SHAP Value': shap_values,
            'Impact': ['Increases Risk' if v > 0 else 'Decreases Risk' for v in shap_values]
        }).sort_values(by='SHAP Value', key=abs, ascending=False)
        
        # Display top 10 features
        top_features = importance_df.head(10)
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=top_features['SHAP Value'],
                y=top_features['Feature'],
                orientation='h',
                marker_color=['#e74c3c' if v > 0 else '#2ecc71' for v in top_features['SHAP Value']],
                text=[f"{v:.4f}" for v in top_features['SHAP Value']],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Top 10 Feature Contributions to Prediction",
            xaxis_title="SHAP Value",
            yaxis_title="Features",
            height=500,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display feature contribution table
        st.markdown('<div class="sub-header">Detailed Feature Contributions</div>', 
                    unsafe_allow_html=True)
        
        st.dataframe(
            top_features.style.format({'SHAP Value': '{:.4f}'}),
            use_container_width=True,
            hide_index=True
        )
        
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
    
    st.markdown("""
    <div class="info-box">
    <p>This system uses machine learning to predict the risk of diabetic nephropathy 
    based on clinical parameters. Enter patient information below to get a prediction 
    with explainable AI insights.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model = load_trained_model()
    
    if model is None:
        st.error("Model could not be loaded. Please ensure the model file exists in the 'models' directory.")
        st.stop()
    
    # Load feature names
    feature_names = load_feature_names()
    
    # Create input fields
    user_inputs = create_input_fields(feature_names)
    
    # Predict button
    st.markdown("---")
    predict_button = st.button("🔮 Predict Risk", type="primary", use_container_width=True)
    
    if predict_button:
        # Show loading spinner
        with st.spinner("Processing prediction..."):
            # Preprocess input
            input_df = preprocess_input(user_inputs, feature_names)
            
            # Make prediction
            prediction, probability = make_prediction(model, input_df)
            
            # Display results
            st.markdown("---")
            display_prediction_result(prediction, probability)
            
            # Display SHAP explanation
            st.markdown("---")
            display_shap_explanation(model, input_df, feature_names)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d; padding: 2rem 0;">
        <p><strong>Disclaimer:</strong> This system is for educational and research purposes only. 
        It should not be used as a substitute for professional medical advice, diagnosis, or treatment.</p>
        <p>© 2024 Diabetic Nephropathy Prediction System</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
