"""
Model Module

This module provides functions for training machine learning models.
Currently supports XGBoost Classifier for diabetic nephropathy prediction.
"""

import xgboost as xgb
import joblib
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_xgboost_classifier(
    X_train,
    y_train,
    params: Optional[Dict[str, Any]] = None,
    random_state: int = 42
) -> xgb.XGBClassifier:
    """
    Train an XGBoost Classifier.
    
    Args:
        X_train: Training features
        y_train: Training target
        params: XGBoost hyperparameters (default: uses sensible defaults)
        random_state: Random seed for reproducibility
    
    Returns:
        Trained XGBoost classifier model
    """
    logger.info("Training XGBoost Classifier...")
    
    # Default hyperparameters
    default_params = {
        'n_estimators': 200,
        'max_depth': 4,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': random_state,
        'eval_metric': 'logloss'
    }
    
    # Update with user-provided parameters
    if params:
        default_params.update(params)
    
    logger.info(f"Model parameters: {default_params}")
    
    # Initialize and train model
    model = xgb.XGBClassifier(**default_params)
    model.fit(X_train, y_train)
    
    logger.info("XGBoost Classifier trained successfully")
    return model


def save_model(
    model,
    model_path: str,
    model_name: str = 'xgboost_model.joblib'
) -> str:
    """
    Save trained model to disk using joblib.
    
    Args:
        model: Trained model object
        model_path: Directory path to save the model
        model_name: Name of the model file
    
    Returns:
        Full path to the saved model
    """
    logger.info(f"Saving model to: {model_path}")
    
    # Create directory if it doesn't exist
    Path(model_path).mkdir(parents=True, exist_ok=True)
    
    # Full path to model file
    full_path = Path(model_path) / model_name
    
    # Save model
    joblib.dump(model, full_path)
    
    logger.info(f"Model saved successfully at: {full_path}")
    return str(full_path)


def load_model(model_path: str):
    """
    Load a saved model from disk.
    
    Args:
        model_path: Full path to the model file
    
    Returns:
        Loaded model object
    """
    logger.info(f"Loading model from: {model_path}")
    
    model = joblib.load(model_path)
    
    logger.info("Model loaded successfully")
    return model


def get_model_feature_importance(model: xgb.XGBClassifier, feature_names: list) -> Dict[str, float]:
    """
    Get feature importance from trained XGBoost model.
    
    Args:
        model: Trained XGBoost model
        feature_names: List of feature names
    
    Returns:
        Dictionary mapping feature names to importance scores
    """
    importance_dict = dict(zip(feature_names, model.feature_importances_))
    
    # Sort by importance
    importance_dict = dict(sorted(importance_dict.items(), 
                                  key=lambda x: x[1], 
                                  reverse=True))
    
    return importance_dict
