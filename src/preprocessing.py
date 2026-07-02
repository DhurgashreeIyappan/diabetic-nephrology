"""
Preprocessing Module

This module provides functions for data preprocessing including:
- Data quality checks
- Missing value handling
- Duplicate removal
- Categorical encoding
- Class imbalance handling
- Train-test splitting
- Feature scaling
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.utils import resample
from typing import Tuple, Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def display_data_quality(df: pd.DataFrame) -> None:
    """
    Display comprehensive data quality report.
    
    Args:
        df: Input DataFrame
    """
    print("\n" + "="*60)
    print("DATA QUALITY REPORT")
    print("="*60)
    
    # Shape
    print(f"\nDataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Data types
    print("\nData Types:")
    print(df.dtypes)
    
    # Missing values
    print("\nMissing Values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Count': missing,
        'Percentage': missing_pct
    })
    print(missing_df[missing_df['Count'] > 0] if missing_df['Count'].sum() > 0 else "No missing values")
    
    # Duplicate rows
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate Rows: {duplicates} ({(duplicates/len(df)*100):.2f}%)")
    
    # Dataset summary
    print("\nDataset Summary:")
    print(df.describe(include='all'))


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = 'mean',
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Args:
        df: Input DataFrame
        strategy: Strategy for imputation ('mean', 'median', 'mode', 'drop')
        columns: Specific columns to process (None = all columns)
    
    Returns:
        DataFrame with missing values handled
    """
    logger.info(f"Handling missing values using strategy: {strategy}")
    
    df_processed = df.copy()
    
    if columns is None:
        columns = df_processed.columns.tolist()
    
    for col in columns:
        if df_processed[col].isnull().sum() > 0:
            if strategy == 'drop':
                df_processed = df_processed.dropna(subset=[col])
                logger.info(f"Dropped rows with missing values in '{col}'")
            
            elif df_processed[col].dtype in ['int64', 'float64']:
                if strategy == 'mean':
                    fill_value = df_processed[col].mean()
                elif strategy == 'median':
                    fill_value = df_processed[col].median()
                else:
                    fill_value = df_processed[col].mean()
                
                df_processed[col].fillna(fill_value, inplace=True)
                logger.info(f"Filled missing values in '{col}' with {strategy}: {fill_value:.2f}")
            
            else:  # Categorical
                if strategy == 'mode':
                    fill_value = df_processed[col].mode()[0]
                else:
                    fill_value = df_processed[col].mode()[0]
                
                df_processed[col].fillna(fill_value, inplace=True)
                logger.info(f"Filled missing values in '{col}' with mode: {fill_value}")
    
    logger.info(f"Missing value handling complete. Remaining shape: {df_processed.shape}")
    return df_processed


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from the dataset.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with duplicates removed
    """
    initial_count = len(df)
    duplicates = df.duplicated().sum()
    
    if duplicates > 0:
        logger.info(f"Found {duplicates} duplicate rows. Removing...")
        df_clean = df.drop_duplicates()
        logger.info(f"Removed {duplicates} duplicates. Shape: {df_clean.shape}")
        return df_clean
    else:
        logger.info("No duplicate rows found.")
        return df


def encode_categorical_columns(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = 'label'
) -> Tuple[pd.DataFrame, dict]:
    """
    Encode categorical columns.
    
    Args:
        df: Input DataFrame
        columns: Specific columns to encode (None = auto-detect)
        method: Encoding method ('label' or 'onehot')
    
    Returns:
        Tuple of (encoded DataFrame, encoders dictionary)
    """
    logger.info(f"Encoding categorical columns using method: {method}")
    
    df_encoded = df.copy()
    encoders = {}
    
    if columns is None:
        columns = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not columns:
        logger.info("No categorical columns found.")
        return df_encoded, encoders
    
    for col in columns:
        if method == 'label':
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            encoders[col] = le
            logger.info(f"Label encoded '{col}' with {len(le.classes_)} unique values")
        
        elif method == 'onehot':
            dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
            df_encoded = pd.concat([df_encoded.drop(col, axis=1), dummies], axis=1)
            encoders[col] = dummies.columns.tolist()
            logger.info(f"One-hot encoded '{col}' into {dummies.shape[1]} columns")
    
    logger.info(f"Categorical encoding complete. Shape: {df_encoded.shape}")
    return df_encoded, encoders


def handle_class_imbalance(
    df: pd.DataFrame,
    target_column: str,
    method: Optional[str] = None
) -> pd.DataFrame:
    """
    Handle class imbalance in the target variable.
    
    Args:
        df: Input DataFrame
        target_column: Name of the target column
        method: Resampling method ('oversample', 'undersample', or None)
    
    Returns:
        DataFrame with balanced classes
    """
    if method is None:
        logger.info("Skipping class imbalance handling")
        return df
    
    logger.info(f"Handling class imbalance using method: {method}")
    
    class_counts = df[target_column].value_counts()
    logger.info(f"Class distribution before balancing:\n{class_counts}")
    
    if method == 'oversample':
        # Oversample minority class
        majority_class = class_counts.idxmax()
        minority_class = class_counts.idxmin()
        
        df_majority = df[df[target_column] == majority_class]
        df_minority = df[df[target_column] == minority_class]
        
        df_minority_upsampled = resample(
            df_minority,
            replace=True,
            n_samples=len(df_majority),
            random_state=42
        )
        
        df_balanced = pd.concat([df_majority, df_minority_upsampled])
        logger.info(f"Oversampled minority class. New shape: {df_balanced.shape}")
    
    elif method == 'undersample':
        # Undersample majority class
        majority_class = class_counts.idxmax()
        minority_class = class_counts.idxmin()
        
        df_majority = df[df[target_column] == majority_class]
        df_minority = df[df[target_column] == minority_class]
        
        df_majority_undersampled = resample(
            df_majority,
            replace=False,
            n_samples=len(df_minority),
            random_state=42
        )
        
        df_balanced = pd.concat([df_majority_undersampled, df_minority])
        logger.info(f"Undersampled majority class. New shape: {df_balanced.shape}")
    
    else:
        logger.warning(f"Unknown method: {method}. Skipping imbalance handling.")
        return df
    
    logger.info(f"Class distribution after balancing:\n{df_balanced[target_column].value_counts()}")
    return df_balanced


def split_features_target(
    df: pd.DataFrame,
    target_column: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split dataset into features (X) and target (y).
    
    Args:
        df: Input DataFrame
        target_column: Name of the target column
    
    Returns:
        Tuple of (X, y) where X is features DataFrame and y is target Series
    """
    logger.info(f"Splitting features and target. Target column: '{target_column}'")
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
    return X, y


def train_test_split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform train-test split.
    
    Args:
        X: Features DataFrame
        y: Target Series
        test_size: Proportion of data for testing (default: 0.2)
        random_state: Random seed for reproducibility
        stratify: Whether to use stratified splitting
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    logger.info(f"Performing {test_size*100:.0f}-{(1-test_size)*100:.0f} train-test split")
    
    stratify_param = y if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_param
    )
    
    logger.info(f"Training set shape: {X_train.shape}")
    logger.info(f"Testing set shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test


def apply_feature_scaling(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    method: str = 'standard',
    columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, object]:
    """
    Apply feature scaling to the dataset.
    
    Args:
        X_train: Training features
        X_test: Testing features
        method: Scaling method ('standard' or 'minmax')
        columns: Specific columns to scale (None = all numeric columns)
    
    Returns:
        Tuple of (scaled_X_train, scaled_X_test, scaler)
    """
    logger.info(f"Applying feature scaling using method: {method}")
    
    if columns is None:
        columns = X_train.select_dtypes(include=[np.number]).columns.tolist()
    
    if not columns:
        logger.info("No numeric columns found for scaling.")
        return X_train, X_test, None
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        logger.warning(f"Unknown method: {method}. Using StandardScaler.")
        scaler = StandardScaler()
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[columns] = scaler.fit_transform(X_train[columns])
    X_test_scaled[columns] = scaler.transform(X_test[columns])
    
    logger.info(f"Scaling applied to {len(columns)} columns")
    logger.info(f"Training set scaled range: [{X_train_scaled[columns].min().min():.2f}, {X_train_scaled[columns].max().max():.2f}]")
    
    return X_train_scaled, X_test_scaled, scaler


def preprocess_pipeline(
    df: pd.DataFrame,
    target_column: str,
    missing_strategy: str = 'mean',
    encode_categorical: bool = True,
    encoding_method: str = 'label',
    handle_imbalance: Optional[str] = None,
    apply_scaling: bool = True,
    scaling_method: str = 'standard',
    test_size: float = 0.2,
    random_state: int = 42
) -> dict:
    """
    Complete preprocessing pipeline.
    
    Args:
        df: Input DataFrame
        target_column: Name of the target column
        missing_strategy: Strategy for missing values ('mean', 'median', 'mode', 'drop')
        encode_categorical: Whether to encode categorical columns
        encoding_method: Method for encoding ('label' or 'onehot')
        handle_imbalance: Method for handling class imbalance ('oversample', 'undersample', or None)
        apply_scaling: Whether to apply feature scaling
        scaling_method: Method for scaling ('standard' or 'minmax')
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary containing all preprocessing artifacts
    """
    logger.info("="*60)
    logger.info("STARTING PREPROCESSING PIPELINE")
    logger.info("="*60)
    
    artifacts = {}
    
    # Step 1: Display data quality
    display_data_quality(df)
    
    # Step 2: Handle missing values
    df_clean = handle_missing_values(df, strategy=missing_strategy)
    artifacts['missing_strategy'] = missing_strategy
    
    # Step 3: Remove duplicates
    df_clean = remove_duplicates(df_clean)
    
    # Step 4: Handle class imbalance (before encoding)
    if handle_imbalance:
        df_clean = handle_class_imbalance(df_clean, target_column, method=handle_imbalance)
        artifacts['imbalance_method'] = handle_imbalance
    
    # Step 5: Encode categorical columns
    if encode_categorical:
        df_encoded, encoders = encode_categorical_columns(df_clean, method=encoding_method)
        artifacts['encoders'] = encoders
        artifacts['encoding_method'] = encoding_method
    else:
        df_encoded = df_clean.copy()
    
    # Step 6: Split features and target
    X, y = split_features_target(df_encoded, target_column)
    
    # Step 7: Train-test split
    X_train, X_test, y_train, y_test = train_test_split_data(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Step 8: Apply feature scaling
    if apply_scaling:
        X_train_scaled, X_test_scaled, scaler = apply_feature_scaling(
            X_train, X_test, method=scaling_method
        )
        artifacts['scaler'] = scaler
        artifacts['scaling_method'] = scaling_method
        X_train_final, X_test_final = X_train_scaled, X_test_scaled
    else:
        X_train_final, X_test_final = X_train, X_test
    
    # Store all artifacts
    artifacts.update({
        'X_train': X_train_final,
        'X_test': X_test_final,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': X.columns.tolist(),
        'target_column': target_column
    })
    
    logger.info("="*60)
    logger.info("PREPROCESSING PIPELINE COMPLETED")
    logger.info("="*60)
    
    return artifacts
