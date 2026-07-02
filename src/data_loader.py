"""
Data Loader Module

This module provides functions to load clinical datasets from various sources.
Currently supports Excel files for diabetic nephropathy prediction.
"""

import pandas as pd
from pathlib import Path
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset(file_path: str, sheet_name: Optional[str] = 0) -> pd.DataFrame:
    """
    Load dataset from an Excel file.
    
    Args:
        file_path: Path to the Excel file
        sheet_name: Name or index of the sheet to load (default: 0)
    
    Returns:
        Loaded DataFrame
    
    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file cannot be read
    """
    try:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Loading dataset from: {file_path}")
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        logger.info(f"Dataset loaded successfully with shape: {df.shape}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise


def display_dataset_info(df: pd.DataFrame) -> None:
    """
    Display basic information about the dataset.
    
    Args:
        df: Input DataFrame
    """
    print("\n" + "="*60)
    print("DATASET INFORMATION")
    print("="*60)
    
    print(f"\nShape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nColumn Names:")
    print(df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.head())
