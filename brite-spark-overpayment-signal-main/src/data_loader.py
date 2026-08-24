"""
Data loading module for cases.csv and payments.csv
"""

import pandas as pd
from pathlib import Path
from src.utils import setup_logging

logger = setup_logging()

def load_cases(filepath="data/raw/cases.csv"):
    """Load cases.csv data."""
    logger.info(f"Loading cases from {filepath}")
    
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Cases file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} cases with {len(df.columns)} columns")
    
    return df

def load_payments(filepath="data/raw/payments.csv"):
    """Load payments.csv data."""
    logger.info(f"Loading payments from {filepath}")
    
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Payments file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} payments with {len(df.columns)} columns")
    
    return df

def validate_cases(df):
    """Validate cases data quality."""
    logger.info("Validating cases data...")
    
    required_cols = [
        'case_id', 'district', 'household_size', 'age_band',
        'language_preference', 'tenure', 'opened_date', 'status',
        'monthly_award', 'payment_adjustments', 'contact_attempts',
        'months_since_review'
    ]
    
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    critical_cols = ['case_id', 'monthly_award', 'household_size']
    df = df.dropna(subset=critical_cols)
    
    logger.info(f"Validated: {len(df)} cases remain")
    return df

def validate_payments(df):
    """Validate payments data quality."""
    logger.info("Validating payments data...")
    
    required_cols = ['payment_id', 'case_id', 'pay_month', 'amount', 'method', 'adjustment']
    
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    df = df.dropna(subset=['case_id', 'amount'])
    
    logger.info(f"Validated: {len(df)} payments remain")
    return df