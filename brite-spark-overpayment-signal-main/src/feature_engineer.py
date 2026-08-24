"""
Feature engineering for ranking.
"""

import pandas as pd
import numpy as np
from src.utils import setup_logging

logger = setup_logging()

NEEDS_FIGURE = {
    1: 1240,
    2: 1670,
    3: 2000,
    4: 2330,
    5: 2660,
    6: 2990
}

def engineer_features(df):
    """Create features for ranking suspicious cases."""
    logger.info("Engineering features...")
    
    df_fe = df.copy()
    
    # 1. Award-to-needs ratio
    df_fe['award_to_needs_ratio'] = df_fe.apply(
        lambda row: row['monthly_award'] / NEEDS_FIGURE.get(row['household_size'], 2000),
        axis=1
    )
    
    # 2. Suspicious award (too high or too low)
    df_fe['suspicious_award_high'] = (df_fe['award_to_needs_ratio'] > 0.8).astype(int)
    df_fe['suspicious_award_low'] = (df_fe['award_to_needs_ratio'] < 0.2).astype(int)
    
    # 3. Payment mismatch
    df_fe['payment_mismatch'] = np.abs(
        df_fe['avg_payment'] - df_fe['monthly_award']
    ) / (df_fe['monthly_award'] + 1)
    df_fe['payment_mismatch'] = df_fe['payment_mismatch'].fillna(0)
    
    # 4. Adjustment intensity
    df_fe['adjustment_intensity'] = df_fe['adjustment_count'] / (df_fe['payment_count'] + 1)
    
    # 5. Contact intensity
    df_fe['contact_intensity'] = df_fe['contact_attempts'] / (df_fe['months_since_review'] + 1)
    
    # 6. Status flags
    df_fe['is_suspended'] = (df_fe['status'] == 'Suspended').astype(int)
    df_fe['is_closed'] = (df_fe['status'] == 'Closed').astype(int)
    
    # 7. Time-based features
    df_fe['opened_date'] = pd.to_datetime(df_fe['opened_date'])
    df_fe['months_since_opened'] = (pd.Timestamp.now() - df_fe['opened_date']).dt.days // 30
    
    # 8. Demographic encoding for fairness checks
    df_fe['age_band_encoded'] = df_fe['age_band'].astype('category').cat.codes
    df_fe['language_encoded'] = df_fe['language_preference'].astype('category').cat.codes
    df_fe['district_encoded'] = df_fe['district'].astype('category').cat.codes
    df_fe['tenure_encoded'] = df_fe['tenure'].astype('category').cat.codes
    
    logger.info(f"Engineered {len(df_fe.columns)} features")
    return df_fe