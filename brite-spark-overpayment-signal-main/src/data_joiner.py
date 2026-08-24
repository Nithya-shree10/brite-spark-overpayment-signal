"""
Join cases and payments data for analysis.
"""

import pandas as pd
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

def join_data(cases_df, payments_df):
    """Join cases and payments on case_id."""
    logger.info("Joining cases and payments...")
    
    # Aggregate payments per case
    payment_agg = payments_df.groupby('case_id').agg({
        'amount': ['mean', 'max', 'min', 'sum', 'count'],
        'adjustment': lambda x: (x == 'Y').sum(),
        'method': lambda x: (x == 'Transfer').sum() / len(x) if len(x) > 0 else 0
    }).reset_index()
    
    # Flatten column names
    payment_agg.columns = [
        'case_id', 
        'avg_payment', 'max_payment', 'min_payment', 
        'total_payments', 'payment_count',
        'adjustment_count', 'transfer_ratio'
    ]
    
    # Join with cases
    joined = cases_df.merge(payment_agg, on='case_id', how='left')
    
    # Fill missing payment aggregates
    joined['avg_payment'] = joined['avg_payment'].fillna(0)
    joined['max_payment'] = joined['max_payment'].fillna(0)
    joined['min_payment'] = joined['min_payment'].fillna(0)
    joined['total_payments'] = joined['total_payments'].fillna(0)
    joined['payment_count'] = joined['payment_count'].fillna(0)
    joined['adjustment_count'] = joined['adjustment_count'].fillna(0)
    joined['transfer_ratio'] = joined['transfer_ratio'].fillna(0)
    
    logger.info(f"Joined: {len(joined)} cases with payment aggregates")
    return joined