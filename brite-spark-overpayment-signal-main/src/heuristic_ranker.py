"""
Heuristic ranker based on policy rules and suspicious patterns.
UPDATED: Day 2 change - reduced weight for Department activity (adjustments, contacts)
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

def rank_by_heuristic(df):
    """Rank cases using heuristic rules."""
    logger.info("Using heuristic ranker (fallback mode)...")
    
    df_rank = df.copy()
    
    # Award-to-needs ratio
    df_rank['award_ratio'] = df_rank.apply(
        lambda row: row['monthly_award'] / NEEDS_FIGURE.get(row['household_size'], 2000),
        axis=1
    )
    df_rank['award_score'] = np.minimum(np.abs(df_rank['award_ratio'] - 0.5) * 2, 1.0)
    
    # Payment mismatch
    df_rank['payment_mismatch'] = np.abs(
        df_rank['avg_payment'] - df_rank['monthly_award']
    ) / (df_rank['monthly_award'] + 1)
    df_rank['mismatch_score'] = np.minimum(df_rank['payment_mismatch'], 1.0)
    
    # Adjustment score - REDUCED WEIGHT (Day 2 change: Department activity != fraud)
    df_rank['adjustment_score'] = np.minimum(df_rank['adjustment_count'] / 5, 1.0)
    
    # Contact score - REDUCED WEIGHT (Day 2 change: communication issues != fraud)
    df_rank['contact_score'] = np.minimum(df_rank['contact_attempts'] / 10, 1.0)
    
    # Review score
    df_rank['review_score'] = np.minimum(df_rank['months_since_review'] / 24, 1.0)
    
    # Status score
    df_rank['status_score'] = df_rank['status'].apply(
        lambda x: 1.0 if x == 'Suspended' else 0.5 if x == 'Closed' else 0.1
    )
    
    # UPDATED WEIGHTS - Day 2 change based on investigator feedback
    # Reduced adjustment and contact weights (Department activity)
    # Increased financial anomaly weights (award, mismatch)
    weights = {
        'award_score': 0.30,        # INCREASED - financial anomaly
        'mismatch_score': 0.25,     # INCREASED - payment mismatch
        'adjustment_score': 0.05,   # REDUCED - Department activity, not fraud
        'contact_score': 0.05,      # REDUCED - Department communication issue
        'review_score': 0.15,       # INCREASED - long overdue review
        'status_score': 0.20        # KEPT - status matters
    }
    
    df_rank['risk_score'] = (
        weights['award_score'] * df_rank['award_score'] +
        weights['mismatch_score'] * df_rank['mismatch_score'] +
        weights['adjustment_score'] * df_rank['adjustment_score'] +
        weights['contact_score'] * df_rank['contact_score'] +
        weights['review_score'] * df_rank['review_score'] +
        weights['status_score'] * df_rank['status_score']
    )
    
    df_rank = df_rank.sort_values('risk_score', ascending=False)
    df_rank['rank'] = range(1, len(df_rank) + 1)
    
    # Normalize
    min_score = df_rank['risk_score'].min()
    max_score = df_rank['risk_score'].max()
    if max_score > min_score:
        df_rank['risk_score_normalized'] = (df_rank['risk_score'] - min_score) / (max_score - min_score)
    else:
        df_rank['risk_score_normalized'] = 0
    
    # Generate reasons
    reasons = []
    for _, row in df_rank.iterrows():
        reason_parts = []
        if row['award_score'] > 0.7:
            ratio = row['award_ratio']
            if ratio > 0.8:
                reason_parts.append(f"award is {ratio*100:.0f}% of needs (unusually high)")
            elif ratio < 0.2:
                reason_parts.append(f"award is {ratio*100:.0f}% of needs (unusually low)")
        if row['mismatch_score'] > 0.5:
            reason_parts.append(f"payment (${row['avg_payment']:.0f}) differs from award (${row['monthly_award']:.0f})")
        if row['adjustment_score'] > 0.3:
            reason_parts.append(f"{int(row['adjustment_count'])} payment adjustments (may be Department activity)")
        if row['contact_score'] > 0.4:
            reason_parts.append(f"{int(row['contact_attempts'])} contact attempts (may be communication issues)")
        if row['review_score'] > 0.5:
            reason_parts.append(f"{int(row['months_since_review'])} months since review")
        if row['status'] == 'Suspended':
            reason_parts.append("case is suspended")
        if not reason_parts:
            reason_parts.append("moderate risk indicators")
        reasons.append("; ".join(reason_parts))
    
    df_rank['reason'] = reasons
    
    logger.info(f"Heuristic ranking complete. Top score: {df_rank['risk_score'].max():.3f}")
    return df_rank
def apply_investigator_feedback(df_ranked):
    """
    Day 2 Change: Re-rank cases based on investigator feedback.
    Reduces score for cases where adjustments/contacts are Department-driven.
    """
    logger.info("Applying investigator feedback to ranking...")
    
    df = df_ranked.copy()
    
    # Calculate a "Department activity score" (not fraud)
    df['dept_activity_score'] = (
        (df['adjustment_count'] / 5) * 0.5 +
        (df['contact_attempts'] / 10) * 0.5
    )
    
    # Penalty up to 30% of the normalized score
    df['penalty'] = df['dept_activity_score'] * 0.3
    
    # Apply penalty to the NORMALIZED risk score, not the raw anomaly score
    # so the penalty scale matches the score scale it's adjusting.
    df['risk_score_original'] = df['risk_score_normalized']
    df['risk_score_normalized'] = (df['risk_score_normalized'] - df['penalty']).clip(lower=0)
    df['risk_score'] = df['risk_score_normalized']  # keep consistent downstream
    
    # Re-rank
    df = df.sort_values('risk_score', ascending=False)
    df['rank'] = range(1, len(df) + 1)
    
    logger.info("Investigator feedback applied. Top 3 cases now: " + 
                ", ".join(df.head(3)['case_id'].tolist()))
    
    return df
