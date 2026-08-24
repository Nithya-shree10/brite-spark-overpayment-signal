"""
Plain-language explainer for top K cases.
Uses actual features from the Isolation Forest model.

FIXES APPLIED:
1. Explicitly sorts by risk_score before selecting top K
2. Uses correct column names (award_to_needs_ratio, not award_ratio)
3. Uses z-score/percentile-based explanations for actual model features
4. Separates model explanation from Department activity context
5. Only uses features that the Isolation Forest model actually uses
6. Handles missing columns gracefully
7. Adds fallback for cases with no extreme features
"""

import pandas as pd
import numpy as np
from src.utils import setup_logging

logger = setup_logging()

# Model features actually used by Isolation Forest
MODEL_FEATURES = [
    'award_to_needs_ratio',
    'payment_mismatch',
    'months_since_review',
    'transfer_ratio'
]

# Human-readable names for features
FEATURE_NAMES = {
    'award_to_needs_ratio': 'Award-to-needs ratio',
    'payment_mismatch': 'Payment mismatch',
    'months_since_review': 'Months since review',
    'transfer_ratio': 'Transfer payment ratio'
}

def explain_top_k(ranked_df, k=20):
    """
    Generate plain-language explanations for top K cases.
    Uses actual features from the Isolation Forest model.
    
    Args:
        ranked_df: DataFrame with risk scores and features
        k: Number of top cases to explain (default 20)
    
    Returns:
        List of dicts with case_id, rank, risk_score, reason, and top_features
    """
    logger.info(f"Generating explanations for top {k} cases...")
    
    # Explicitly sort before selecting top K
    if 'risk_score' in ranked_df.columns:
        sorted_df = ranked_df.sort_values('risk_score', ascending=False)
        top_k = sorted_df.head(k).copy()
        logger.info(f"Sorted by risk_score, selected top {k} cases")
    else:
        logger.warning("No risk_score column found. Using first k rows without sorting.")
        top_k = ranked_df.head(k).copy()
    
    # Calculate population statistics for comparison
    pop_stats = {}
    for col in MODEL_FEATURES:
        if col in ranked_df.columns:
            pop_stats[col] = {
                'mean': ranked_df[col].mean(),
                'std': ranked_df[col].std(),
                'p95': ranked_df[col].quantile(0.95),
                'p5': ranked_df[col].quantile(0.05),
                'min': ranked_df[col].min(),
                'max': ranked_df[col].max()
            }
    
    explanations = []
    
    for idx, row in top_k.iterrows():
        case_id = row.get('case_id', f"CASE_{idx}")
        
        # Calculate which features are most extreme for this case
        feature_extremes = []
        
        for col in MODEL_FEATURES:
            if col not in row or col not in pop_stats:
                continue
            
            value = row[col]
            stats = pop_stats[col]
            mean = stats['mean']
            std = stats['std']
            
            # Skip if standard deviation is zero
            if std == 0 or pd.isna(std):
                continue
            
            # Calculate z-score (how many standard deviations from mean)
            z_score = (value - mean) / std
            
            # Only consider features that are actually unusual (|z| > 1.0)
            if abs(z_score) > 1.0:
                direction = "high" if z_score > 0 else "low"
                
                # Calculate percentile for context
                if (stats['max'] - stats['min']) > 0:
                    percentile = (value - stats['min']) / (stats['max'] - stats['min'])
                else:
                    percentile = 0.5
                
                feature_extremes.append({
                    'feature': col,
                    'value': value,
                    'z_score': z_score,
                    'direction': direction,
                    'magnitude': abs(z_score),
                    'percentile': percentile
                })
        
        # Sort by magnitude (most extreme first)
        feature_extremes.sort(key=lambda x: x['magnitude'], reverse=True)
        
        # Take top 2 most extreme features for explanation
        top_features = feature_extremes[:2]
        
        if top_features:
            # Build explanation from top features
            reason_parts = []
            for feat in top_features:
                display_name = FEATURE_NAMES.get(feat['feature'], feat['feature'])
                value = feat['value']
                direction = feat['direction']
                
                # Format value based on feature type
                if feat['feature'] in ['award_to_needs_ratio', 'transfer_ratio']:
                    value_str = f"{value*100:.0f}%"
                elif feat['feature'] == 'months_since_review':
                    value_str = f"{int(value)} months"
                elif feat['feature'] == 'payment_mismatch':
                    value_str = f"{value:.2f}"
                else:
                    value_str = f"{value:.2f}"
                
                reason_parts.append(f"{display_name} is {direction} ({value_str})")
            
            primary_reason = "; ".join(reason_parts)
        else:
            # No extreme features - explain anomaly score instead
            risk_score = row.get('risk_score_normalized', row.get('risk_score', 0.5))
            primary_reason = f"high anomaly score ({risk_score:.3f}) with no single extreme feature"
        
        # Add Department activity as CONTEXT (not part of model explanation)
        context_parts = []
        if 'adjustment_count' in row and row['adjustment_count'] > 2:
            context_parts.append(f"{int(row['adjustment_count'])} adjustments (Department activity)")
        if 'contact_attempts' in row and row['contact_attempts'] > 5:
            context_parts.append(f"{int(row['contact_attempts'])} contact attempts (communication)")
        
        if context_parts:
            full_reason = f"{primary_reason} [context: {'; '.join(context_parts)}]"
        else:
            full_reason = primary_reason
        
        explanations.append({
            'case_id': str(case_id),
            'rank': int(row.get('rank', idx + 1)),
            'risk_score': float(row.get('risk_score_normalized', row.get('risk_score', 0.5))),
            'reason': full_reason,
            'top_features': top_features  # Include for transparency
        })
    
    logger.info(f"Explanations generated for {len(explanations)} cases")
    return explanations


def generate_simple_explanation(row):
    """
    Generate a simple explanation for a single case.
    Used as fallback or for ad-hoc explanations.
    
    Args:
        row: A single row from a DataFrame
    
    Returns:
        String explanation
    """
    # Check if there's already a reason
    if 'reason' in row and pd.notna(row['reason']):
        return row['reason']
    
    # Check model features
    reason_parts = []
    
    if 'award_to_needs_ratio' in row:
        ratio = row['award_to_needs_ratio']
        if ratio > 0.8:
            reason_parts.append(f"award is {ratio*100:.0f}% of needs (high)")
        elif ratio < 0.2:
            reason_parts.append(f"award is {ratio*100:.0f}% of needs (low)")
    
    if 'payment_mismatch' in row and row['payment_mismatch'] > 0.3:
        reason_parts.append(f"payment mismatch of {row['payment_mismatch']:.2f}")
    
    if 'months_since_review' in row and row['months_since_review'] > 12:
        reason_parts.append(f"{int(row['months_since_review'])} months since review")
    
    if 'transfer_ratio' in row and row['transfer_ratio'] > 0.8:
        reason_parts.append(f"high transfer ratio ({row['transfer_ratio']*100:.0f}%)")
    
    if not reason_parts:
        reason_parts.append("moderate risk indicators")
    
    return "; ".join(reason_parts)


def get_feature_importance(case_row, feature_cols=None):
    """
    Calculate feature importance for a single case.
    Uses z-scores relative to population.
    
    Args:
        case_row: A single row from a DataFrame
        feature_cols: List of feature columns to consider
    
    Returns:
        Dictionary with feature names and their z-scores
    """
    if feature_cols is None:
        feature_cols = MODEL_FEATURES
    
    importance = {}
    for col in feature_cols:
        if col in case_row:
            # We can't compute z-score without population stats
            # This is a placeholder for future enhancement
            importance[col] = 0.0
    
    return importance