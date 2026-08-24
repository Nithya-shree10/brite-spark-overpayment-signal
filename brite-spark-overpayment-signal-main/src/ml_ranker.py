"""
Unsupervised ML ranking using Isolation Forest.
UPDATED: Day 2 change - removed Department activity features (adjustment_intensity, contact_intensity)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path
from src.utils import setup_logging

logger = setup_logging()

def train_ranker(df):
    """Train an Isolation Forest for anomaly detection."""
    logger.info("Training Isolation Forest ranker...")
    
    # UPDATED: Day 2 change - removed adjustment_intensity and contact_intensity
    # These features were causing false positives for "busy files"
    feature_cols = [
        'award_to_needs_ratio',    # Financial anomaly
        'payment_mismatch',        # Payment inconsistency
        'months_since_review',     # Long overdue
        'transfer_ratio'           # Payment method pattern
        # 'adjustment_intensity'  # REMOVED - Department activity, not fraud
        # 'contact_intensity'     # REMOVED - Department communication issue
    ]
    
    available_cols = [col for col in feature_cols if col in df.columns]
    logger.info(f"Using features: {available_cols}")
    
    X = df[available_cols].fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('iforest', IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        ))
    ])
    
    pipeline.fit(X)
    scores = pipeline.decision_function(X)
    anomaly_scores = -scores
    
    df_rank = df.copy()
    df_rank['risk_score'] = anomaly_scores
    
    # Normalize
    min_score = df_rank['risk_score'].min()
    max_score = df_rank['risk_score'].max()
    if max_score > min_score:
        df_rank['risk_score_normalized'] = (df_rank['risk_score'] - min_score) / (max_score - min_score)
    else:
        df_rank['risk_score_normalized'] = 0
    
    model_path = Path("models")
    model_path.mkdir(exist_ok=True)
    joblib.dump(pipeline, model_path / "anomaly_ranker.pkl")
    logger.info(f"Model saved to {model_path / 'anomaly_ranker.pkl'}")
    
    return pipeline, available_cols, df_rank

def predict_and_rank(model, feature_cols, df):
    """Score and rank cases using trained model."""
    logger.info("Generating anomaly scores and rankings...")
    
    X = df[feature_cols].fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    
    scores = model.decision_function(X)
    anomaly_scores = -scores
    
    df_rank = df.copy()
    df_rank['risk_score'] = anomaly_scores
    
    # Normalize
    min_score = df_rank['risk_score'].min()
    max_score = df_rank['risk_score'].max()
    if max_score > min_score:
        df_rank['risk_score_normalized'] = (df_rank['risk_score'] - min_score) / (max_score - min_score)
    else:
        df_rank['risk_score_normalized'] = 0
    
    df_rank = df_rank.sort_values('risk_score', ascending=False)
    df_rank['rank'] = range(1, len(df_rank) + 1)
    
    logger.info(f"Ranking complete. Top anomaly score: {df_rank['risk_score'].max():.4f}")
    return df_rank