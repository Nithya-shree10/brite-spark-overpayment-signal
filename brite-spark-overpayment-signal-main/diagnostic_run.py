#!/usr/bin/env python
"""
DIAGNOSTIC (Day-2, post-hoc): Bias investigation.

Reconstructed diagnostic — the original script referenced in DECISIONS.md was
never committed to the repo. This rebuilds it from the documented method:
retrain an otherwise-identical Isolation Forest with `adjustment_intensity`
and `contact_intensity` restored (the two features removed during the Day-2
false-positive fix), and recompute fairness on the result.

This is READ-ONLY diagnostics. It does not change the submitted model,
submitted outputs, or models/anomaly_ranker.pkl.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.utils import setup_logging, ensure_directory, save_text
from src.data_loader import load_cases, load_payments, validate_cases, validate_payments
from src.data_joiner import join_data
from src.feature_engineer import engineer_features
from src.fairness import compute_fairness, generate_fairness_report

logger = setup_logging()

def train_diagnostic_ranker(df):
    """Same as src/ml_ranker.train_ranker, but with the two removed features restored."""
    feature_cols = [
        'award_to_needs_ratio',
        'payment_mismatch',
        'months_since_review',
        'transfer_ratio',
        'adjustment_intensity',   # RESTORED for diagnostic purposes only
        'contact_intensity',      # RESTORED for diagnostic purposes only
    ]
    available_cols = [c for c in feature_cols if c in df.columns]
    logger.info(f"[DIAGNOSTIC] Using features: {available_cols}")

    X = df[available_cols].fillna(0).replace([np.inf, -np.inf], 0)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('iforest', IsolationForest(contamination=0.1, random_state=42, n_estimators=100))
    ])
    pipeline.fit(X)
    scores = pipeline.decision_function(X)

    df_rank = df.copy()
    df_rank['risk_score'] = -scores
    df_rank = df_rank.sort_values('risk_score', ascending=False)
    df_rank['rank'] = range(1, len(df_rank) + 1)
    return df_rank

def main():
    logger.info("=" * 60)
    logger.info("DIAGNOSTIC RUN — Bias investigation (features restored)")
    logger.info("=" * 60)

    cases_df = validate_cases(load_cases("data/raw/cases.csv"))
    payments_df = validate_payments(load_payments("data/raw/payments.csv"))
    joined_df = join_data(cases_df, payments_df)
    features_df = engineer_features(joined_df)

    diagnostic_ranked = train_diagnostic_ranker(features_df)

    fairness_results = compute_fairness(diagnostic_ranked)
    fairness_report = generate_fairness_report(fairness_results)

    header = (
        "DIAGNOSTIC RUN — adjustment_intensity + contact_intensity RESTORED\n"
        "This is NOT the submitted model. For bias-investigation purposes only.\n"
        + "=" * 60 + "\n\n"
    )

    ensure_directory("output")
    save_text(header + fairness_report, "diagnostic_bias_investigation.txt")

    lang = fairness_results.get('language_preference', {})
    logger.info(f"\nlanguage_preference max disparity (diagnostic): {lang.get('max_disparity'):.2f} "
                f"({lang.get('status')})")
    logger.info(f"language_preference flagged counts (diagnostic): {lang.get('flagged_counts')}")
    logger.info("\nSaved output/diagnostic_bias_investigation.txt")

if __name__ == "__main__":
    sys.exit(main())