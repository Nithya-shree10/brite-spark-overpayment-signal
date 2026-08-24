#!/usr/bin/env python
"""
Main orchestrator for Problem 6: The Overpayment Signal.
UPDATED: Day 2 change - incorporates investigator feedback to reduce false positives
"""

import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.utils import setup_logging, ensure_directory, save_output, save_text, save_json
from src.data_loader import load_cases, load_payments, validate_cases, validate_payments
from src.data_joiner import join_data
from src.feature_engineer import engineer_features
from src.heuristic_ranker import rank_by_heuristic, apply_investigator_feedback
from src.ml_ranker import train_ranker, predict_and_rank
from src.explainer import explain_top_k
from src.fairness import compute_fairness, generate_fairness_report

logger = setup_logging()

def main():
    logger.info("=" * 60)
    logger.info("BRITE SPARK 2026 - PROBLEM 6: THE OVERPAYMENT SIGNAL")
    logger.info("=" * 60)

    logger.info("\n[STEP 1] Loading data...")
    cases_df = load_cases("data/raw/cases.csv")
    payments_df = load_payments("data/raw/payments.csv")

    logger.info("\n[STEP 2] Validating data...")
    cases_df = validate_cases(cases_df)
    payments_df = validate_payments(payments_df)

    logger.info("\n[STEP 3] Joining cases and payments...")
    joined_df = join_data(cases_df, payments_df)

    logger.info("\n[STEP 4] Engineering features...")
    features_df = engineer_features(joined_df)

    logger.info("\n[STEP 5] Generating rankings...")
    try:
        model, feature_cols, scored_df = train_ranker(features_df)
        logger.info("ML ranker trained successfully.")
        ranked_df = predict_and_rank(model, feature_cols, scored_df)
        using_ml = True
        logger.info("Rankings generated using ML (Isolation Forest).")
    except Exception as e:
        logger.error(f"ML ranking failed: {e}")
        logger.info("Falling back to heuristic ranker...")
        ranked_df = rank_by_heuristic(features_df)
        using_ml = False
        logger.info("Rankings generated using heuristic fallback.")

    logger.info("\n[STEP 5b] Applying investigator feedback (Day 2 change)...")
    ranked_df = apply_investigator_feedback(ranked_df)
    logger.info("Investigator feedback applied successfully.")

    logger.info("\n[STEP 6] Generating explanations...")
    explanations = explain_top_k(ranked_df, k=20)

    logger.info("\n[STEP 7] Computing fairness...")
    fairness_results = compute_fairness(ranked_df)
    fairness_report = generate_fairness_report(fairness_results)

    logger.info("\n[STEP 8] Saving outputs...")

    top_20 = ranked_df.head(20).copy()
    output_cols = ['case_id', 'rank', 'risk_score', 'risk_score_normalized',
                   'monthly_award', 'district', 'age_band', 'language_preference', 'tenure', 'status']
    existing_cols = [col for col in output_cols if col in top_20.columns]
    save_output(top_20[existing_cols], "ranked_list.csv")
    logger.info("Saved ranked_list.csv")

    save_text(fairness_report, "fairness_report.txt")
    logger.info("Saved fairness_report.txt")

    save_json(explanations, "explanations.json")
    logger.info("Saved explanations.json")

    limitations = generate_limitations_statement()
    save_text(limitations, "model_limitations.txt")
    logger.info("Saved model_limitations.txt")

    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE!")
    logger.info("=" * 60)
    logger.info("\nOutput files:")
    logger.info("  - output/ranked_list.csv (top 20 cases)")
    logger.info("  - output/fairness_report.txt")
    logger.info("  - output/explanations.json")
    logger.info("  - output/model_limitations.txt")
    logger.info(f"\nModel used: {'ML (Isolation Forest)' if using_ml else 'Heuristic (fallback)'}")
    logger.info("=" * 60)

    return 0

def generate_limitations_statement():
    """Generate the required statement with specific technical limitations."""
    lines = []
    lines.append("=" * 60)
    lines.append("MODEL LIMITATIONS AND BOUNDARIES")
    lines.append("=" * 60)
    lines.append("")
    lines.append("## GOVERNANCE POLICY - What This Model Must Never Do")
    lines.append("")
    lines.append("### AUTOMATIC DECISIONS")
    lines.append("This model must NEVER be used to automatically approve, deny, suspend,")
    lines.append("or terminate any benefit or service. Every case flagged by this model")
    lines.append("requires human review.")
    lines.append("")
    lines.append("### SOLE BASIS FOR ACTION")
    lines.append("This model must NEVER be the sole basis for any action. It is a triage")
    lines.append("tool to help investigators prioritize their work, not a replacement")
    lines.append("for professional judgment.")
    lines.append("")
    lines.append("### FINANCIAL DECISIONS")
    lines.append("This model must NEVER be used to make financial decisions or calculate")
    lines.append("benefit amounts.")
    lines.append("")
    lines.append("### DISCRIMINATORY USE")
    lines.append("This model must NEVER be used to target specific demographic groups.")
    lines.append("If bias is detected, the model must be retrained.")
    lines.append("")
    lines.append("### EXTERNAL USE")
    lines.append("This model must NEVER be shared with or used by external organizations")
    lines.append("without explicit approval.")
    lines.append("")
    lines.append("=" * 60)
    lines.append("")
    lines.append("## TECHNICAL LIMITATIONS - Known Gaps in This Implementation")
    lines.append("")
    lines.append("### 1. EXPLANATIONS ARE HEURISTIC, NOT MODEL-DERIVED")
    lines.append("The plain-language explanations identify which of the 4 model features")
    lines.append("is most extreme for each case. However, Isolation Forest does not")
    lines.append("provide per-feature attribution the way tree classifiers do. The")
    lines.append("explanations are a reasonable approximation, not a true derivation")
    lines.append("from model internals.")
    lines.append("")
    lines.append("### 2. FAIRNESS METRICS MEASURE SELECTION RATE ONLY")
    lines.append("The fairness analysis compares who gets flagged (selection rate) across")
    lines.append("demographic groups. It does NOT compare error rates (false positive/")
    lines.append("false negative rates by group) because we do not have ground-truth")
    lines.append("labels. We cannot distinguish justified detection from bias.")
    lines.append("")
    lines.append("### 3. SMALL SAMPLE SIZE IN FLAGGED SET (n=20)")
    lines.append("All fairness comparisons are based on only 20 flagged cases. A single")
    lines.append("case moving between groups changes a group's flagged share by 5")
    lines.append("percentage points. Per-group disparities should be read as directional")
    lines.append("signals, not precise measurements.")
    lines.append("")
    lines.append("### 4. UNDER-REPRESENTATION AND BIAS INVESTIGATION")
    lines.append("In our submitted model, 'Other' language speakers (9.8% of population)")
    lines.append("received 0 of 20 flags. A diagnostic check (see DECISIONS.md and")
    lines.append("output/diagnostic_bias_investigation.txt) found that restoring two")
    lines.append("features removed during the day-two fix reverses this: 'Other' speakers")
    lines.append("become the MOST over-flagged group, at 2.54x the base rate. This suggests")
    lines.append("the demographic signal in this data runs substantially through")
    lines.append("administrative-activity features we removed for a separate, legitimate")
    lines.append("reason. See DECISIONS.md for full discussion.")
    lines.append("")
    lines.append("### 5. DATA VALIDATION UNTESTED")
    lines.append("The synthetic data provided had no missing values or quality issues,")
    lines.append("so the validation/cleaning logic was not stress-tested. In production")
    lines.append("with real data, more robust cleaning would be needed.")
    lines.append("")
    lines.append("=" * 60)
    lines.append("")
    lines.append("## WHAT WE WOULD FIX FIRST WITH MORE TIME")
    lines.append("")
    lines.append("1. Outcome Labels: Obtain ground-truth labels to compute false")
    lines.append("   positive/false negative rates by demographic group.")
    lines.append("")
    lines.append("2. Larger Flagged Set: Increase to top 50 cases for more stable")
    lines.append("   fairness statistics.")
    lines.append("")
    lines.append("3. Model-Derived Explanations: Replace heuristic explanations with")
    lines.append("   model-agnostic attribution methods (LIME, SHAP).")
    lines.append("")
    lines.append("4. Resolve the Bias Investigation Tension: determine whether")
    lines.append("   adjustment_intensity and contact_intensity reflect legitimate")
    lines.append("   Department workload or investigator bias, so they can be safely")
    lines.append("   restored, redesigned, or confidently left out. See DECISIONS.md.")
    lines.append("")
    lines.append("=" * 60)
    lines.append("")
    lines.append("## WHAT WOULD NEED TO BE TRUE BEFORE DEPLOYMENT")
    lines.append("")
    lines.append("1. Fairness validation on real-world data (not synthetic)")
    lines.append("2. Independent audit of model outputs")
    lines.append("3. Clear appeals process for flagged cases")
    lines.append("4. Regular bias monitoring and retraining schedule")
    lines.append("5. Transparent reporting of false positive rates")
    lines.append("6. Human oversight at all stages")
    lines.append("7. Legal and ethical review")
    lines.append("8. Stakeholder consultation with affected communities")
    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)

if __name__ == "__main__":
    sys.exit(main())