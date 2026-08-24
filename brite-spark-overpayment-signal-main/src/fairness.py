"""
Fairness metrics across demographic fields.
UPDATED: Fixed top_20 selection, added counts, improved reporting,
          added under-representation detection, sample size caveat.
UPDATED: Recommendations now generated dynamically from results,
          not hardcoded, so they always match the actual run.
"""

import pandas as pd
import numpy as np
from src.utils import setup_logging

logger = setup_logging()

def compute_fairness(df, risk_col='risk_score'):
    logger.info("Computing fairness metrics...")

    demographic_fields = ['district', 'age_band', 'language_preference', 'tenure']
    results = {}

    if risk_col in df.columns:
        top_20 = df.nlargest(20, risk_col)
    else:
        logger.warning(f"Risk column '{risk_col}' not found. Using first 20 rows.")
        top_20 = df.head(20)

    total_population = len(df)

    for field in demographic_fields:
        if field not in df.columns:
            continue

        logger.info(f"Analyzing fairness for: {field}")

        pop_dist = df[field].value_counts(normalize=True)
        pop_counts = df[field].value_counts()

        flagged_dist = top_20[field].value_counts(normalize=True)
        flagged_counts = top_20[field].value_counts()

        disparity = {}
        for group in flagged_dist.index:
            pop_rate = pop_dist.get(group, 0.001)
            flagged_rate = flagged_dist.get(group, 0)
            if pop_rate > 0:
                disparity[group] = flagged_rate / pop_rate
            else:
                disparity[group] = np.inf

        max_disparity = max(disparity.values()) if disparity else 1.0

        under_represented = []
        for group in pop_counts.index:
            pop_pct = pop_counts[group] / total_population
            flagged_count = flagged_counts.get(group, 0)
            flagged_pct = flagged_count / 20 if flagged_count > 0 else 0
            if pop_pct > 0.05 and flagged_pct < 0.02:
                under_represented.append({
                    'group': group,
                    'population_pct': pop_pct,
                    'flagged_count': flagged_count,
                    'flagged_pct': flagged_pct
                })

        results[field] = {
            'population_distribution': pop_dist.to_dict(),
            'flagged_distribution': flagged_dist.to_dict(),
            'population_counts': pop_counts.to_dict(),
            'flagged_counts': flagged_counts.to_dict(),
            'disparity_ratios': disparity,
            'max_disparity': max_disparity,
            'status': (
                "Fair" if max_disparity < 1.5 else
                "Potential Bias" if max_disparity < 2.5 else
                "Significant Bias"
            ),
            'under_represented': under_represented
        }

    return results


def generate_fairness_report(results):
    lines = []
    lines.append("=" * 60)
    lines.append("FAIRNESS ANALYSIS REPORT")
    lines.append("=" * 60)
    lines.append("")

    lines.append("NOTE: Flagged set size is only 20 cases.")
    lines.append("   Per-group disparities should be read as directional signals,")
    lines.append("   not precise measurements. A 1-case shift changes a group's")
    lines.append("   flagged share by 5 percentage points.")
    lines.append("")

    for field, data in results.items():
        lines.append(f"  {field}:")
        lines.append(f"    Max Disparity: {data['max_disparity']:.2f}")
        lines.append(f"    Status: {data['status']}")

        total_pop = sum(data['population_counts'].values())
        lines.append(f"    Population counts:")
        for group, count in data['population_counts'].items():
            pct = (count / total_pop * 100)
            lines.append(f"      - {group}: {count} cases ({pct:.1f}%)")

        lines.append(f"    Flagged counts (top 20):")
        if data['flagged_counts']:
            for group, count in data['flagged_counts'].items():
                lines.append(f"      - {group}: {count} cases")
        else:
            lines.append(f"      - No groups flagged")

        lines.append(f"    Under-represented groups (population >5% but <2% flags):")
        if data['under_represented']:
            for item in data['under_represented']:
                lines.append(f"      - {item['group']}: {item['flagged_count']} flags ({item['population_pct']*100:.1f}% of population)")
        else:
            lines.append(f"      - None detected")

        if data['max_disparity'] > 1.5:
            lines.append(f"    Groups with higher flag rates:")
            for group, ratio in data['disparity_ratios'].items():
                if ratio > 1.5:
                    lines.append(f"      - {group}: {ratio:.2f}x higher")
        else:
            lines.append(f"    No significant over-representation detected")
        lines.append("")

    lines.append("=" * 60)
    lines.append("")
    lines.append("LIMITATIONS OF THIS FAIRNESS ANALYSIS:")
    lines.append("")
    lines.append("1. SELECTION RATE ONLY: This report measures who gets flagged,")
    lines.append("   not whether those flags are correct. Without ground-truth")
    lines.append("   labels, we cannot distinguish justified detection from bias.")
    lines.append("")
    lines.append("2. SMALL SAMPLE: With only 20 flagged cases, per-group")
    lines.append("   disparities are noisy and should be read as directional")
    lines.append("   signals, not precise measurements.")
    lines.append("")
    lines.append("3. UNDER-REPRESENTATION: Some groups may receive zero or very")
    lines.append("   few flags despite meaningful population share. See the")
    lines.append("   per-field breakdown above for specifics.")
    lines.append("")
    lines.append("=" * 60)
    lines.append("")
    lines.append("RECOMMENDATIONS:")
    lines.append("")

    rec_num = 1

    elevated = []
    for field, data in results.items():
        for group, ratio in data.get("disparity_ratios", {}).items():
            if ratio > 1.5:
                elevated.append(f"{group} ({field}, {ratio:.2f}x)")
    if elevated:
        lines.append(f"{rec_num}. Elevated flag rates observed for: {', '.join(elevated)}.")
        lines.append("   These groups warrant closer investigator attention.")
        lines.append("")
        rec_num += 1

    under_rep = []
    for field, data in results.items():
        for item in data.get("under_represented", []):
            under_rep.append(f"{item['group']} ({field}, {item['flagged_count']} flags, "
                              f"{item['population_pct']*100:.1f}% of population)")
    if under_rep:
        lines.append(f"{rec_num}. Investigate under-flagging for: {', '.join(under_rep)}.")
        lines.append("   These groups received disproportionately few flags relative")
        lines.append("   to population share.")
        lines.append("")
        rec_num += 1

    lines.append(f"{rec_num}. This model should NOT be used for automatic decisions.")
    lines.append("")
    rec_num += 1
    lines.append(f"{rec_num}. Human review is mandatory for all flagged cases.")
    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)