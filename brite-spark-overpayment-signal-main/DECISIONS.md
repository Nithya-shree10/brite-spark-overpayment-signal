# DECISIONS.md

## What I Chose

- **Approach:** Unsupervised anomaly detection (Isolation Forest) with heuristic fallback
  - Chosen because there are **no labels** in the data
  - Isolation Forest is interpretable and handles outliers well
  
- **Final features used in the Isolation Forest model (4 total):**
  - `award_to_needs_ratio` – captures over-award intensity relative to documented need.
  - `payment_mismatch` – flags payments deviating from expected amounts (formula: `|actual - expected| / expected`).
  - `months_since_review` – time decay; older reviews increase latent risk.
  - `transfer_ratio` – proportion of funds transferred vs. retained; high values indicate leakage risk.
  
  *Why these 4?* `adjustment_intensity` and `contact_intensity` were initially included but removed after day-two investigator feedback identified them as reflecting Department administrative activity rather than resident behavior (see "How I Handled the Day-Two Change" below). A later diagnostic (see "Bias Investigation Finding" below) shows this removal had a significant, unintended side effect on fairness outcomes.

- **Explainability:** Plain-language reasons for each top 20 case
  - Each case gets a human-readable explanation
  - Explains WHY the case is suspicious, not just a score

- **Fairness:** Checked across demographic fields
  - `district` (Calder Central, Weybridge, Northgate, Ash Hill)
  - `age_band` (18-29, 30-44, 45-59, 60-74, 75+)
  - `language_preference` (English, Spanish, Other)
  - `tenure` (Private tenancy, Social tenancy, etc.)

- **Model Limitations Statement:** Created a clear boundary document
  - What the model must NEVER be used for
  - Requirements before real-world deployment

## What I Rejected

- **Supervised ML:** No labels available in the data
- **Deep Learning:** Overkill for this dataset size
- **Rule-based only:** Too simplistic; ML provides better anomaly detection
- **SHAP explanations:** Complex for non-technical investigators
- **Real-time API:** Not required; batch processing is sufficient
- **Dashboard:** Not required; CLI output is acceptable

## What I Cut for Time

- Hyperparameter tuning (used default Isolation Forest parameters)
- Ensemble methods (would improve accuracy but not required)
- Interactive dashboard (not assessed)
- Docker container (virtual environment sufficient)
- Unit tests (would add in production)
- Cross-validation (would add in production)

## What My Solution Does NOT Do

- Does NOT automatically approve/deny anything
- Does NOT make financial decisions
- Does NOT replace human judgment
- Does NOT handle real-time streaming data
- Does NOT provide confidence intervals
- Does NOT have a user interface
- Does NOT handle missing values automatically (drops rows with nulls)
- Does NOT include hyperparameter tuning
- Does NOT work with data outside the provided format

## What I Would Fix First

1. **Fairness mitigation** if bias is detected in the fairness report
2. **More sophisticated feature engineering** (e.g., month-over-month payment trends)
3. **External validation** on real data (not synthetic)
4. **Probability calibration** to improve ranking accuracy
5. **Cross-validation** for more robust evaluation
6. **Better imputation** for missing data (instead of dropping rows)
7. **Interactive dashboard** for investigators to explore flagged cases

## How I Handled the Day-Two Change

**Change Received:** Investigator feedback that the ranking was flagging "busy files" with high administrative activity (adjustments, contact attempts) as suspicious, when these were actually Department errors, not resident fraud. The investigator specifically noted: "Busy files with a lot of Departmental activity on them. That is not the same thing as a file where something is wrong."

**How I Responded:** I implemented three changes:

1. **Updated weights in the heuristic ranker:** Reduced the weight of `adjustment_score` from 0.15 to 0.05 and `contact_score` from 0.10 to 0.05. Increased the weight of financial anomaly features: `award_score` from 0.25 to 0.30 and `mismatch_score` from 0.20 to 0.25.

2. **Removed Department activity features from ML model:** Removed `adjustment_intensity` and `contact_intensity` from the Isolation Forest feature list. The ML model now focuses only on financial anomalies: `award_to_needs_ratio`, `payment_mismatch`, `months_since_review`, and `transfer_ratio`.

3. **Added post-processing penalty:** Created a new function `apply_investigator_feedback()` that penalizes cases with high "Department activity" (adjustments + contacts combined), reducing their risk score by up to 30%.

**Time to Implement:** ~25 minutes

**Lessons Learned:**

1. **Domain expertise is critical:** What looks like "signal" to a model may be "noise" to a human. Adjustments and contact attempts are Department activity, not fraud indicators.

2. **The modular design made this change easy:** I only modified three files (`heuristic_ranker.py`, `ml_ranker.py`, `main.py`) without breaking anything else.

3. **Always validate with real users:** The investigator's feedback revealed a fundamental flaw in the ranking logic that would have been impossible to catch without human review.

4. **Heuristic fallback is valuable:** The heuristic ranker allowed quick weight adjustments without needing to retrain the ML model.

## Iterative Fixes & Critical Discoveries (Day 2)

While implementing the investigator feedback and preparing the final submission, I discovered and fixed three additional issues:

1. **Top-20 Selection Bug:** The fairness/explanation code originally selected the top 20 cases with `df.nlargest(20, risk_col) if 'rank' in df.columns else df.head(20)`. When a `rank` column wasn't present, this silently fell back to the first 20 rows in whatever order the DataFrame happened to be in — not sorted by risk at all. Fixed by always sorting explicitly on `risk_score` before selecting the top 20, removing the conditional entirely.

2. **`award_ratio` Naming Bug:** Explanation-generation code checked for a column named `award_ratio`, but the model's actual feature was `award_to_needs_ratio`. This meant the check silently never fired for any case. Fixed by correcting the name and rebuilding explanations to derive from actual per-case z-scores/percentiles against the population, rather than static feature-name matching.

3. **Under-representation Discovery (Fairness Deep-Dive):** While computing fairness, we discovered that specific groups (e.g., "Other" language speakers) were significantly under-represented in the top-20 flags relative to their population share — in one run, receiving zero flags despite meaningful population share. Rather than silently ignoring this, we surfaced it explicitly in the fairness report and chose **not** to artificially re-weight the model without policy review, because forcing parity without understanding the underlying need-distribution could mask real fraud or overpayments. This caveat is stated in both the fairness report and limitations. See "Bias Investigation Finding" below for how this connects to a deeper issue.

4. **Risk Score Scale Mismatch (Day-Two Fix Bug):** `apply_investigator_feedback()` applied a penalty of up to 0.3 to `risk_score` — but for the ML (Isolation Forest) path, `risk_score` holds the *raw* anomaly score (observed max ~0.20), not the normalized [0,1] score (`risk_score_normalized`) computed separately. A penalty of up to 0.3 against a score that tops out around 0.20 meant the day-two adjustment could dominate the ranking entirely — effectively re-ranking by Department activity rather than lightly discounting it. We caught this by tracing the scale of each variable through the pipeline rather than trusting that the code "looked" correct. **Verified impact:** fixing this changed 2 of the top 3 flagged cases (previously C-32502, C-32404, C-32671; after the fix, C-32502, C-32367, C-33379). Fixed by applying the penalty to `risk_score_normalized` instead of the raw score, and keeping `risk_score` consistent with the normalized value downstream.

## Bias Investigation Finding

The problem spec states that a straightforward model on the obvious features should flag one group at roughly 3x the base rate. Our submitted model (4 features, post day-two fix) showed a maximum disparity of 1.90x (district: Ash Hill) and 1.77x (tenure: Private tenancy) — short of that figure. This prompted a targeted investigation.

**Diagnostic:** we retrained an otherwise-identical Isolation Forest model, restoring `adjustment_intensity` and `contact_intensity` — the two features removed in the day-two fix — and recomputed fairness on the result. (Code: `diagnostic_run.py`; full output: `diagnostic_bias_investigation.txt`.)

**Follow-up finding:** the day-two fix removes these two features from the ML model's
inputs, but a separate post-processing step (`apply_investigator_feedback()` in
`src/heuristic_ranker.py`) still uses raw `contact_attempts` as a penalty subtracted
from every case's score. Since Other/Spanish-speaking residents average roughly 2x the
contact attempts of English speakers, this penalty pushes them *down* the ranking —
the likely reason "Other" shows 0/20 flags in the submitted model, versus 2.54x
over-flagged in the diagnostic above. Same underlying signal, opposite direction
depending on where it enters the pipeline. See `FINDING-inverted-sign-penalty.md` for
the full write-up and a proposed (unapplied) fix.

**Result:** `language_preference` disparity jumped from **1.15x (Fair)** to **2.54x (Significant Bias)** — close to the spec's ~3x figure. More strikingly, the affected group reversed entirely:

- **Submitted model:** "Other" language speakers received **0 of 20 flags** (flagged as under-represented in our fairness report).
- **Diagnostic model:** "Other" language speakers became the **most over-flagged group**, at 2.54x the base rate.

**Interpretation:** the demographic signal in this dataset appears to run substantially through administrative-activity patterns (adjustment and contact frequency), not primarily through the core financial-anomaly features. Our day-two fix — a legitimate response to the investigator's correct observation that "busy files ≠ fraud" — had the side effect of removing the main channel carrying this signal from the ML model. This was not a deliberate choice to obscure bias; it was an unintended consequence of a defensible engineering decision made under time pressure, and we only surfaced it through a deliberate post-hoc check.

**What we did with this finding:** we disclose it here rather than resolve it. The problem spec explicitly treats fixing fairness issues as a stretch goal, not a floor requirement, and states an honest failure is worth more than an unjustified fix. Re-including `adjustment_intensity`/`contact_intensity` would restore the bias signal's visibility but reintroduce the investigator's correctly identified false-positive problem (busy files flagged as fraud). We do not have a validated way to separate "administrative activity that correlates with group membership for legitimate reasons" from "administrative activity that reflects investigator bias in who gets extra scrutiny" without ground-truth labels or domain input we don't have access to in this exercise. Resolving that tension is the single most important open question before this system could be deployed (see below).

## What I Would Need to Be True Before Deployment

1. Fairness validation on real-world data (not synthetic)
2. Independent audit of model outputs
3. Clear appeals process for flagged cases
4. Regular bias monitoring and retraining schedule
5. Transparent reporting of false positive rates
6. Human oversight at all stages
7. Legal and ethical review
8. Stakeholder consultation with affected communities
9. **Investigator feedback loop:** Regular review of top cases by investigators to catch false patterns
10. **Domain expert validation:** specifically, whether `adjustment_intensity` and `contact_intensity` reflect legitimate Department workload patterns or investigator bias in who receives extra scrutiny — see "Bias Investigation Finding" above. This is now the single most consequential open question, since it directly determines whether restoring these features to the model would reduce or worsen fairness outcomes.
