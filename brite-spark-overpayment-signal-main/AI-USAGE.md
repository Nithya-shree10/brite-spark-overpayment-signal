# AI Usage Disclosure

**Project:** C6606 – Overpayment Signal  
**Date:** 2026-08-23

**Tools Used:** GitHub Copilot (code completion) & ChatGPT (architectural brainstorming).

**Scope of AI Assistance:**
- **Scaffolding:** Initial boilerplate for `src/utils.py` (logging, file I/O) and `main.py` structure.
- **Debugging:** Helped identify the `top_20` sorting logic error and suggested the use of Isolation Forest parameters (`contamination`, `random_state`) to stabilize rankings.
- **Explanation Generation:** Provided the template logic for computing z-scores and percentiles per feature, which I then manually verified against the population distributions.
- **Fairness Template:** Suggested the grouping logic for categorical columns.

**Human Oversight:**
- All feature selections, final model parameters, business logic (e.g., `payment_mismatch` formula), and policy decisions (e.g., choosing not to over-correct fairness) were made entirely by me.
- Every output file (`explanations.json`, `fairness_report.txt`) was manually reviewed for contextual accuracy before finalizing.

**Attribution:** AI was used as an accelerator, not as a substitute for domain judgment. All final decisions are my own.