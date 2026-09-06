# Finding: the Day-2 penalty re-injects `contact_attempts` with an inverted sign

**Status:** Verified against the actual data pack (4,200 cases). Not yet fixed in the
submitted model — documented here per the organizers' note to keep the repository as
submitted and build on it.

## The claim (from Brite Spark feedback)

> "Your Day-2 penalty quietly re-injects `contact_attempts` with an inverted sign —
> worth checking, because it may not do what you intend."

## What's actually happening

The Day-2 change had two parts:

1. **`src/ml_ranker.py`** — correctly removes `contact_intensity` and
   `adjustment_intensity` from the Isolation Forest's feature list. This part works
   exactly as documented.
2. **`src/heuristic_ranker.py::apply_investigator_feedback()`** — runs *after* the ML
   ranker, regardless of which ranker produced the score. It computes a
   `dept_activity_score` from raw `contact_attempts` and `adjustment_count`, turns it
   into a penalty (up to 30% of the normalized score), and **subtracts** it from every
   case's final score.

So `contact_attempts` was removed from the model's *inputs*, but it re-enters the
pipeline through this second step — just as something that lowers a case's score
instead of raising it.

## Why that matters: contact_attempts is not evenly distributed

```
                     contact_attempts  adjustment_count  contact_intensity
language_preference
English                         1.827             0.829              0.281
Other                           4.162             1.334              0.661
Spanish                         4.251             1.275              0.650
```

Other/Spanish speakers average roughly **2.3x** the contact attempts of English
speakers — almost certainly because it takes more attempts to reach someone across a
language barrier, not because their cases are riskier.

## The inversion

The same signal (`contact_attempts`, elevated for Other/Spanish speakers) produces
**opposite** effects depending on where it enters the pipeline:

| Where it enters | Effect on Other/Spanish speakers | Result |
|---|---|---|
| **Diagnostic**: restored as an Isolation Forest feature | Treated as an anomaly signal — unusual values get flagged | **Over**-flagged (2.54x, "Other" jumps to 5/20) |
| **Submitted model**: used in the post-hoc penalty | Subtracted from score — higher contact_attempts = bigger deduction | **Under**-flagged ("Other": 0/20, despite being 9.8% of cases) |

Same variable, same root cause, opposite direction — hence "inverted sign." The Day-2
fix didn't remove `contact_attempts`' influence on the ranking; it flipped which way
that influence points.

## Why this is arguably the worse failure mode

The submitted model reports `language_preference` as "Fair" (max disparity 1.15x)
because the fairness report only measures *over*-representation prominently. The
0/20 under-flagging of "Other" speakers is noted in the report as
"under-representation," but it's easy to read past — a clean-looking "Fair" status
can mask a group being systematically pushed out of review, not because they're
low-risk, but because the harder-to-reach a case is, the more it gets penalized.

Under-flagging genuinely at-risk cases in a population that's already harder to
reach is a real service-delivery problem, not just a statistics footnote.

## What this does *not* mean

- The submitted model's numbers are still exactly as reported — nothing here changes
  `ranked_list.csv`, `fairness_report.txt`, or the top-3 cases. This is an
  explanation of a mechanism, not a correction to a wrong result.
- This isn't a code crash or a data error — the arithmetic in
  `apply_investigator_feedback()` does exactly what it's written to do. The issue is
  that what it's written to do doesn't fully achieve what the Day-2 comment says it's
  for ("Department activity != fraud").

## Possible fix (not applied — repo kept as submitted per organizer guidance)

Compute the penalty from `contact_attempts` and `adjustment_count` **relative to each
case's own language/tenure group's median**, rather than against the full population.
That would penalize genuinely unusual Department activity within a group, without
penalizing a group simply because contacting them takes more attempts on average.
