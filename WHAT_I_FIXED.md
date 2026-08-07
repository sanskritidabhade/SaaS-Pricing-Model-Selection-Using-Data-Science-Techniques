# SaaS Pricing Project — Model Audit & Corrections

A short, honest record of defects I found in my first modeling pass and how I
corrected them. Written so I can walk an interviewer through it confidently.

## Why this document exists

The first version of the pipeline produced impressive-looking numbers — a
revenue model with R²=1.000 and a churn AUC of 0.940. Those are exactly the kind
of results that should make an analyst *suspicious*, not proud. On review, four
of them were artifacts. I rebuilt the pipeline to be honest. The corrected
numbers are lower, and that's the point.

## The five defects and the fixes

**1. Revenue model leakage (R²=1.000 → 0.967).**
The original set `price`, `mrr`, and `mrr_amount` to the same column, then
trained a model to predict `mrr` using a feature list that included `mrr`. It was
predicting a value from itself, so the "forecast" was perfect and meaningless.
Fix: model ARPU from genuinely independent drivers (plan price, discount, margin,
channel, region, contract length), with an explicit leakage guard that asserts no
target-derived column is in the feature set, plus a per-feature correlation check.
Honest cross-validated R² is 0.968 ± 0.001.

**2. Collinear churn features (AUC 0.940 → 0.889).**
The churn model used `mrr` and `arr` as separate features, but `arr = 12 × mrr`,
so they're perfectly redundant — inflating apparent performance. Fix: drop the
duplicate, and exclude the churn-rate column itself from the features so the model
isn't reading the answer. Honest AUC is 0.889.

**3. A fabricated 69% churn rate.**
The recommendation claimed to "maintain churn near the acceptable 20%" while the
underlying number was 68.9%. That came from rolling 5,000 *subscription* records
(multiple per customer) up into a customer-level rate incorrectly. Fix: use the
customer-level monthly churn that actually exists in the data — mean ≈ 5.3%,
which is realistic for SaaS and actually comparable to a 20% guardrail.

**4. Circular LTV (LTV/CAC 300+ → ~33).**
Original LTV = `mrr × (days_since_signup / 30)`, so a customer's "value" grew with
how long ago they signed up rather than their economics — producing LTV/CAC ratios
over 300 and forcing nonsensical segment thresholds. Fix: the standard formula,
LTV = ARPU × gross_margin × (1 / monthly_churn). Ratios land around 33 — healthy,
believable, and consistent across segments.

**5. The optimisation inherited all of it.**
The "$1,753.88 optimal price" and "−93.4% LTV/CAC improvement" were computed from
the leaked revenue model and the fabricated churn. The negative "improvement" was
the giveaway: the recommendation was worse on its own headline metric. Fix: rerun
the scenario analysis on the honest elasticity (−1.03, roughly unit-elastic) with
a real LTV/CAC ≥ 3.0 guardrail, and present the result explicitly as a *modelled
scenario under stated assumptions*, not a point forecast.

## The 30-second interview version

> "My first pass hit R²=1.000 on the revenue model. A perfect R² is almost always
> a leakage tell, so I dug in — I'd accidentally fed the target back in as a
> feature. Fixing that dropped it to a genuine 0.97. While I was in there I found
> the churn number was rolled up wrong (69% instead of ~5%) and my LTV formula was
> circular. I rebuilt the whole thing to be honest. The corrected numbers are less
> flashy, but I'd rather present something I can defend than something that falls
> apart under questioning."

That story demonstrates the single most valuable analyst instinct: being
skeptical of your own too-good results.

## Files

- `02_modeling_CORRECTED.ipynb` — the rebuilt, leakage-free pipeline (runs top to bottom)
- `fixed_modeling.py` — same logic as a plain script
- `business_narrative_CORRECTED.txt` — internally consistent executive summary
- `WHAT_I_FIXED.md` — this document
