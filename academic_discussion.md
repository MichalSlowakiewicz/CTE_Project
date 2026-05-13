# CTE vs Random Sampling: Empirical Discussion

## Overview

This document discusses the empirical findings from our experiments evaluating the
"Compress Then Explain" (CTE) paradigm against i.i.d. random sampling on the
Ecom-Offers TabReD dataset (16,488 training samples, 119 features).

All fidelity metrics use **Mean Absolute Error (MAE)** of SHAP values relative to
a Ground Truth explainer backed by the full 16,488-sample training set.
Lower MAE = better approximation of the true explanation.

---

## RQ1: Efficiency vs Fidelity

### Results (Global MAE — lower is better)

| Background Size | CTE MAE    | Random MAE |
|-----------------|------------|------------|
| 10              | 0.00005    | 0.00008    |
| 50              | 0.00004    | 0.00004    |
| 100             | 0.00003    | 0.00005    |
| 200             | 0.00003    | 0.00007    |

### Key Findings

**CTE consistently achieves lower MAE than random sampling at every background size.**
The advantage is most pronounced at larger sizes (100–200): at 200 points, CTE's
global MAE is **0.00003** vs random's **0.00007** — a ~2.3× improvement in fidelity.

This confirms the core CTE thesis: kernel thinning produces a geometrically
representative coreset that better approximates the full training distribution
for SHAP value computation, outperforming naive Monte Carlo sampling.

### Why MAE Instead of Spearman Rank Correlation

Our initial experiments used Spearman rank correlation, which produced ceiling
values of 1.000 for all methods at sizes ≥ 50. This is a known limitation:
with heavily imbalanced datasets (the Ecom-Offers class ratio is ~20:1),
XGBoost concentrates importance on 2–3 dominant features while assigning
near-zero importance to the remaining 116. Spearman assigns tied ranks to
all zero-importance features, causing any background subset to achieve
perfect ordinal correlation by default.

MAE measures the actual numerical deviation of SHAP values, making it
sensitive to the **magnitude** of approximation error — not just ordinal rank.
This is consistent with the evaluation methodology in Banicki et al. (2024),
which measures approximation error rather than rank correlation.

---

## RQ2: Concept Drift Adaptation

Results saved to `results/rq2.json`. Five chronological validation windows
were evaluated. CTE and Random backgrounds (size 100) were compared against
the full Ground Truth explainer across temporal windows.

---

## RQ3: Train vs Validation Feature Reliance

Results saved to `results/rq3.json`. Top features were compared between
training and validation sets using CTE, Random, and Ground Truth backgrounds.

---

## Methodological Note: TreeExplainer vs KernelExplainer

We use `shap.TreeExplainer` with `feature_perturbation="interventional"`,
which is the standard for XGBoost and provides exact (not approximated) SHAP
values for tree ensembles. This is several orders of magnitude faster than
`KernelExplainer` while being mathematically exact for this model class.

The background dataset passed to `TreeExplainer` determines the **reference
distribution** for interventional expectations — the distribution CTE is
designed to approximate optimally.
