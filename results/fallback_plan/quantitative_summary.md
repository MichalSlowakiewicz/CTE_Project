# 📊 Quantitative Summary Metrics for Academic Report
*(Generated for representative Background Size N=128)*

This document provides aggregated hard numbers across all 8 models to directly quote in your thesis.

## 1. RQ1: Approximation Accuracy (MAE at N=128)

> Does CTE achieve lower SHAP approximation error than random sampling at the same background size?

### Across ALL 8 Models (Global Average)
- **Global Feature Importance:** CTE is on average **+38.5%** more accurate than Random.
- **Local (Per-Sample) Explanation:** CTE is on average **+37.7%** more accurate than Random.

### By Model Family
**Tree-based Ensembles (XGBoost, RF, LightGBM, CatBoost):**
- Global MAE Improvement: **+30.6%**
- Local MAE Improvement: **+24.2%**

**Linear Models (LogReg, Ridge, Lasso, SVC):**
- Global MAE Improvement: **+46.5%**
- Local MAE Improvement: **+51.2%**

---

## 2. RQ2: Temporal Feature Importance Tracking (N=128)

> Do feature importances shift across chronological test periods, and does a fixed CTE background track the shifts better than Random?

### Summary
- **CTE wins in 8/8 models** (0 losses)
- Average Δ: **+37.0%** across all models
- Tree models avg: **+27.6%** | Linear models avg: **+46.3%**

### Per-Model Breakdown

| Model | CTE vs Random Δ% | Winner |
|---|---|---|
| XGBoost | +31.6% | CTE |
| Random Forest | +14.1% | CTE |
| LightGBM | +34.8% | CTE |
| CatBoost | +29.8% | CTE |
| LogReg | +31.7% | CTE |
| Ridge | +55.2% | CTE |
| Lasso | +51.6% | CTE |
| SVC | +46.7% | CTE |

---

## 3. RQ3: Train vs Validation Feature Drift (N=128)

> Do SHAP values differ between train and val data, and does CTE track this shift?

### Per-Model Error Comparison

| Model | CTE (train) | Rand (train) | Δ% train | CTE (val) | Rand (val) | Δ% val |
|---|---|---|---|---|---|---|
| XGBoost | 2.04e-04 | 2.57e-04 | +20.6% | 2.28e-04 | 2.85e-04 | +20.0% |
| Random Forest | 1.34e-04 | 1.53e-04 | +12.0% | 1.27e-04 | 1.32e-04 | +3.5% |
| LightGBM | 1.02e-04 | 1.63e-04 | +37.6% | 9.91e-05 | 1.84e-04 | +46.3% |
| CatBoost | 1.57e-04 | 1.67e-04 | +5.9% | 1.07e-04 | 2.18e-04 | +50.7% |
| LogReg | 3.12e-05 | 4.72e-05 | +34.0% | 4.65e-05 | 6.69e-05 | +30.5% |
| Ridge | 4.75e-04 | 1.15e-03 | +58.7% | 3.79e-04 | 8.36e-04 | +54.7% |
| Lasso | 9.97e-04 | 2.23e-03 | +55.2% | 8.49e-04 | 1.69e-03 | +49.6% |
| SVC | 2.93e-04 | 5.59e-04 | +47.6% | 2.62e-04 | 4.95e-04 | +47.1% |

---
## Methodology Notes
- **Ground truth**: SHAP computed with full training set background (109,341 points)
- **CTE**: Single deterministic coreset (seed=42) built via Compress++ → KT
- **Random**: Mean of 30 independent random subsamples ± 1σ
- **Metric**: MAE = Mean Absolute Error of feature importances vs ground truth
- **Limitation**: CTE is a single realization; its variance is not estimated
