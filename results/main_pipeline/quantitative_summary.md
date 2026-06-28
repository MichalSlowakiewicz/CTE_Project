# 📊 Quantitative Summary Metrics for Academic Report
*(Generated for representative Background Size N=128)*

This document provides aggregated hard numbers across all 8 models to directly quote in your thesis.

## 1. RQ1: Approximation Accuracy (MAE at N=128)

> Does CTE achieve lower SHAP approximation error than random sampling at the same background size?

### Across ALL 8 Models (Global Average)
- **Global Feature Importance:** CTE is on average **+36.1%** more accurate than Random.
- **Local (Per-Sample) Explanation:** CTE is on average **+37.2%** more accurate than Random.

### By Model Family
**Tree-based Ensembles (XGBoost, RF, LightGBM, CatBoost):**
- Global MAE Improvement: **+29.9%**
- Local MAE Improvement: **+23.0%**

**Linear Models (LogReg, Ridge, Lasso, SVC):**
- Global MAE Improvement: **+42.3%**
- Local MAE Improvement: **+51.3%**

---

## 2. RQ2: Temporal Feature Importance Tracking (N=128)

> Do feature importances shift across chronological test periods, and does a fixed CTE background track the shifts better than Random?

### Summary
- **CTE wins in 8/8 models** (0 losses)
- Average Δ: **+36.4%** across all models
- Tree models avg: **+26.6%** | Linear models avg: **+46.1%**

### Per-Model Breakdown

| Model | CTE vs Random Δ% | Winner |
|---|---|---|
| XGBoost | +24.8% | CTE |
| Random Forest | +13.9% | CTE |
| LightGBM | +31.5% | CTE |
| CatBoost | +36.3% | CTE |
| LogReg | +35.2% | CTE |
| Ridge | +49.9% | CTE |
| Lasso | +51.4% | CTE |
| SVC | +47.8% | CTE |

---

## 3. RQ3: Train vs Validation Feature Drift (N=128)

> Do SHAP values differ between train and val data, and does CTE track this shift?

### Per-Model Error Comparison

| Model | CTE (train) | Rand (train) | Δ% train | CTE (val) | Rand (val) | Δ% val |
|---|---|---|---|---|---|---|
| XGBoost | 2.10e-04 | 2.57e-04 | +18.5% | 2.12e-04 | 2.85e-04 | +25.4% |
| Random Forest | 1.27e-04 | 1.53e-04 | +16.7% | 1.07e-04 | 1.32e-04 | +19.1% |
| LightGBM | 1.24e-04 | 1.63e-04 | +24.2% | 1.10e-04 | 1.84e-04 | +40.1% |
| CatBoost | 1.48e-04 | 1.67e-04 | +11.5% | 1.49e-04 | 2.18e-04 | +31.4% |
| LogReg | 2.28e-05 | 4.72e-05 | +51.6% | 4.91e-05 | 6.69e-05 | +26.5% |
| Ridge | 5.31e-04 | 1.15e-03 | +53.8% | 4.41e-04 | 8.36e-04 | +47.3% |
| Lasso | 9.79e-04 | 2.23e-03 | +56.0% | 8.64e-04 | 1.69e-03 | +48.8% |
| SVC | 2.75e-04 | 5.59e-04 | +50.9% | 2.66e-04 | 4.95e-04 | +46.3% |

---
## Methodology Notes
- **Ground truth**: SHAP computed with full training set background (109,341 points)
- **CTE**: Reproducible coreset built via Compress++ → KT; ensemble of 30 seeds for variability estimation
- **Random**: Mean of 30 independent random subsamples ± 1σ
- **Metric**: MAE = Mean Absolute Error of feature importances vs ground truth
- **Both CTE and Random**: 30 independent repetitions per size for fair comparison
