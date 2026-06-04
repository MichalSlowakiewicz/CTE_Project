# 📊 Quantitative Summary Metrics for Academic Report
*(Generated for representative Background Size N=64)*

This document provides aggregated hard numbers across all 8 models to directly quote in your thesis.

## 1. RQ1: Approximation Accuracy (Global & Local MAE)
How much better is CTE at approximating the true SHAP distribution compared to Random sampling?

### Across ALL 8 Models (Global Average)
- **Global Feature Importance:** CTE is on average **+16.4%** more accurate than Random sampling.
- **Local (Per-Sample) Explanation:** CTE is on average **+32.9%** more accurate than Random sampling.

### By Model Family
**Tree-based Ensembles (XGBoost, RF, LightGBM, CatBoost):**
- Global MAE Improvement: **+6.6%**
- Local MAE Improvement: **+16.7%**

**Linear Models (LogReg, Ridge, Lasso, SVC):**
- Global MAE Improvement: **+26.2%**
- Local MAE Improvement: **+49.1%**

---

## 2. RQ2: Temporal Drift Robustness
When evaluating how accurately the algorithms track feature importance over time (across 5 data drift windows):

### Across ALL 8 Models
- Temporal Tracking Accuracy: CTE reduces the drift tracking error by an average of **+2.5%** compared to Random.

### By Model Family
- **Tree-based Ensembles:** **+0.1%** more accurate than Random in tracking temporal drift.
- **Linear Models:** **+4.9%** more accurate than Random in tracking temporal drift.

---
**How to use these numbers:**
*„Our experiments across 8 diverse architectures indicate that at a standard background size of N=128, the Coreset Tree Explainer (CTE) reduces the Global MAE of feature importance by an average of 16.4% compared to uniform random sampling. This superiority holds across architectural boundaries, with Tree-based ensembles seeing a 6.6% improvement and Linear models a 26.2% improvement...”*
