# Compress Then Explain (CTE) — Ecom Offers Research Pipeline

Implementation of the **Compress Then Explain (CTE)** paradigm using Kernel Thinning
for accelerated SHAP explanations on temporal TabReD datasets.

The pipeline empirically demonstrates that geometrically compressed coresets (CTE)
produce lower-error SHAP explanations than i.i.d. random sampling — measured via
**Mean Absolute Error (MAE)** against a Gold Standard Ground Truth backed by the
full 16,488-sample training set.

---

## 🗂 Project Structure

```
experiments/
  prepare_cache.py     # Step 1: build model + coreset cache
  rq1_efficiency.py    # RQ1: efficiency vs. fidelity (MAE)
  rq2_drift.py         # RQ2: concept drift over temporal windows
  rq3_train_val.py     # RQ3: train vs. validation discrepancy
evaluation/
  metrics.py           # mae_shap, spearman, pearson, top_k_overlap
  visualize.py         # generates results/*.png charts
results/               # JSON results + publication-ready PNG plots
academic_discussion.md # empirical findings and methodological notes
```

---

## 🚀 Getting Started

### 1. Installation
```bash
pip install -r requirements.txt
```

> 💡 The preprocessed dataset (`data/ecom-offers/`) and pre-computed coresets
> (`data/cache/`) are **already included** in this repository. No data download
> or preprocessing is required.

### 2. Run the Experiments (immediately, no setup needed)
```bash
python experiments/rq1_efficiency.py   # ~2 min
python experiments/rq2_drift.py        # ~2 min
python experiments/rq3_train_val.py    # ~1 min
```

### 3. Generate Plots
```bash
python evaluation/visualize.py
```
Charts are saved to `results/`.

### ♻️ Optional: Regenerate Cache from Scratch
If you want to retrain the model or recompute coresets:
```bash
python experiments/prepare_cache.py
```
> ⚠️ This takes ~10–15 minutes (Kernel Thinning over 16k samples).

---

## 📊 Key Results (RQ1 — Global MAE, lower is better)

| Background Size | CTE MAE  | Random MAE | CTE Advantage |
|-----------------|----------|------------|---------------|
| 10              | 0.00005  | 0.00008    | 1.6×          |
| 50              | 0.00004  | 0.00004    | ≈ tie         |
| 100             | 0.00003  | 0.00005    | 1.7×          |
| 200             | 0.00003  | 0.00007    | **2.3×**      |

See `academic_discussion.md` for full analysis and methodological notes.

---

## ⚙️ Technical Notes

- **Explainer**: `shap.TreeExplainer` with `feature_perturbation="interventional"`
- **Ground Truth**: full 16,488-sample training set as reference distribution
- **Fidelity metric**: MAE of mean-absolute SHAP values vs. Ground Truth
- **Model**: XGBoost with `scale_pos_weight` for class imbalance + early stopping
