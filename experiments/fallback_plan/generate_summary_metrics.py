"""
generate_summary_metrics.py — Quantitative summary for the academic report.

Generates hard numbers aggregated across all 8 models for RQ1, RQ2, and RQ3.
Output: results/fallback_plan/quantitative_summary.md
"""

import json
from pathlib import Path
import numpy as np


def load_json(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def calculate_improvement(baseline, new_val):
    """Positive = CTE better (lower error), negative = Random better."""
    if baseline == 0:
        return 0
    return ((baseline - new_val) / baseline) * 100


def generate_summary():
    base_dir = Path(__file__).parent.parent.parent / "results" / "fallback_plan"

    tree_models = ['xgb_tree', 'rf_tree', 'lgb_tree', 'catboost_tree']
    linear_models = ['logreg_linear', 'ridge_linear', 'lasso_linear', 'svc_linear']
    all_models = tree_models + linear_models

    pretty = {
        'xgb_tree': 'XGBoost', 'rf_tree': 'Random Forest',
        'lgb_tree': 'LightGBM', 'catboost_tree': 'CatBoost',
        'logreg_linear': 'LogReg', 'ridge_linear': 'Ridge',
        'lasso_linear': 'Lasso', 'svc_linear': 'SVC',
    }

    target_size = 128  # Representative background size

    # ── RQ1 ──
    rq1_tree_g, rq1_tree_l = [], []
    rq1_linear_g, rq1_linear_l = [], []

    for model in all_models:
        is_tree = model in tree_models
        rq1_data = load_json(base_dir / model / 'rq1.json')
        if not rq1_data:
            continue
        for exp in rq1_data['experiments']:
            if exp['size'] == target_size:
                cte_g = exp['cte']['mae_global']
                cte_l = exp['cte']['mae_local']
                if 'mae_global_mean' in exp['random']:
                    rand_g = exp['random']['mae_global_mean']
                    rand_l = exp['random']['mae_local_mean']
                else:
                    rand_g = exp['random']['mae_global']
                    rand_l = exp['random']['mae_local']
                imp_g = calculate_improvement(rand_g, cte_g)
                imp_l = calculate_improvement(rand_l, cte_l)
                if is_tree:
                    rq1_tree_g.append(imp_g)
                    rq1_tree_l.append(imp_l)
                else:
                    rq1_linear_g.append(imp_g)
                    rq1_linear_l.append(imp_l)

    def safe_mean(lst):
        return np.mean(lst) if lst else 0

    all_g = safe_mean(rq1_tree_g + rq1_linear_g)
    all_l = safe_mean(rq1_tree_l + rq1_linear_l)
    t_g, t_l = safe_mean(rq1_tree_g), safe_mean(rq1_tree_l)
    l_g, l_l = safe_mean(rq1_linear_g), safe_mean(rq1_linear_l)

    # ── RQ2 ──
    rq2_per_model = {}  # model_name → avg improvement %
    for model in all_models:
        rq2_data = load_json(base_dir / model / 'rq2.json')
        if not rq2_data:
            continue

        windows = rq2_data['windows']
        new_format = 'cte_mae_global' in windows[0]
        cte_maes, rand_maes = [], []

        for w in windows:
            if new_format:
                cte_maes.append(w['cte_mae_global'])
                rand_maes.append(w['rand_mae_global_mean'])
            else:
                t = np.array(w['truth_importance'])
                c = np.array(w['cte_importance'])
                r = np.array(w.get('rand_importance_mean', w.get('rand_importance', [])))
                cte_maes.append(np.mean(np.abs(t - c)))
                rand_maes.append(np.mean(np.abs(t - r)))

        avg_cte = np.mean(cte_maes)
        avg_rand = np.mean(rand_maes)
        imp = calculate_improvement(avg_rand, avg_cte)
        rq2_per_model[model] = imp

    rq2_tree = [rq2_per_model[m] for m in tree_models if m in rq2_per_model]
    rq2_linear = [rq2_per_model[m] for m in linear_models if m in rq2_per_model]
    rq2_all = rq2_tree + rq2_linear
    rq2_wins = sum(1 for v in rq2_all if v > 0)
    rq2_losses = sum(1 for v in rq2_all if v <= 0)

    # ── RQ3 ──
    rq3_per_model = {}
    for model in all_models:
        rq3_data = load_json(base_dir / model / 'rq3.json')
        if not rq3_data:
            continue
        if 'cte' in rq3_data and 'train_mae' in rq3_data.get('cte', {}):
            rq3_per_model[model] = {
                'cte_train': rq3_data['cte']['train_mae'],
                'cte_val': rq3_data['cte']['val_mae'],
                'rand_train': rq3_data['random']['train_mae_mean'],
                'rand_val': rq3_data['random']['val_mae_mean'],
            }

    # ── Generate Report ──
    rq2_table_lines = []
    for m in all_models:
        if m in rq2_per_model:
            v = rq2_per_model[m]
            sign = '+' if v > 0 else ''
            winner = 'CTE' if v > 0 else 'Random'
            rq2_table_lines.append(f"| {pretty.get(m, m)} | {sign}{v:.1f}% | {winner} |")

    rq3_table_lines = []
    for m in all_models:
        if m in rq3_per_model:
            d = rq3_per_model[m]
            imp_train = calculate_improvement(d['rand_train'], d['cte_train'])
            imp_val = calculate_improvement(d['rand_val'], d['cte_val'])
            rq3_table_lines.append(
                f"| {pretty.get(m, m)} | {d['cte_train']:.2e} | {d['rand_train']:.2e} | "
                f"{imp_train:+.1f}% | {d['cte_val']:.2e} | {d['rand_val']:.2e} | {imp_val:+.1f}% |"
            )

    report = f"""# 📊 Quantitative Summary Metrics for Academic Report
*(Generated for representative Background Size N={target_size})*

This document provides aggregated hard numbers across all 8 models to directly quote in your thesis.

## 1. RQ1: Approximation Accuracy (MAE at N={target_size})

> Does CTE achieve lower SHAP approximation error than random sampling at the same background size?

### Across ALL 8 Models (Global Average)
- **Global Feature Importance:** CTE is on average **{all_g:+.1f}%** more accurate than Random.
- **Local (Per-Sample) Explanation:** CTE is on average **{all_l:+.1f}%** more accurate than Random.

### By Model Family
**Tree-based Ensembles (XGBoost, RF, LightGBM, CatBoost):**
- Global MAE Improvement: **{t_g:+.1f}%**
- Local MAE Improvement: **{t_l:+.1f}%**

**Linear Models (LogReg, Ridge, Lasso, SVC):**
- Global MAE Improvement: **{l_g:+.1f}%**
- Local MAE Improvement: **{l_l:+.1f}%**

---

## 2. RQ2: Temporal Feature Importance Tracking (N={target_size})

> Do feature importances shift across chronological test periods, and does a fixed CTE background track the shifts better than Random?

### Summary
- **CTE wins in {rq2_wins}/{len(rq2_all)} models** ({rq2_losses} losses)
- Average Δ: **{safe_mean(rq2_all):+.1f}%** across all models
- Tree models avg: **{safe_mean(rq2_tree):+.1f}%** | Linear models avg: **{safe_mean(rq2_linear):+.1f}%**

### Per-Model Breakdown

| Model | CTE vs Random Δ% | Winner |
|---|---|---|
{chr(10).join(rq2_table_lines)}

---

## 3. RQ3: Train vs Validation Feature Drift (N={target_size})

> Do SHAP values differ between train and val data, and does CTE track this shift?

"""

    if rq3_table_lines:
        report += """### Per-Model Error Comparison

| Model | CTE (train) | Rand (train) | Δ% train | CTE (val) | Rand (val) | Δ% val |
|---|---|---|---|---|---|---|
"""
        report += '\n'.join(rq3_table_lines)
    else:
        report += "*RQ3 data in old format — re-run rq3_train_val.py for detailed comparison.*\n"

    report += """

---
## Methodology Notes
- **Ground truth**: SHAP computed with full training set background (109,341 points)
- **CTE**: Single deterministic coreset (seed=42) built via Compress++ → KT
- **Random**: Mean of 30 independent random subsamples ± 1σ
- **Metric**: MAE = Mean Absolute Error of feature importances vs ground truth
- **Limitation**: CTE is a single realization; its variance is not estimated
"""

    out_path = base_dir / 'quantitative_summary.md'
    with open(out_path, 'w') as f:
        f.write(report)

    print(f"Quantitative summary generated at {out_path}")


if __name__ == "__main__":
    generate_summary()
