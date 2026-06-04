import json
from pathlib import Path
import numpy as np

def load_json(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None

def calculate_improvement(baseline, new_val):
    if baseline == 0: return 0
    return ((baseline - new_val) / baseline) * 100

def generate_summary():
    base_dir = Path(__file__).parent.parent.parent / "results" / "fallback_plan"
    
    tree_models = ['xgb_tree', 'rf_tree', 'lgb_tree', 'catboost_tree']
    linear_models = ['logreg_linear', 'ridge_linear', 'lasso_linear', 'svc_linear']
    
    all_models = tree_models + linear_models
    
    # We will focus on a representative background size for the hard numbers in the text, e.g., N=64
    target_size = 64
    
    rq1_tree_improvements_global = []
    rq1_tree_improvements_local = []
    rq1_linear_improvements_global = []
    rq1_linear_improvements_local = []
    
    rq2_tree_improvements = []
    rq2_linear_improvements = []

    for model in all_models:
        is_tree = model in tree_models
        model_dir = base_dir / model
        
        # --- RQ1 Metrics ---
        rq1_data = load_json(model_dir / 'rq1.json')
        if rq1_data:
            for exp in rq1_data['experiments']:
                if exp['size'] == target_size:
                    cte_g = exp['cte']['mae_global']
                    rand_g = exp['random']['mae_global']
                    cte_l = exp['cte']['mae_local']
                    rand_l = exp['random']['mae_local']
                    
                    imp_g = calculate_improvement(rand_g, cte_g)
                    imp_l = calculate_improvement(rand_l, cte_l)
                    
                    if is_tree:
                        rq1_tree_improvements_global.append(imp_g)
                        rq1_tree_improvements_local.append(imp_l)
                    else:
                        rq1_linear_improvements_global.append(imp_g)
                        rq1_linear_improvements_local.append(imp_l)
                        
        # --- RQ2 Metrics ---
        rq2_data = load_json(model_dir / 'rq2.json')
        if rq2_data:
            model_cte_maes = []
            model_rand_maes = []
            for w in rq2_data['windows']:
                t = np.array(w['truth_importance'])
                c = np.array(w['cte_importance'])
                r = np.array(w['rand_importance'])
                model_cte_maes.append(np.mean(np.abs(t - c)))
                model_rand_maes.append(np.mean(np.abs(t - r)))
                
            avg_cte = np.mean(model_cte_maes)
            avg_rand = np.mean(model_rand_maes)
            imp_drift = calculate_improvement(avg_rand, avg_cte)
            
            if is_tree:
                rq2_tree_improvements.append(imp_drift)
            else:
                rq2_linear_improvements.append(imp_drift)

    # Calculate Aggregates
    def safe_mean(l): return np.mean(l) if l else 0
    
    t_g = safe_mean(rq1_tree_improvements_global)
    t_l = safe_mean(rq1_tree_improvements_local)
    l_g = safe_mean(rq1_linear_improvements_global)
    l_l = safe_mean(rq1_linear_improvements_local)
    
    all_g = safe_mean(rq1_tree_improvements_global + rq1_linear_improvements_global)
    all_l = safe_mean(rq1_tree_improvements_local + rq1_linear_improvements_local)
    
    t_drift = safe_mean(rq2_tree_improvements)
    l_drift = safe_mean(rq2_linear_improvements)
    all_drift = safe_mean(rq2_tree_improvements + rq2_linear_improvements)

    report = f"""# 📊 Quantitative Summary Metrics for Academic Report
*(Generated for representative Background Size N={target_size})*

This document provides aggregated hard numbers across all 8 models to directly quote in your thesis.

## 1. RQ1: Approximation Accuracy (Global & Local MAE)
How much better is CTE at approximating the true SHAP distribution compared to Random sampling?

### Across ALL 8 Models (Global Average)
- **Global Feature Importance:** CTE is on average **{all_g:+.1f}%** more accurate than Random sampling.
- **Local (Per-Sample) Explanation:** CTE is on average **{all_l:+.1f}%** more accurate than Random sampling.

### By Model Family
**Tree-based Ensembles (XGBoost, RF, LightGBM, CatBoost):**
- Global MAE Improvement: **{t_g:+.1f}%**
- Local MAE Improvement: **{t_l:+.1f}%**

**Linear Models (LogReg, Ridge, Lasso, SVC):**
- Global MAE Improvement: **{l_g:+.1f}%**
- Local MAE Improvement: **{l_l:+.1f}%**

---

## 2. RQ2: Temporal Drift Robustness
When evaluating how accurately the algorithms track feature importance over time (across 5 data drift windows):

### Across ALL 8 Models
- Temporal Tracking Accuracy: CTE reduces the drift tracking error by an average of **{all_drift:+.1f}%** compared to Random.

### By Model Family
- **Tree-based Ensembles:** **{t_drift:+.1f}%** more accurate than Random in tracking temporal drift.
- **Linear Models:** **{l_drift:+.1f}%** more accurate than Random in tracking temporal drift.

---
**How to use these numbers:**
*„Our experiments across 8 diverse architectures indicate that at a standard background size of N=64, the Coreset Tree Explainer (CTE) reduces the Global MAE of feature importance by an average of {all_g:.1f}% compared to uniform random sampling. This superiority holds across architectural boundaries, with Tree-based ensembles seeing a {t_g:.1f}% improvement and Linear models a {l_g:.1f}% improvement...”*
"""
    
    out_path = base_dir / 'quantitative_summary.md'
    with open(out_path, 'w') as f:
        f.write(report)
        
    print(f"Quantitative summary generated at {out_path}")

if __name__ == "__main__":
    generate_summary()
