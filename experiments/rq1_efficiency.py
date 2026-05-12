import time
import json
import numpy as np
import shap
from pathlib import Path
import warnings

import sys
sys.path.append(str(Path(__file__).parent.parent))

from data.data_loader import load_dataset
from models.trainer import train_model
from cte_compression.kernel_thinning import build_cte_background, build_iid_background
from evaluation.metrics import spearman_rank_correlation, speedup_ratio

def run_rq1():
    print("Loading dataset...")
    try:
        data = load_dataset(data_dir=Path(__file__).parent.parent / 'data' / 'ecom-offers')
    except Exception as e:
        warnings.warn(f"Failed to load TabReD dataset, falling back to synthetic. Error: {e}")
        data = load_dataset()
    
    # Take a subsample for training so we can run this quickly, e.g., 50k
    train_size = min(50_000, len(data['X_train']))
    # Sample using pandas methods safely
    sampled_indices = data['X_train'].sample(train_size, random_state=42).index
    X_train = data['X_train'].loc[sampled_indices].reset_index(drop=True)
    y_train = data['y_train'][sampled_indices]
    
    print(f"Training XGBoost model on {train_size} samples...")
    model = train_model(X_train, y_train, X_val=data['X_val'], y_val=data['y_val'], verbose=True)
    
    # We will explain a subset of validation set
    val_explain_size = min(100, len(data['X_val']))
    X_val_explain = data['X_val'].sample(val_explain_size, random_state=42).reset_index(drop=True)
    
    # Ground truth baseline (large random background)
    print("Computing Ground Truth SHAP (Random 500 background)...")
    bg_truth = build_iid_background(X_train, size=500)
    
    # Wrapper for predict_proba to return class 1 probability directly to speed up SHAP
    def pred_fn(X):
        return model.predict_proba(X)[:, 1]

    explainer_truth = shap.KernelExplainer(pred_fn, bg_truth)
    t0 = time.time()
    shap_truth = explainer_truth.shap_values(X_val_explain)
    time_truth = time.time() - t0
    
    bg_sizes = [10, 50, 100, 200]
    results = []
    
    for size in bg_sizes:
        print(f"\n--- Background size: {size} ---")
        
        # CTE
        print("  Building CTE background...")
        t_cte_build_start = time.time()
        bg_cte = build_cte_background(X_train, target_size=size, verbose=False)
        t_cte_build = time.time() - t_cte_build_start
        
        print("  Computing CTE SHAP...")
        explainer_cte = shap.KernelExplainer(pred_fn, bg_cte)
        t_cte_shap_start = time.time()
        shap_cte = explainer_cte.shap_values(X_val_explain)
        t_cte_shap = time.time() - t_cte_shap_start
        
        corr_cte_global = spearman_rank_correlation(shap_truth, shap_cte, global_importance=True)
        corr_cte_local = spearman_rank_correlation(shap_truth, shap_cte, global_importance=False)
        
        # Random
        print("  Building Random background...")
        bg_rand = build_iid_background(X_train, size=size)
        
        print("  Computing Random SHAP...")
        explainer_rand = shap.KernelExplainer(pred_fn, bg_rand)
        t_rand_shap_start = time.time()
        shap_rand = explainer_rand.shap_values(X_val_explain)
        t_rand_shap = time.time() - t_rand_shap_start
        
        corr_rand_global = spearman_rank_correlation(shap_truth, shap_rand, global_importance=True)
        corr_rand_local = spearman_rank_correlation(shap_truth, shap_rand, global_importance=False)
        
        res = {
            "size": size,
            "cte": {
                "build_time": t_cte_build,
                "shap_time": t_cte_shap,
                "total_time": t_cte_build + t_cte_shap,
                "corr_global": corr_cte_global,
                "corr_local": corr_cte_local,
                "speedup_vs_truth": speedup_ratio(time_truth, t_cte_shap)
            },
            "random": {
                "build_time": 0.0,
                "shap_time": t_rand_shap,
                "total_time": t_rand_shap,
                "corr_global": corr_rand_global,
                "corr_local": corr_rand_local,
                "speedup_vs_truth": speedup_ratio(time_truth, t_rand_shap)
            }
        }
        results.append(res)
        
        print(f"  CTE global corr: {corr_cte_global:.3f}, local corr: {corr_cte_local:.3f}")
        print(f"  Rand global corr: {corr_rand_global:.3f}, local corr: {corr_rand_local:.3f}")
        
    out_dir = Path(__file__).parent.parent / 'results'
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / 'rq1.json', 'w') as f:
        json.dump({
            "truth_time": time_truth,
            "experiments": results
        }, f, indent=2)
    print("\nResults saved to results/rq1.json")

if __name__ == "__main__":
    run_rq1()
