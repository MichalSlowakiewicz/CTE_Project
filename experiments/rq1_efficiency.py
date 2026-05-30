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
from evaluation.metrics import mae_shap, speedup_ratio
import xgboost as xgb
import pandas as pd
import joblib

def run_rq1():
    print("Loading dataset...")
    try:
        data = load_dataset(data_dir=Path(__file__).parent.parent / 'data' / 'ecom-offers')
    except Exception as e:
        warnings.warn(f"Failed to load TabReD dataset, falling back to synthetic. Error: {e}")
        data = load_dataset()
    
    cache_dir = Path(__file__).parent.parent / 'data' / 'cache'
    
    if not (cache_dir / 'xgb_model.pkl').exists():
        raise FileNotFoundError("Cache not found! Please run `python experiments/prepare_cache.py` first.")
    
    print("Loading cached XGBoost model...")
    model = joblib.load(cache_dir / 'xgb_model.pkl')
    
    # We will explain a subset of validation set
    val_explain_size = min(100, len(data['X_val']))
    X_val_explain = data['X_val'].sample(val_explain_size, random_state=42).reset_index(drop=True)
    
    print("Loading cached Ground Truth Background (Full Dataset)...")
    bg_truth = pd.read_parquet(cache_dir / 'bg_truth_full.parquet')
    
    # Removed loading of bg_cte_200 here, we will load sizes dynamically
    
    X_train = data['X_train'] # needed for random background sampling
    
    explainer_truth = shap.TreeExplainer(model, data=bg_truth, model_output="probability", feature_perturbation="interventional")
    t0 = time.time()
    shap_truth = explainer_truth.shap_values(X_val_explain)
    time_truth = time.time() - t0
    
    bg_sizes = [10, 50, 100, 200]
    results = []
    
    for size in bg_sizes:
        print(f"\n--- Background size: {size} ---")
        
        # CTE
        print(f"  Loading cached CTE background (size {size})...")
        t_cte_build_start = time.time()
        # Load precomputed pure CTE coreset
        bg_cte_path = cache_dir / f'bg_cte_{size}.parquet'
        if not bg_cte_path.exists():
            print(f"Warning: {bg_cte_path} not found. Running prepare_cache.py is recommended.")
            bg_cte = build_cte_background(X_train, target_size=size, verbose=False)
        else:
            bg_cte = pd.read_parquet(bg_cte_path)
        t_cte_build = time.time() - t_cte_build_start
        
        print("  Computing CTE SHAP...")
        explainer_cte = shap.TreeExplainer(model, data=bg_cte, model_output="probability", feature_perturbation="interventional")
        t_cte_shap_start = time.time()
        shap_cte = explainer_cte.shap_values(X_val_explain)
        t_cte_shap = time.time() - t_cte_shap_start
        
        mae_cte_global = mae_shap(shap_truth, shap_cte, global_importance=True)
        mae_cte_local = mae_shap(shap_truth, shap_cte, global_importance=False)
        
        # Random
        print("  Building Random background...")
        bg_rand = build_iid_background(X_train, size=size)
        
        print("  Computing Random SHAP...")
        explainer_rand = shap.TreeExplainer(model, data=bg_rand, model_output="probability", feature_perturbation="interventional")
        t_rand_shap_start = time.time()
        shap_rand = explainer_rand.shap_values(X_val_explain)
        t_rand_shap = time.time() - t_rand_shap_start
        
        mae_rand_global = mae_shap(shap_truth, shap_rand, global_importance=True)
        mae_rand_local = mae_shap(shap_truth, shap_rand, global_importance=False)
        
        res = {
            "size": size,
            "cte": {
                "build_time": t_cte_build,
                "shap_time": t_cte_shap,
                "total_time": t_cte_build + t_cte_shap,
                "mae_global": mae_cte_global,
                "mae_local": mae_cte_local,
                "speedup_vs_truth": speedup_ratio(time_truth, t_cte_shap)
            },
            "random": {
                "build_time": 0.0,
                "shap_time": t_rand_shap,
                "total_time": t_rand_shap,
                "mae_global": mae_rand_global,
                "mae_local": mae_rand_local,
                "speedup_vs_truth": speedup_ratio(time_truth, t_rand_shap)
            }
        }
        results.append(res)
        
        print(f"  CTE global MAE: {mae_cte_global:.5f}, local MAE: {mae_cte_local:.5f}")
        print(f"  Rand global MAE: {mae_rand_global:.5f}, local MAE: {mae_rand_local:.5f}")
        
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
