import time
import json
import numpy as np
import pandas as pd
import shap
from pathlib import Path
import warnings

import sys
sys.path.append(str(Path(__file__).parent.parent))

from data.data_loader import load_dataset
from models.trainer import train_model
from cte_compression.kernel_thinning import build_cte_background, build_iid_background
from evaluation.metrics import mean_absolute_shap
import xgboost as xgb
import joblib

def run_rq2():
    print("Loading dataset for RQ2...")
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
    
        
    X_train = data['X_train'] # needed for random background sampling

    bg_size = 100
    print(f"Loading CTE Background (size {bg_size})...")
    bg_cte_path = cache_dir / f'bg_cte_{bg_size}.parquet'
    if not bg_cte_path.exists():
        print(f"Warning: {bg_cte_path} not found. Generating on the fly...")
        bg_cte = build_cte_background(X_train, target_size=bg_size, verbose=False)
        bg_cte.to_parquet(bg_cte_path)
    else:
        bg_cte = pd.read_parquet(bg_cte_path)
    explainer_cte = shap.TreeExplainer(model, data=bg_cte, model_output="probability", feature_perturbation="interventional")
    
    print(f"Building Random background (size {bg_size})...")
    bg_rand = build_iid_background(X_train, size=bg_size)
    explainer_rand = shap.TreeExplainer(model, data=bg_rand, model_output="probability", feature_perturbation="interventional")
    
    print(f"Loading cached Ground Truth Background (Full Dataset)...")
    bg_truth = pd.read_parquet(cache_dir / 'bg_truth_full.parquet')
    explainer_truth = shap.TreeExplainer(model, data=bg_truth, model_output="probability", feature_perturbation="interventional")
    
    # We will slice the validation set into 5 chronological windows
    # `periods_val` has the timestamp or period index. We can sort `X_val` by `periods_val`.
    X_val = data['X_val'].copy()
    X_val['period'] = data['periods_val']
    X_val_sorted = X_val.sort_values(by='period').reset_index(drop=True)
    X_val_sorted = X_val_sorted.drop(columns=['period'])
    
    n_windows = 5
    window_size = len(X_val_sorted) // n_windows
    
    # Subsample within windows to avoid spending hours computing SHAP
    samples_per_window = min(40, window_size)
    
    results = []
    
    for i in range(n_windows):
        print(f"\n--- Processing Temporal Window {i+1}/{n_windows} ---")
        start_idx = i * window_size
        end_idx = start_idx + window_size if i < n_windows - 1 else len(X_val_sorted)
        
        X_window = X_val_sorted.iloc[start_idx:end_idx]
        X_window_sample = X_window.sample(samples_per_window, random_state=42).reset_index(drop=True)
        
        print("  Computing GT SHAP...")
        shap_truth = explainer_truth.shap_values(X_window_sample)
        
        print("  Computing CTE SHAP...")
        shap_cte = explainer_cte.shap_values(X_window_sample)
        
        print("  Computing Random SHAP...")
        shap_rand = explainer_rand.shap_values(X_window_sample)
        
        imp_truth = mean_absolute_shap(shap_truth).tolist()
        imp_cte = mean_absolute_shap(shap_cte).tolist()
        imp_rand = mean_absolute_shap(shap_rand).tolist()
        
        results.append({
            "window": i + 1,
            "truth_importance": imp_truth,
            "cte_importance": imp_cte,
            "rand_importance": imp_rand
        })
        
    out_dir = Path(__file__).parent.parent / 'results'
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / 'rq2.json', 'w') as f:
        json.dump({
            "feature_names": data['feature_names'],
            "windows": results
        }, f, indent=2)
    print("\nResults saved to results/rq2.json")

if __name__ == "__main__":
    run_rq2()
