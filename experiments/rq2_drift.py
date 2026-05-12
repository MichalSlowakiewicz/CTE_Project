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

def run_rq2():
    print("Loading dataset for RQ2...")
    try:
        data = load_dataset(data_dir=Path(__file__).parent.parent / 'data' / 'ecom-offers')
    except Exception as e:
        warnings.warn(f"Failed to load TabReD dataset, falling back to synthetic. Error: {e}")
        data = load_dataset()
        
    train_size = min(50_000, len(data['X_train']))
    sampled_indices = data['X_train'].sample(train_size, random_state=42).index
    X_train = data['X_train'].loc[sampled_indices].reset_index(drop=True)
    y_train = data['y_train'][sampled_indices]
    
    print(f"Training XGBoost model on {train_size} samples...")
    model = train_model(X_train, y_train, X_val=data['X_val'], y_val=data['y_val'], verbose=True)
    
    def pred_fn(X):
        return model.predict_proba(X)[:, 1]

    # Create backgrounds
    bg_size = 100
    print(f"Building CTE background (size {bg_size})...")
    bg_cte = build_cte_background(X_train, target_size=bg_size, verbose=False)
    explainer_cte = shap.KernelExplainer(pred_fn, bg_cte)
    
    print(f"Building Random background (size {bg_size})...")
    bg_rand = build_iid_background(X_train, size=bg_size)
    explainer_rand = shap.KernelExplainer(pred_fn, bg_rand)
    
    print(f"Building Ground Truth background (size 1000)...")
    bg_truth = build_iid_background(X_train, size=1000)
    explainer_truth = shap.KernelExplainer(pred_fn, bg_truth)
    
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
