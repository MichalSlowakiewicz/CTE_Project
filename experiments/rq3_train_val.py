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

def run_rq3():
    print("Loading dataset for RQ3...")
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

    bg_size = 100
    print(f"Building CTE background (size {bg_size})...")
    bg_cte = build_cte_background(X_train, target_size=bg_size, verbose=False)
    explainer_cte = shap.KernelExplainer(pred_fn, bg_cte)
    
    print(f"Building Random background (size {bg_size})...")
    bg_rand = build_iid_background(X_train, size=bg_size)
    explainer_rand = shap.KernelExplainer(pred_fn, bg_rand)
    
    print(f"Building Ground Truth background (size 500)...")
    bg_truth = build_iid_background(X_train, size=500)
    explainer_truth = shap.KernelExplainer(pred_fn, bg_truth)
    
    # Explain on train vs val
    n_explain = min(100, len(data['X_val']))
    X_train_explain = X_train.sample(n_explain, random_state=42).reset_index(drop=True)
    X_val_explain = data['X_val'].sample(n_explain, random_state=42).reset_index(drop=True)
    
    results = {}
    
    for split_name, X_explain in [('Train', X_train_explain), ('Validation', X_val_explain)]:
        print(f"\n--- Computing SHAP for {split_name} set ---")
        
        print("  GT...")
        shap_truth = explainer_truth.shap_values(X_explain)
        print("  CTE...")
        shap_cte = explainer_cte.shap_values(X_explain)
        print("  Random...")
        shap_rand = explainer_rand.shap_values(X_explain)
        
        results[split_name.lower()] = {
            "truth_importance": mean_absolute_shap(shap_truth).tolist(),
            "cte_importance": mean_absolute_shap(shap_cte).tolist(),
            "rand_importance": mean_absolute_shap(shap_rand).tolist()
        }
        
    out_dir = Path(__file__).parent.parent / 'results'
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / 'rq3.json', 'w') as f:
        json.dump({
            "feature_names": data['feature_names'],
            "train": results['train'],
            "val": results['validation']
        }, f, indent=2)
    print("\nResults saved to results/rq3.json")

if __name__ == "__main__":
    run_rq3()
