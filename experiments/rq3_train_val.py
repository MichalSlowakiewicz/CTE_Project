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

def run_rq3():
    print("Loading dataset for RQ3...")
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
