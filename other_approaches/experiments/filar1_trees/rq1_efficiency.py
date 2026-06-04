import time
import json
import numpy as np
import shap
import pandas as pd
import joblib
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from data.data_loader import load_dataset
from cte_compression.kernel_thinning import build_iid_background
from evaluation.metrics import mae_shap, speedup_ratio

def run_filar1_rq1():
    print("=== Filar 1 (TreeExplainer) - RQ1: Efficiency vs Fidelity ===")
    
    cache_dir = Path(__file__).parent.parent.parent / 'data' / 'cache_full'
    out_dir = Path(__file__).parent.parent.parent / 'results' / 'filar1_trees'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    model = joblib.load(cache_dir / 'xgb_model.pkl')
    data = load_dataset(data_dir=Path(__file__).parent.parent.parent / 'data' / 'ecom-offers')
    
    X_train, X_val = data['X_train'], data['X_val']
    
    # We explain a subset of validation data for speed in iterative experiments
    X_val_explain = X_val.sample(min(200, len(X_val)), random_state=42).reset_index(drop=True)
    
    print("Loading Ground Truth Background...")
    bg_truth = pd.read_parquet(cache_dir / 'bg_truth_full.parquet')
    
    t0 = time.time()
    explainer_truth = shap.TreeExplainer(model, data=bg_truth, model_output="probability", feature_perturbation="interventional")
    shap_truth = explainer_truth.shap_values(X_val_explain)
    time_truth = time.time() - t0
    
    bg_sizes = [10, 50, 100, 200, 500]
    results = []
    
    for size in tqdm(bg_sizes, desc="RQ1 (Filar 1) sizes"):
        bg_cte = pd.read_parquet(cache_dir / f'bg_cte_{size}.parquet')
        
        t0 = time.time()
        explainer_cte = shap.TreeExplainer(model, data=bg_cte, model_output="probability", feature_perturbation="interventional")
        shap_cte = explainer_cte.shap_values(X_val_explain)
        t_cte = time.time() - t0
        
        bg_rand = build_iid_background(X_train, size=size)
        t0 = time.time()
        explainer_rand = shap.TreeExplainer(model, data=bg_rand, model_output="probability", feature_perturbation="interventional")
        shap_rand = explainer_rand.shap_values(X_val_explain)
        t_rand = time.time() - t0
        
        res = {
            "size": size,
            "cte": {"mae": float(mae_shap(shap_truth, shap_cte)), "speedup": float(speedup_ratio(time_truth, t_cte))},
            "random": {"mae": float(mae_shap(shap_truth, shap_rand)), "speedup": float(speedup_ratio(time_truth, t_rand))}
        }
        results.append(res)
        
    with open(out_dir / 'rq1.json', 'w') as f:
        json.dump({"truth_time": time_truth, "experiments": results}, f, indent=2)
    print("\nResults saved to results/filar1_trees/rq1.json")

if __name__ == "__main__":
    run_filar1_rq1()
