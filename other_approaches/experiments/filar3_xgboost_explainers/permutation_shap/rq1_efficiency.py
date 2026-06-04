import time
import json
import numpy as np
import shap
import pandas as pd
import joblib
import multiprocessing
from pathlib import Path
from tqdm import tqdm
from functools import partial
import sys

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from data.data_loader import load_dataset
from evaluation.metrics import mae_shap

def explain_samples_permutation(X_val_batch, model_predict_proba, bg_data):
    masker = shap.maskers.Independent(bg_data)
    explainer = shap.PermutationExplainer(model_predict_proba, masker)
    
    # We explain the probability of class 1
    # npermutations=10 is approx 2 * 10 * 119 + 1 = 2381 evals
    shap_vals = explainer(X_val_batch, max_evals=2381, silent=True)
    return shap_vals.values[..., 1]

def run_filar5_rq1():
    print("=== Filar 5 (PermutationSHAP) - RQ1: Efficiency & Accuracy ===")
    
    cache_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'cache_full'
    out_dir = Path(__file__).parent.parent.parent.parent / 'results' / 'filar3_xgboost_explainers' / 'permutation_shap'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    model = joblib.load(cache_dir / 'catboost_model.pkl')
    predict_fn = model.predict_proba
    
    data = load_dataset(data_dir=Path(__file__).parent.parent.parent.parent / 'data' / 'ecom-offers')
    X_val = data['X_val']
    
    X_val_explain = X_val.sample(min(100, len(X_val)), random_state=42).reset_index(drop=True)
    
    # Ground truth (Limit to 10000 to match the 20x paper methodology for max size 200)
    bg_truth_full = pd.read_parquet(cache_dir / 'bg_truth_full.parquet')
    bg_truth_sampled = bg_truth_full.sample(1000, random_state=42)
    
    num_cores = 1
    batches = np.array_split(X_val_explain, len(X_val_explain))
    
    print("Computing Ground Truth SHAP (PermutationSHAP)...")
    t0 = time.time()
    with multiprocessing.Pool(num_cores) as pool:
        func = partial(explain_samples_permutation, model_predict_proba=predict_fn, bg_data=bg_truth_sampled)
        shap_truth = np.vstack(list(tqdm(pool.imap(func, batches), total=len(batches), desc="Ground Truth")))
    time_truth = time.time() - t0
    print(f"Ground Truth computed in {time_truth:.2f} s")
    
    bg_sizes = [10, 50, 100, 200]
    results = []
    
    for size in tqdm(bg_sizes, desc="CTE Sizes"):
        bg_cte = pd.read_parquet(cache_dir / f'bg_cte_{size}.parquet')
        
        t0 = time.time()
        with multiprocessing.Pool(num_cores) as pool:
            func = partial(explain_samples_permutation, model_predict_proba=predict_fn, bg_data=bg_cte)
            shap_cte = np.vstack(list(tqdm(pool.imap(func, batches), total=len(batches), desc="Ground Truth")))
        time_cte = time.time() - t0
        
        mae = float(mae_shap(shap_truth, shap_cte))
        
        res = {
            "size": size,
            "mae": mae,
            "time_cte": time_cte,
            "time_truth": time_truth,
            "speedup": time_truth / time_cte if time_cte > 0 else 0
        }
        results.append(res)
        
    with open(out_dir / 'rq1.json', 'w') as f:
        json.dump(results, f, indent=2)
        
if __name__ == "__main__":
    run_filar5_rq1()
