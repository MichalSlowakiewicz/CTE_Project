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
    shap_vals = explainer(X_val_batch, max_evals=2381, silent=True)
    return shap_vals.values[..., 1]

def run_filar5_rq3():
    print("=== Filar 5 (PermutationSHAP) - RQ3: Train vs Val Discrepancy ===")
    
    cache_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'cache_full'
    out_dir = Path(__file__).parent.parent.parent.parent / 'results' / 'filar3_xgboost_explainers' / 'permutation_shap'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    model = joblib.load(cache_dir / 'catboost_model.pkl')
    predict_fn = model.predict_proba
    
    data = load_dataset(data_dir=Path(__file__).parent.parent.parent.parent / 'data' / 'ecom-offers')
    X_train, X_val = data['X_train'], data['X_val']
    
    X_train_explain = X_train.sample(min(100, len(X_train)), random_state=42).reset_index(drop=True)
    X_val_explain = X_val.sample(min(100, len(X_val)), random_state=42).reset_index(drop=True)
    
    bg_truth_sampled = pd.read_parquet(cache_dir / 'bg_truth_full.parquet').sample(1000, random_state=42)
    
    num_cores = 1
    
    print("Computing Ground Truth SHAP (Train vs Val)...")
    with multiprocessing.Pool(num_cores) as pool:
        func = partial(explain_samples_permutation, model_predict_proba=predict_fn, bg_data=bg_truth_sampled)
        shap_truth_train = np.vstack(pool.map(func, np.array_split(X_train_explain, num_cores)))
        shap_truth_val = np.vstack(pool.map(func, np.array_split(X_val_explain, len(X_val_explain))))
        
    bg_sizes = [10, 50, 100, 200]
    results = []
    
    for size in tqdm(bg_sizes, desc="RQ3 (Filar 5) sizes"):
        bg_cte = pd.read_parquet(cache_dir / f'bg_cte_{size}.parquet')
        
        with multiprocessing.Pool(num_cores) as pool:
            func = partial(explain_samples_permutation, model_predict_proba=predict_fn, bg_data=bg_cte)
            shap_cte_train = np.vstack(pool.map(func, np.array_split(X_train_explain, num_cores)))
            shap_cte_val = np.vstack(pool.map(func, np.array_split(X_val_explain, len(X_val_explain))))
        
        res = {
            "size": size,
            "mae_train_set": float(mae_shap(shap_truth_train, shap_cte_train)),
            "mae_val_set": float(mae_shap(shap_truth_val, shap_cte_val))
        }
        results.append(res)
        
    with open(out_dir / 'rq3.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_filar5_rq3()
