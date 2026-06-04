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
from cte_compression.kernel_thinning import build_cte_background

def explain_samples_kernel(X_val_batch, model_predict_proba, bg_data):
    explainer = shap.KernelExplainer(model_predict_proba, bg_data)
    return explainer.shap_values(X_val_batch, l1_reg="num_features(10)", silent=True)[1]

def run_filar3_rq4():
    print("=== Filar 3 (Black-Box KernelExplainer) - RQ4: Break-even Point ===")
    
    cache_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'cache_full'
    out_dir = Path(__file__).parent.parent.parent.parent / 'results' / 'filar3_xgboost_explainers' / 'kernel_shap'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    model = joblib.load(cache_dir / 'catboost_model.pkl')
    predict_fn = model.predict_proba
    
    data = load_dataset(data_dir=Path(__file__).parent.parent.parent.parent / 'data' / 'ecom-offers')
    X_train, X_val = data['X_train'], data['X_val']
    
    val_explain_sizes = [10, 50, 100, 200]
    bg_size = 50 
    
    print(f"Measuring CTE Compression time for target size {bg_size}...")
    t0 = time.time()
    bg_cte = build_cte_background(X_train, target_size=bg_size, verbose=False)
    compression_time = time.time() - t0
    
    bg_truth_sampled = pd.read_parquet(cache_dir / 'bg_truth_full.parquet').sample(1000, random_state=42)
    
    num_cores = 1
    results = []
    
    for n_val in tqdm(val_explain_sizes, desc="Explained Samples"):
        X_val_explain = X_val.sample(min(n_val, len(X_val)), random_state=42).reset_index(drop=True)
        batches = np.array_split(X_val_explain, len(X_val_explain))
        
        t0 = time.time()
        with multiprocessing.Pool(num_cores) as pool:
            func = partial(explain_samples_kernel, model_predict_proba=predict_fn, bg_data=bg_cte)
            _ = np.vstack(list(tqdm(pool.imap(func, batches), total=len(batches), desc="Ground Truth")))
        explain_time_cte = time.time() - t0
        
        t0 = time.time()
        with multiprocessing.Pool(num_cores) as pool:
            func = partial(explain_samples_kernel, model_predict_proba=predict_fn, bg_data=bg_truth_sampled)
            _ = np.vstack(list(tqdm(pool.imap(func, batches), total=len(batches), desc="Ground Truth")))
        explain_time_truth = time.time() - t0
        
        res = {
            "num_val_samples": n_val,
            "cte": {
                "compression_time": compression_time,
                "explain_time": explain_time_cte,
                "total_time": compression_time + explain_time_cte
            },
            "truth_bg_1000": {
                "explain_time": explain_time_truth,
                "total_time": explain_time_truth
            }
        }
        results.append(res)
        
    with open(out_dir / 'rq4.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_filar3_rq4()
