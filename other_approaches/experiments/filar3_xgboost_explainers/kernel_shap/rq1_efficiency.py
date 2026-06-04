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
from cte_compression.kernel_thinning import build_iid_background
from evaluation.metrics import mae_shap, speedup_ratio

def explain_samples_kernel(X_val_batch, model_predict_proba, bg_data):
    explainer = shap.KernelExplainer(model_predict_proba, bg_data)
    return explainer.shap_values(X_val_batch, l1_reg="num_features(10)", silent=True)[1]

def run_filar3_rq1():
    print("=== Filar 3 (Black-Box KernelExplainer) - RQ1: Efficiency vs Fidelity ===")
    
    cache_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'cache_full'
    out_dir = Path(__file__).parent.parent.parent.parent / 'results' / 'filar3_xgboost_explainers' / 'kernel_shap'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    model = joblib.load(cache_dir / 'catboost_model.pkl')
    predict_fn = model.predict_proba
    
    data = load_dataset(data_dir=Path(__file__).parent.parent.parent.parent / 'data' / 'ecom-offers')
    X_train, X_val = data['X_train'], data['X_val']
    
    X_val_explain = X_val.sample(min(100, len(X_val)), random_state=42).reset_index(drop=True)
    
    # Ground truth (Limit to 10000 to match the 20x paper methodology for max size 200)
    bg_truth_full = pd.read_parquet(cache_dir / 'bg_truth_full.parquet')
    bg_truth_sampled = bg_truth_full.sample(1000, random_state=42)
    
    num_cores = 1
    batches = np.array_split(X_val_explain, len(X_val_explain))
    
    print("Computing Ground Truth SHAP...")
    t0 = time.time()
    with multiprocessing.Pool(num_cores) as pool:
        func = partial(explain_samples_kernel, model_predict_proba=predict_fn, bg_data=bg_truth_sampled)
        shap_truth = np.vstack(list(tqdm(pool.imap(func, batches), total=len(batches), desc="Ground Truth")))
    time_truth = time.time() - t0
    
    bg_sizes = [10, 50, 100, 200]
    results = []
    
    for size in tqdm(bg_sizes, desc="RQ1 (Filar 3) sizes"):
        bg_cte = pd.read_parquet(cache_dir / f'bg_cte_{size}.parquet')
        
        t0 = time.time()
        with multiprocessing.Pool(num_cores) as pool:
            func = partial(explain_samples_kernel, model_predict_proba=predict_fn, bg_data=bg_cte)
            shap_cte = np.vstack(list(tqdm(pool.imap(func, batches), total=len(batches), desc="Ground Truth")))
        t_cte = time.time() - t0
        
        bg_rand = build_iid_background(X_train, size=size)
        
        t0 = time.time()
        with multiprocessing.Pool(num_cores) as pool:
            func = partial(explain_samples_kernel, model_predict_proba=predict_fn, bg_data=bg_rand)
            shap_rand = np.vstack(list(tqdm(pool.imap(func, batches), total=len(batches), desc="Ground Truth")))
        t_rand = time.time() - t0
        
        res = {
            "size": size,
            "cte": {"mae": float(mae_shap(shap_truth, shap_cte)), "speedup": float(speedup_ratio(time_truth, t_cte))},
            "random": {"mae": float(mae_shap(shap_truth, shap_rand)), "speedup": float(speedup_ratio(time_truth, t_rand))}
        }
        results.append(res)
        
    with open(out_dir / 'rq1.json', 'w') as f:
        json.dump({"truth_time": time_truth, "experiments": results}, f, indent=2)

if __name__ == "__main__":
    run_filar3_rq1()
