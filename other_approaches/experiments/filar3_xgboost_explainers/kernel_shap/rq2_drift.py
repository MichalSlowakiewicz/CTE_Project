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
from evaluation.metrics import mae_shap

def explain_samples_kernel(X_val_batch, model_predict_proba, bg_data):
    explainer = shap.KernelExplainer(model_predict_proba, bg_data)
    return explainer.shap_values(X_val_batch, l1_reg="num_features(10)", silent=True)[1]

def run_filar3_rq2():
    print("=== Filar 3 (Black-Box KernelExplainer) - RQ2: Concept Drift ===")
    
    cache_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'cache_full'
    out_dir = Path(__file__).parent.parent.parent.parent / 'results' / 'filar3_xgboost_explainers' / 'kernel_shap'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    model = joblib.load(cache_dir / 'catboost_model.pkl')
    predict_fn = model.predict_proba
    
    data = load_dataset(data_dir=Path(__file__).parent.parent.parent.parent / 'data' / 'ecom-offers')
    X_train, X_test = data['X_train'], data['X_test']
    
    X_test_explain = X_test.sample(min(100, len(X_test)), random_state=42).reset_index(drop=True)
    
    bg_truth_test = shap.kmeans(X_test, 10000).data
    
    num_cores = 1
    batches = np.array_split(X_test_explain, len(X_test_explain))
    
    print("Computing Ground Truth SHAP on Test background...")
    with multiprocessing.Pool(num_cores) as pool:
        func = partial(explain_samples_kernel, model_predict_proba=predict_fn, bg_data=bg_truth_test)
        shap_truth = np.vstack(list(tqdm(pool.imap(func, batches), total=len(batches), desc="Ground Truth")))
    
    bg_sizes = [50, 100, 200]
    results = []
    
    for size in tqdm(bg_sizes, desc="RQ2 (Filar 3) sizes"):
        bg_cte_past = pd.read_parquet(cache_dir / f'bg_cte_{size}.parquet')
        with multiprocessing.Pool(num_cores) as pool:
            func = partial(explain_samples_kernel, model_predict_proba=predict_fn, bg_data=bg_cte_past)
            shap_past = np.vstack(list(tqdm(pool.imap(func, batches), total=len(batches), desc="Ground Truth")))
            
        bg_cte_future = build_cte_background(X_test, target_size=size, verbose=False)
        with multiprocessing.Pool(num_cores) as pool:
            func = partial(explain_samples_kernel, model_predict_proba=predict_fn, bg_data=bg_cte_future)
            shap_future = np.vstack(list(tqdm(pool.imap(func, batches), total=len(batches), desc="Ground Truth")))
        
        res = {
            "size": size,
            "mae_past_bg": float(mae_shap(shap_truth, shap_past)),
            "mae_future_bg": float(mae_shap(shap_truth, shap_future))
        }
        results.append(res)
        
    with open(out_dir / 'rq2.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_filar3_rq2()
