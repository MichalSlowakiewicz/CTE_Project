import time
import json
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from data.data_loader import load_dataset
from evaluation.metrics import mae_shap

try:
    import sage
except ImportError:
    pass

def run_filar6_rq1():
    print("=== Filar 6 (KernelSAGE) - RQ1: Efficiency & Accuracy ===")
    
    cache_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'cache_full'
    out_dir = Path(__file__).parent.parent.parent.parent / 'results' / 'filar3_xgboost_explainers' / 'kernel_sage'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    model = joblib.load(cache_dir / 'catboost_model.pkl')
    data = load_dataset(data_dir=Path(__file__).parent.parent.parent.parent / 'data' / 'ecom-offers')
    X_val, y_val = data['X_val'], data['y_val']
    
    sample_size = min(500, len(X_val))
    sampled_indices = X_val.sample(sample_size, random_state=42).index
    X_val_sample = X_val.loc[sampled_indices].values
    y_val_sample = y_val[sampled_indices]
    
    # Ground truth (Limit to 10000 to match the 20x paper methodology for max size 200)
    bg_truth_full = pd.read_parquet(cache_dir / 'bg_truth_full.parquet')
    bg_truth_sampled = bg_truth_full.sample(1000, random_state=42).values
    
    imputer_truth = sage.MarginalImputer(model.predict_proba, bg_truth_sampled)
    estimator_truth = sage.KernelEstimator(imputer_truth, 'cross_entropy')
    
    print("Computing Ground Truth SAGE (KernelSAGE)...")
    t0 = time.time()
    sage_truth_values = estimator_truth(X_val_sample, y_val_sample, n_jobs=-1, batch_size=256, bar=True).values
    time_truth = time.time() - t0
    print(f"Ground Truth computed in {time_truth:.2f} s")
    
    bg_sizes = [10, 50, 100, 200]
    results = []
    
    for size in tqdm(bg_sizes, desc="CTE Sizes"):
        bg_cte = pd.read_parquet(cache_dir / f'bg_cte_{size}.parquet').values
        
        imputer_cte = sage.MarginalImputer(model.predict_proba, bg_cte)
        estimator_cte = sage.KernelEstimator(imputer_cte, 'cross_entropy')
        
        t0 = time.time()
        sage_cte_values = estimator_cte(X_val_sample, y_val_sample, n_jobs=-1, batch_size=256, bar=True).values
        time_cte = time.time() - t0
        
        mae = float(np.mean(np.abs(sage_truth_values - sage_cte_values)))
        
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
    run_filar6_rq1()
