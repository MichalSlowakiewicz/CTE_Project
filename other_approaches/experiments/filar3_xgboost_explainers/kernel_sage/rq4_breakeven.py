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
from cte_compression.kernel_thinning import build_cte_background

try:
    import sage
except ImportError:
    pass

def run_filar6_rq4():
    print("=== Filar 6 (KernelSAGE) - RQ4: Break-even Point ===")
    
    cache_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'cache_full'
    out_dir = Path(__file__).parent.parent.parent.parent / 'results' / 'filar3_xgboost_explainers' / 'kernel_sage'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    model = joblib.load(cache_dir / 'catboost_model.pkl')
    data = load_dataset(data_dir=Path(__file__).parent.parent.parent.parent / 'data' / 'ecom-offers')
    X_train, X_val, y_val = data['X_train'], data['X_val'], data['y_val']
    
    val_explain_sizes = [50, 100, 200, 500, 1000]
    bg_size = 50 
    
    print(f"Measuring CTE Compression time for target size {bg_size}...")
    t0 = time.time()
    bg_cte = build_cte_background(X_train, target_size=bg_size, verbose=False).values
    compression_time = time.time() - t0
    
    bg_truth_sampled = pd.read_parquet(cache_dir / 'bg_truth_full.parquet').sample(1000, random_state=42).values
    
    imputer_cte = sage.MarginalImputer(model.predict_proba, bg_cte)
    estimator_cte = sage.KernelEstimator(imputer_cte, 'cross_entropy')
    
    imputer_truth = sage.MarginalImputer(model.predict_proba, bg_truth_sampled)
    estimator_truth = sage.KernelEstimator(imputer_truth, 'cross_entropy')
    
    results = []
    
    for n_val in tqdm(val_explain_sizes, desc="Explained Samples"):
        idx_val = X_val.sample(min(n_val, len(X_val)), random_state=42).index
        X_val_sample, y_val_sample = X_val.loc[idx_val].values, y_val[idx_val]
        
        t0 = time.time()
        _ = estimator_cte(X_val_sample, y_val_sample, n_jobs=-1, batch_size=256, bar=True)
        explain_time_cte = time.time() - t0
        
        t0 = time.time()
        _ = estimator_truth(X_val_sample, y_val_sample, n_jobs=-1, batch_size=256, bar=True)
        explain_time_truth = time.time() - t0
        
        res = {
            "num_val_samples": n_val,
            "cte": {
                "compression_time": compression_time,
                "explain_time": explain_time_cte,
                "total_time": compression_time + explain_time_cte
            },
            "truth_bg_10000": {
                "explain_time": explain_time_truth,
                "total_time": explain_time_truth
            }
        }
        results.append(res)
        
    with open(out_dir / 'rq4.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_filar6_rq4()
