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
from cte_compression.kernel_thinning import build_iid_background
from evaluation.metrics import speedup_ratio

try:
    import sage
except ImportError:
    pass

def run_filar4_rq1():
    print("=== Filar 4 (Wyjaśnienia Globalne SAGE) - RQ1: Efficiency vs Fidelity ===")
    
    cache_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'cache_full'
    out_dir = Path(__file__).parent.parent.parent.parent / 'results' / 'filar3_xgboost_explainers' / 'permutation_sage'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    model = joblib.load(cache_dir / 'catboost_model.pkl')
    data = load_dataset(data_dir=Path(__file__).parent.parent.parent.parent / 'data' / 'ecom-offers')
    X_train, X_val, y_val = data['X_train'], data['X_val'], data['y_val']
    
    # We use a subset of validation for estimating global SAGE to save time
    val_size = min(500, len(X_val))
    sampled_indices = X_val.sample(val_size, random_state=42).index
    X_val_sample = X_val.loc[sampled_indices].values
    y_val_sample = y_val[sampled_indices]
    
    print("Loading Ground Truth Background...")
    bg_truth = pd.read_parquet(cache_dir / 'bg_truth_full.parquet').values
    
    # SAGE uses MarginalImputer
    imputer_truth = sage.MarginalImputer(model.predict_proba, bg_truth)
    estimator_truth = sage.PermutationEstimator(imputer_truth, 'cross_entropy')
    
    t0 = time.time()
    sage_truth_obj = estimator_truth(X_val_sample, y_val_sample, n_jobs=-1, batch_size=256, bar=True)
    time_truth = time.time() - t0
    sage_truth_values = sage_truth_obj.values
    
    bg_sizes = [10, 50, 100, 200]
    results = []
    
    for size in tqdm(bg_sizes, desc="RQ1 (Filar 4) sizes"):
        bg_cte = pd.read_parquet(cache_dir / f'bg_cte_{size}.parquet').values
        imputer_cte = sage.MarginalImputer(model.predict_proba, bg_cte)
        estimator_cte = sage.PermutationEstimator(imputer_cte, 'cross_entropy')
        
        t0 = time.time()
        sage_cte_obj = estimator_cte(X_val_sample, y_val_sample, n_jobs=-1, batch_size=256, bar=True)
        t_cte = time.time() - t0
        sage_cte_values = sage_cte_obj.values
        
        bg_rand = build_iid_background(X_train, size=size).values
        imputer_rand = sage.MarginalImputer(model.predict_proba, bg_rand)
        estimator_rand = sage.PermutationEstimator(imputer_rand, 'cross_entropy')
        
        t0 = time.time()
        sage_rand_obj = estimator_rand(X_val_sample, y_val_sample, n_jobs=-1, batch_size=256, bar=True)
        t_rand = time.time() - t0
        sage_rand_values = sage_rand_obj.values
        
        mae_cte = np.mean(np.abs(sage_truth_values - sage_cte_values))
        mae_rand = np.mean(np.abs(sage_truth_values - sage_rand_values))
        
        res = {
            "size": size,
            "cte": {"mae": float(mae_cte), "speedup": float(speedup_ratio(time_truth, t_cte))},
            "random": {"mae": float(mae_rand), "speedup": float(speedup_ratio(time_truth, t_rand))}
        }
        results.append(res)
        
    with open(out_dir / 'rq1.json', 'w') as f:
        json.dump({"truth_time": time_truth, "experiments": results}, f, indent=2)
    print("\nResults saved to results/filar4_sage/rq1.json")

if __name__ == "__main__":
    run_filar4_rq1()
