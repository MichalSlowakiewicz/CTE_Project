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

def run_filar6_rq3():
    print("=== Filar 6 (KernelSAGE) - RQ3: Train vs Val Discrepancy ===")
    
    cache_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'cache_full'
    out_dir = Path(__file__).parent.parent.parent.parent / 'results' / 'filar3_xgboost_explainers' / 'kernel_sage'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    model = joblib.load(cache_dir / 'catboost_model.pkl')
    data = load_dataset(data_dir=Path(__file__).parent.parent.parent.parent / 'data' / 'ecom-offers')
    X_train, X_val = data['X_train'], data['X_val']
    y_train, y_val = data['y_train'], data['y_val']
    
    sample_size = min(500, len(X_val))
    
    idx_train = X_train.sample(sample_size, random_state=42).index
    X_train_sample, y_train_sample = X_train.loc[idx_train].values, y_train[idx_train]
    
    idx_val = X_val.sample(sample_size, random_state=42).index
    X_val_sample, y_val_sample = X_val.loc[idx_val].values, y_val[idx_val]
    
    bg_truth_sampled = pd.read_parquet(cache_dir / 'bg_truth_full.parquet').sample(1000, random_state=42).values
    
    imputer_truth = sage.MarginalImputer(model.predict_proba, bg_truth_sampled)
    estimator_truth = sage.KernelEstimator(imputer_truth, 'cross_entropy')
    
    print("Computing Ground Truth SAGE (Train vs Val)...")
    sage_truth_train = estimator_truth(X_train_sample, y_train_sample, n_jobs=-1, batch_size=256, bar=True).values
    sage_truth_val = estimator_truth(X_val_sample, y_val_sample, n_jobs=-1, batch_size=256, bar=True).values
    
    bg_sizes = [10, 50, 100, 200]
    results = []
    
    for size in tqdm(bg_sizes, desc="RQ3 (Filar 6) sizes"):
        bg_cte = pd.read_parquet(cache_dir / f'bg_cte_{size}.parquet').values
        
        imputer_cte = sage.MarginalImputer(model.predict_proba, bg_cte)
        estimator_cte = sage.KernelEstimator(imputer_cte, 'cross_entropy')
        
        sage_cte_train = estimator_cte(X_train_sample, y_train_sample, n_jobs=-1, batch_size=256, bar=True).values
        sage_cte_val = estimator_cte(X_val_sample, y_val_sample, n_jobs=-1, batch_size=256, bar=True).values
        
        mae_train = np.mean(np.abs(sage_truth_train - sage_cte_train))
        mae_val = np.mean(np.abs(sage_truth_val - sage_cte_val))
        
        res = {
            "size": size,
            "mae_train_set": float(mae_train),
            "mae_val_set": float(mae_val)
        }
        results.append(res)
        
    with open(out_dir / 'rq3.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_filar6_rq3()
