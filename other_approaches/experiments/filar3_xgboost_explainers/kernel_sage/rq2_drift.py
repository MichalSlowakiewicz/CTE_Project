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
from evaluation.metrics import mae_shap

try:
    import sage
except ImportError:
    pass

def run_filar6_rq2():
    print("=== Filar 6 (KernelSAGE) - RQ2: Concept Drift ===")
    
    cache_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'cache_full'
    out_dir = Path(__file__).parent.parent.parent.parent / 'results' / 'filar3_xgboost_explainers' / 'kernel_sage'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    model = joblib.load(cache_dir / 'catboost_model.pkl')
    data = load_dataset(data_dir=Path(__file__).parent.parent.parent.parent / 'data' / 'ecom-offers')
    X_train, X_test, y_test = data['X_train'], data['X_test'], data['y_test']
    
    test_size = min(500, len(X_test))
    sampled_indices = X_test.sample(test_size, random_state=42).index
    X_test_sample = X_test.loc[sampled_indices].values
    y_test_sample = y_test[sampled_indices]
    
    bg_truth_test = X_test.sample(min(10000, len(X_test)), random_state=42).values
    imputer_truth = sage.MarginalImputer(model.predict_proba, bg_truth_test)
    estimator_truth = sage.KernelEstimator(imputer_truth, 'cross_entropy')
    
    print("Computing Ground Truth SAGE on Test background...")
    sage_truth_values = estimator_truth(X_test_sample, y_test_sample, n_jobs=-1, batch_size=256, bar=True).values
    
    bg_sizes = [50, 100, 200]
    results = []
    
    for size in tqdm(bg_sizes, desc="RQ2 (Filar 6) sizes"):
        bg_cte_past = pd.read_parquet(cache_dir / f'bg_cte_{size}.parquet').values
        imputer_past = sage.MarginalImputer(model.predict_proba, bg_cte_past)
        estimator_past = sage.KernelEstimator(imputer_past, 'cross_entropy')
        sage_past_values = estimator_past(X_test_sample, y_test_sample, n_jobs=-1, batch_size=256, bar=True).values
        
        bg_cte_future = build_cte_background(X_test, target_size=size, verbose=False).values
        imputer_future = sage.MarginalImputer(model.predict_proba, bg_cte_future)
        estimator_future = sage.KernelEstimator(imputer_future, 'cross_entropy')
        sage_future_values = estimator_future(X_test_sample, y_test_sample, n_jobs=-1, batch_size=256, bar=True).values
        
        mae_past = np.mean(np.abs(sage_truth_values - sage_past_values))
        mae_future = np.mean(np.abs(sage_truth_values - sage_future_values))
        
        res = {
            "size": size,
            "mae_past_bg": float(mae_past),
            "mae_future_bg": float(mae_future)
        }
        results.append(res)
        
    with open(out_dir / 'rq2.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_filar6_rq2()
