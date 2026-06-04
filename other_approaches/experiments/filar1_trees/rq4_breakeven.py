import time
import json
import shap
import pandas as pd
import joblib
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from data.data_loader import load_dataset
from cte_compression.kernel_thinning import build_cte_background, build_iid_background

def run_filar1_rq4():
    print("=== Filar 1 (TreeExplainer) - RQ4: Break-even Point ===")
    
    cache_dir = Path(__file__).parent.parent.parent / 'data' / 'cache_full'
    out_dir = Path(__file__).parent.parent.parent / 'results' / 'filar1_trees'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    model = joblib.load(cache_dir / 'xgb_model.pkl')
    data = load_dataset(data_dir=Path(__file__).parent.parent.parent / 'data' / 'ecom-offers')
    
    X_train, X_val = data['X_train'], data['X_val']
    
    # We will test explaining 10, 50, 100, 500, 1000 validation samples
    # to see at what point the cost of building CTE pays off
    val_explain_sizes = [10, 50, 100, 500, 1000]
    
    bg_size = 50 # Fixed compression target size for this test
    
    # 1. Measure compression time
    print(f"Measuring CTE Compression time for target size {bg_size}...")
    t0 = time.time()
    bg_cte = build_cte_background(X_train, target_size=bg_size, verbose=False)
    compression_time = time.time() - t0
    print(f"Compression time: {compression_time:.2f} s")
    
    # Pre-build Random background
    bg_rand_full = X_train # Truth
    
    explainer_cte = shap.TreeExplainer(model, data=bg_cte, model_output="probability", feature_perturbation="interventional")
    explainer_truth = shap.TreeExplainer(model, data=bg_rand_full, model_output="probability", feature_perturbation="interventional")
    
    results = []
    
    for n_val in tqdm(val_explain_sizes, desc="Number of Explained Samples"):
        X_val_explain = X_val.sample(min(n_val, len(X_val)), random_state=42).reset_index(drop=True)
        
        t0 = time.time()
        shap_cte = explainer_cte.shap_values(X_val_explain)
        explain_time_cte = time.time() - t0
        
        t0 = time.time()
        shap_truth = explainer_truth.shap_values(X_val_explain)
        explain_time_truth = time.time() - t0
        
        total_time_cte = compression_time + explain_time_cte
        total_time_truth = explain_time_truth
        
        res = {
            "num_val_samples": n_val,
            "cte": {
                "compression_time": compression_time,
                "explain_time": explain_time_cte,
                "total_time": total_time_cte
            },
            "truth_full_bg": {
                "explain_time": total_time_truth,
                "total_time": total_time_truth
            }
        }
        results.append(res)
        
    with open(out_dir / 'rq4.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to results/filar1_trees/rq4.json")

if __name__ == "__main__":
    run_filar1_rq4()
