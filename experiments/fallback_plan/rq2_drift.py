import time
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from data.data_loader import load_dataset
from cte_compression.kernel_thinning import build_cte_background, build_iid_background
from evaluation.metrics import mae_shap
from experiments.fallback_plan.common import PAIRS, load_mlp_model, compute_shap

def run_all_rq2():
    print("=== Fallback Plan - RQ2: Concept Drift (Temporal Windows) ===")
    
    cache_dir = Path(__file__).parent.parent.parent / 'data' / 'cache_full'
    out_dir = Path(__file__).parent.parent.parent / 'results' / 'fallback_plan'
    
    data = load_dataset(data_dir=Path(__file__).parent.parent.parent / 'data' / 'ecom-offers')
    X_train, X_test = data['X_train'], data['X_test']
    feature_names = list(X_train.columns)
    
    # GROUND TRUTH BACKGROUND: Full Train set
    print(f"\nLoading Full Train set as Ground Truth Background ({len(X_train)} points)...")
    bg_truth = pd.read_parquet(cache_dir / 'bg_truth_full.parquet')
    
    # We will use N=128 for the CTE and Random backgrounds to match the power of two scaling
    target_size = 128
    bg_cte = pd.read_parquet(cache_dir / f'bg_cte_{target_size}.parquet')
    bg_rand = build_iid_background(X_train, size=target_size)
    
    # Create 5 chronological windows from the test set
    num_windows = 5
    window_size = len(X_test) // num_windows
    
    for pair in PAIRS:
        pair_out_dir = out_dir / pair['name']
        pair_out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n---> Starting RQ2 for pair: {pair['name']}")
        
        try:
            if pair['family'] == 'neural':
                model = load_mlp_model(cache_dir, X_train.shape[1])
            else:
                model = joblib.load(cache_dir / pair['model_file'])
        except Exception as e:
            print(f"Skipping {pair['name']} due to model load error: {e}")
            continue
            
        try:
            windows_results = []
            
            for w in range(num_windows):
                start_idx = w * window_size
                # If it's the last window, take the rest
                end_idx = (w + 1) * window_size if w < num_windows - 1 else len(X_test)
                
                X_window = X_test.iloc[start_idx:end_idx]
                
                # To keep computation fast, we explain a sample of 100 points from each window
                X_window_explain = X_window.sample(min(100, len(X_window)), random_state=42).reset_index(drop=True)
                
                print(f"   Computing Window {w+1}/{num_windows}...")
                
                shap_truth = compute_shap(pair, model, bg_truth, X_window_explain)
                shap_cte = compute_shap(pair, model, bg_cte, X_window_explain)
                shap_rand = compute_shap(pair, model, bg_rand, X_window_explain)
                
                truth_imp = np.mean(np.abs(shap_truth), axis=0).tolist()
                cte_imp = np.mean(np.abs(shap_cte), axis=0).tolist()
                rand_imp = np.mean(np.abs(shap_rand), axis=0).tolist()
                
                windows_results.append({
                    "window": w + 1,
                    "truth_importance": truth_imp,
                    "cte_importance": cte_imp,
                    "rand_importance": rand_imp
                })
                
            res = {
                "feature_names": feature_names,
                "windows": windows_results
            }
            
            with open(pair_out_dir / 'rq2.json', 'w') as f:
                json.dump(res, f, indent=2)
            print(f"   RQ2 finished successfully.")
            
        except Exception as e:
            print(f"Error computing RQ2 for {pair['name']}: {e}")

if __name__ == "__main__":
    run_all_rq2()
