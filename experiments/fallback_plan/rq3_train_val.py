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
from evaluation.metrics import mae_shap
from experiments.fallback_plan.common import PAIRS, load_mlp_model, compute_shap

def run_all_rq3():
    print("=== Fallback Plan - RQ3: Train vs Val Background ===")
    
    cache_dir = Path(__file__).parent.parent.parent / 'data' / 'cache_full'
    out_dir = Path(__file__).parent.parent.parent / 'results' / 'fallback_plan'
    
    data = load_dataset(data_dir=Path(__file__).parent.parent.parent / 'data' / 'ecom-offers')
    X_train, X_val = data['X_train'], data['X_val']
    
    # We explain validation data
    X_val_explain = X_val.sample(min(100, len(X_val)), random_state=42).reset_index(drop=True)
    
    print("\nLoading Full Train and Val Backgrounds...")
    # RQ3 compares SHAP computed with CTE(Train) vs CTE(Val)
    bg_truth_train = pd.read_parquet(cache_dir / 'bg_truth_full.parquet')
    bg_truth_val = X_val  # Full val set
    
    bg_sizes = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    
    for pair in PAIRS:
        pair_out_dir = out_dir / pair['name']
        pair_out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n---> Starting RQ3 for pair: {pair['name']}")
        
        try:
            if pair['family'] == 'neural':
                model = load_mlp_model(cache_dir, X_train.shape[1])
            else:
                model = joblib.load(cache_dir / pair['model_file'])
        except Exception as e:
            print(f"Skipping {pair['name']} due to model load error: {e}")
            continue
            
        try:
            print("   Computing Ground Truth on Train and Val full datasets...")
            shap_truth_train = compute_shap(pair, model, bg_truth_train, X_val_explain)
            shap_truth_val = compute_shap(pair, model, bg_truth_val, X_val_explain)
            
            # Wyciągamy średnie wartości SHAP dla każdej cechy
            truth_imp_train = np.mean(np.abs(shap_truth_train), axis=0).tolist()
            truth_imp_val = np.mean(np.abs(shap_truth_val), axis=0).tolist()
            
            # Jak duża jest różnica bazowa między pełnym tłem Train a Val?
            baseline_diff = {
                "global_mae": float(mae_shap(shap_truth_train, shap_truth_val, global_importance=True)),
                "local_mae": float(mae_shap(shap_truth_train, shap_truth_val, global_importance=False))
            }
            
            rq3_results = []
            for size in bg_sizes:
                # 1. CTE na zbiorze Treningowym
                bg_cte_train = pd.read_parquet(cache_dir / f'bg_cte_{size}.parquet')
                shap_cte_train = compute_shap(pair, model, bg_cte_train, X_val_explain)
                
                res = {
                    "size": size,
                    "cte_train_vs_truth_val": {
                        "global_mae": float(mae_shap(shap_cte_train, shap_truth_val, global_importance=True)),
                        "local_mae": float(mae_shap(shap_cte_train, shap_truth_val, global_importance=False))
                    }
                }
                rq3_results.append(res)
                
            final_json = {
                "feature_names": list(X_train.columns),
                "train": {"truth_importance": truth_imp_train},
                "val": {"truth_importance": truth_imp_val},
                "baseline_diff_train_val": baseline_diff,
                "experiments": rq3_results
            }
                
            with open(pair_out_dir / 'rq3.json', 'w') as f:
                json.dump(final_json, f, indent=2)
            print(f"   RQ3 finished successfully.")
            
        except Exception as e:
            print(f"Error computing RQ3 for {pair['name']}: {e}")

if __name__ == "__main__":
    run_all_rq3()
