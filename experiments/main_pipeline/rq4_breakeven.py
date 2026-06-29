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
from cte_compression.kernel_thinning import build_cte_background
from experiments.main_pipeline.common import PAIRS, compute_shap

def run_all_rq4():
    print("=== Main Pipeline - RQ4: Breakeven Time ===")
    
    cache_dir = Path(__file__).parent.parent.parent / 'data' / 'cache_full'
    out_dir = Path(__file__).parent.parent.parent / 'results' / 'main_pipeline'
    
    data = load_dataset(data_dir=Path(__file__).parent.parent.parent / 'data' / 'ecom-offers')
    X_train, X_val = data['X_train'], data['X_val']
    
    # We explain 100 validation samples
    X_val_explain = X_val.sample(min(100, len(X_val)), random_state=42).reset_index(drop=True)
    
    # We measure how long it takes to COMPRESS 100k points to 50 points
    target_size = 50
    print(f"\nMeasuring compression time for target_size={target_size}...")
    t0 = time.time()
    # We pass use_kt=True to match our actual implementation of build_iid_background vs CTE
    build_cte_background(X_train, target_size=target_size, verbose=False)
    compression_time = time.time() - t0
    print(f"Compression took {compression_time:.2f} seconds.")
    
    for pair in PAIRS:
        pair_out_dir = out_dir / pair['name']
        pair_out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n---> Starting RQ4 for pair: {pair['name']}")
        
        try:
            model = joblib.load(cache_dir / pair['model_file'])
        except Exception as e:
            print(f"Skipping {pair['name']} due to model load error: {e}")
            continue
            
        try:
            # Time for full ground truth (100k)
            bg_truth_full = pd.read_parquet(cache_dir / 'bg_truth_full.parquet')
            t0 = time.time()
            compute_shap(pair, model, bg_truth_full, X_val_explain)
            truth_time_100 = time.time() - t0
            
            rq4_experiments = []
            
            for size in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
                print(f"   Testing breakeven for CTE size {size}...")
                # 1. Compression (Kernel Thinning build time)
                t_build_start = time.time()
                build_cte_background(X_train, target_size=size, verbose=False)
                build_time = time.time() - t_build_start
                
                # 2. SHAP computation time
                bg_cte = pd.read_parquet(cache_dir / f'bg_cte_{size}.parquet')
                t_shap_start = time.time()
                compute_shap(pair, model, bg_cte, X_val_explain)
                cte_time_100 = time.time() - t_shap_start
                
                delta = truth_time_100 - cte_time_100
                
                if delta > 0:
                    n_star_batches = build_time / delta
                    n_star_samples = n_star_batches * 100
                else:
                    n_star_batches = None
                    n_star_samples = None
                    
                rq4_experiments.append({
                    "size": size,
                    "build_time": build_time,
                    "truth_time_per_batch": truth_time_100,
                    "cte_time_per_batch": cte_time_100,
                    "delta": delta,
                    "n_star_batches": n_star_batches,
                    "n_star_samples": n_star_samples
                })
                
            final_json = {
                "n_explain_per_batch": 100,
                "experiments": rq4_experiments
            }
            
            with open(pair_out_dir / 'rq4.json', 'w') as f:
                json.dump(final_json, f, indent=2)
            print(f"   RQ4 finished for {pair['name']}.")
            
        except Exception as e:
            print(f"Error computing RQ4 for {pair['name']}: {e}")

if __name__ == "__main__":
    run_all_rq4()
