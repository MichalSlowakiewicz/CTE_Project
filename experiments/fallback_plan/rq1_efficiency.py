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
from cte_compression.kernel_thinning import build_iid_background
from evaluation.metrics import mae_shap, relative_error_shap, speedup_ratio
from experiments.fallback_plan.common import PAIRS, compute_shap

# Number of independent random sampling repetitions for statistical rigor
N_REPS = 30


def run_all_rqs():
    print(f"=== Fallback Plan: Running All {len(PAIRS)} Pairs (Random: {N_REPS} reps) ===")
    
    cache_dir = Path(__file__).parent.parent.parent / 'data' / 'cache_full'
    out_dir = Path(__file__).parent.parent.parent / 'results' / 'fallback_plan'
    
    data = load_dataset(data_dir=Path(__file__).parent.parent.parent / 'data' / 'ecom-offers')
    X_train, X_val = data['X_train'], data['X_val']
    
    # We explain 100 validation samples
    X_val_explain = X_val.sample(min(100, len(X_val)), random_state=42).reset_index(drop=True)
    
    # GROUND TRUTH: We can use the full 100,000 points because these are model-specific fast explainers!
    print("\nLoading Full 100k Ground Truth Background...")
    bg_truth_full = pd.read_parquet(cache_dir / 'bg_truth_full.parquet')
    
    bg_sizes = [4, 8, 16, 32, 64, 128, 256]
    
    for pair in PAIRS:
        pair_out_dir = out_dir / pair['name']
        pair_out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n---> Starting pair: {pair['name']} ({pair['explainer']})")
        
        # 1. Load Model
        try:
            model = joblib.load(cache_dir / pair['model_file'])
        except Exception as e:
            print(f"Skipping {pair['name']} due to model load error: {e}")
            continue
            
        # 2. Compute Ground Truth (RQ1)
        try:
            print(f"   Computing Ground Truth for {pair['name']} (100k points)...")
            t0 = time.time()
            shap_truth = compute_shap(pair, model, bg_truth_full, X_val_explain)
            time_truth = time.time() - t0
            print(f"   Ground Truth computed in {time_truth:.2f}s")
            
            # RQ1: Efficiency vs Fidelity
            rq1_results = []
            for size in bg_sizes:
                bg_cte = pd.read_parquet(cache_dir / f'bg_cte_{size}.parquet')
                
                t0 = time.time()
                shap_cte = compute_shap(pair, model, bg_cte, X_val_explain)
                t_cte = time.time() - t0
                
                # --- Random sampling: N_REPS repetitions ---
                rand_mae_globals = []
                rand_mae_locals = []
                rand_re_globals = []
                rand_re_locals = []
                rand_times = []
                
                for rep in range(N_REPS):
                    seed = 42 + rep
                    bg_rand = build_iid_background(X_train, size=size, seed=seed)
                    t0 = time.time()
                    shap_rand = compute_shap(pair, model, bg_rand, X_val_explain)
                    t_rand = time.time() - t0
                    
                    rand_mae_globals.append(float(mae_shap(shap_truth, shap_rand, global_importance=True)))
                    rand_mae_locals.append(float(mae_shap(shap_truth, shap_rand, global_importance=False)))
                    rand_re_globals.append(float(relative_error_shap(shap_truth, shap_rand, global_importance=True)))
                    rand_re_locals.append(float(relative_error_shap(shap_truth, shap_rand, global_importance=False)))
                    rand_times.append(t_rand)
                
                res = {
                    "size": size,
                    "cte": {
                        "mae_global": float(mae_shap(shap_truth, shap_cte, global_importance=True)),
                        "mae_local": float(mae_shap(shap_truth, shap_cte, global_importance=False)),
                        "re_global": float(relative_error_shap(shap_truth, shap_cte, global_importance=True)),
                        "re_local": float(relative_error_shap(shap_truth, shap_cte, global_importance=False)),
                        "speedup_vs_truth": float(speedup_ratio(time_truth, t_cte))
                    },
                    "random": {
                        "mae_global_mean": float(np.mean(rand_mae_globals)),
                        "mae_global_std": float(np.std(rand_mae_globals)),
                        "mae_local_mean": float(np.mean(rand_mae_locals)),
                        "mae_local_std": float(np.std(rand_mae_locals)),
                        "re_global_mean": float(np.mean(rand_re_globals)),
                        "re_global_std": float(np.std(rand_re_globals)),
                        "re_local_mean": float(np.mean(rand_re_locals)),
                        "re_local_std": float(np.std(rand_re_locals)),
                        "speedup_vs_truth": float(speedup_ratio(time_truth, np.mean(rand_times))),
                        "n_reps": N_REPS
                    }
                }
                rq1_results.append(res)
                print(f"      N={size}: CTE_g={res['cte']['mae_global']:.2e} (RE={res['cte']['re_global']:.3f}), "
                      f"Rand_g={res['random']['mae_global_mean']:.2e}±{res['random']['mae_global_std']:.2e}")
                
            with open(pair_out_dir / 'rq1.json', 'w') as f:
                json.dump({"truth_time": time_truth, "experiments": rq1_results}, f, indent=2)
            print(f"   RQ1 finished successfully.")
            
        except Exception as e:
            import traceback
            print(f"Error computing RQ1 for {pair['name']}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    run_all_rqs()

