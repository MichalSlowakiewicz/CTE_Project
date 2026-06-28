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
from evaluation.metrics import mae_shap, relative_error_shap, speedup_ratio
from experiments.main_pipeline.common import PAIRS, compute_shap

# Number of independent repetitions for both CTE and random sampling.
N_REPS = 30


def run_all_rqs():
    print(f"=== Main Pipeline: Running All {len(PAIRS)} Pairs ===")
    print(f"    CTE ensemble: {N_REPS} seeds  |  Random: {N_REPS} reps")

    cache_dir = Path(__file__).parent.parent.parent / 'data' / 'cache_full'
    out_dir = Path(__file__).parent.parent.parent / 'results' / 'main_pipeline'

    data = load_dataset(data_dir=Path(__file__).parent.parent.parent / 'data' / 'ecom-offers')
    X_train, X_val = data['X_train'], data['X_val']

    # We explain 100 validation samples
    X_val_explain = X_val.sample(min(100, len(X_val)), random_state=42).reset_index(drop=True)

    # GROUND TRUTH: full 100k+ points
    print("\nLoading Full 100k Ground Truth Background...")
    bg_truth_full = pd.read_parquet(cache_dir / 'bg_truth_full.parquet')

    bg_sizes = [4, 8, 16, 32, 64, 128, 256]

    # Pre-generate CTE and Random backgrounds to save massive amounts of time
    # (avoid recomputing the exact same sets for every model)
    print("\nPre-computing CTE and Random backgrounds for all sizes and seeds...")
    bg_ctes_dict = {}
    bg_rands_dict = {}
    cte_compute_times = {}

    for size in bg_sizes:
        print(f"   Size {size}...")
        bg_ctes_dict[size] = []
        bg_rands_dict[size] = []
        compute_times = []
        
        for rep in tqdm(range(N_REPS), desc=f"Generating CTE (N={size})", leave=False):
            t0 = time.time()
            bg_ctes_dict[size].append(build_cte_background(X_train, target_size=size, verbose=False, seed=rep))
            compute_times.append(time.time() - t0)
            
            bg_rands_dict[size].append(build_iid_background(X_train, size=size, seed=42 + rep))
            
        cte_compute_times[size] = compute_times

    print("\nBackground pre-computation complete. Starting evaluations.")

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

        # 2. Compute Ground Truth
        try:
            print(f"   Computing Ground Truth for {pair['name']} ({len(bg_truth_full)} points)...")
            t0 = time.time()
            shap_truth = compute_shap(pair, model, bg_truth_full, X_val_explain)
            time_truth = time.time() - t0
            print(f"   Ground Truth computed in {time_truth:.2f}s")

            # RQ1: Efficiency vs Fidelity
            rq1_results = []
            for size in bg_sizes:

                # ─── CTE Ensemble: N_REPS independent coresets ───
                cte_mae_globals = []
                cte_mae_locals = []
                cte_re_globals = []
                cte_re_locals = []
                cte_times = []

                for rep in range(N_REPS):
                    t0 = time.time()
                    shap_cte = compute_shap(pair, model, bg_ctes_dict[size][rep], X_val_explain)
                    t_shap = time.time() - t0
                    
                    # Total time = precomputed compression time + SHAP time
                    t_total = cte_compute_times[size][rep] + t_shap

                    cte_mae_globals.append(float(mae_shap(shap_truth, shap_cte, global_importance=True)))
                    cte_mae_locals.append(float(mae_shap(shap_truth, shap_cte, global_importance=False)))
                    cte_re_globals.append(float(relative_error_shap(shap_truth, shap_cte, global_importance=True)))
                    cte_re_locals.append(float(relative_error_shap(shap_truth, shap_cte, global_importance=False)))
                    cte_times.append(t_total)

                # ─── Random sampling: N_REPS repetitions ───
                rand_mae_globals = []
                rand_mae_locals = []
                rand_re_globals = []
                rand_re_locals = []
                rand_times = []

                for rep in range(N_REPS):
                    t0 = time.time()
                    shap_rand = compute_shap(pair, model, bg_rands_dict[size][rep], X_val_explain)
                    t_rand = time.time() - t0

                    rand_mae_globals.append(float(mae_shap(shap_truth, shap_rand, global_importance=True)))
                    rand_mae_locals.append(float(mae_shap(shap_truth, shap_rand, global_importance=False)))
                    rand_re_globals.append(float(relative_error_shap(shap_truth, shap_rand, global_importance=True)))
                    rand_re_locals.append(float(relative_error_shap(shap_truth, shap_rand, global_importance=False)))
                    rand_times.append(t_rand)

                res = {
                    "size": size,
                    "cte": {
                        "mae_global_mean": float(np.mean(cte_mae_globals)),
                        "mae_global_std": float(np.std(cte_mae_globals)),
                        "mae_local_mean": float(np.mean(cte_mae_locals)),
                        "mae_local_std": float(np.std(cte_mae_locals)),
                        "re_global_mean": float(np.mean(cte_re_globals)),
                        "re_global_std": float(np.std(cte_re_globals)),
                        "re_local_mean": float(np.mean(cte_re_locals)),
                        "re_local_std": float(np.std(cte_re_locals)),
                        "speedup_vs_truth": float(speedup_ratio(time_truth, np.mean(cte_times))),
                        "n_reps": N_REPS,
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
                print(f"      N={size}: CTE_g={res['cte']['mae_global_mean']:.2e}±{res['cte']['mae_global_std']:.2e}, "
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
