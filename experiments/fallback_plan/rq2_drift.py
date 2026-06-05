"""
rq2_drift.py — RQ2: Temporal Feature Importance Drift

Question: Do feature importances shift across chronological validation windows,
           and does a fixed CTE background track the shifts?

Design:
  - Split the TEST set by its natural time periods (6 periods in TabReD ecom_offers)
  - For each period, explain 100 samples using:
      (a) Ground truth: full train background (109k points)
      (b) CTE: pre-built coreset of size N=128 (from train)
      (c) Random(×30): 30 independent random subsamples of size 128 (from train)
  - Measure: MAE and Relative Error of CTE vs truth, Random vs truth
  - Output: per-period feature importance vectors for drift visualization

The key insight: both CTE and Random backgrounds are FIXED from training data.
As we move to later test periods (further from training distribution),
we test whether CTE's kernel-optimized summary degrades faster or slower than Random.
"""

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
from evaluation.metrics import mae_shap, relative_error_shap
from experiments.fallback_plan.common import PAIRS, compute_shap

# Number of independent random sampling repetitions for statistical rigor
N_REPS = 30
TARGET_SIZE = 128  # Background size for CTE and Random


def run_all_rq2():
    print(f"=== RQ2: Temporal Drift (Period-based, Random: {N_REPS} reps) ===")

    cache_dir = Path(__file__).parent.parent.parent / 'data' / 'cache_full'
    out_dir = Path(__file__).parent.parent.parent / 'results' / 'fallback_plan'

    data = load_dataset(data_dir=Path(__file__).parent.parent.parent / 'data' / 'ecom-offers')
    X_train, X_test = data['X_train'], data['X_test']
    feature_names = list(X_train.columns)

    # Load period metadata for test set
    periods_test = np.load(
        Path(__file__).parent.parent.parent / 'data' / 'ecom-offers' / 'periods_test.npy'
    )
    unique_periods = np.unique(periods_test)
    print(f"\nTest set: {len(X_test):,} samples, {len(unique_periods)} periods: {unique_periods}")

    # Backgrounds
    print(f"\nLoading backgrounds (truth=full train, CTE=N={TARGET_SIZE})...")
    bg_truth = pd.read_parquet(cache_dir / 'bg_truth_full.parquet')
    bg_cte = pd.read_parquet(cache_dir / f'bg_cte_{TARGET_SIZE}.parquet')

    for pair in PAIRS:
        pair_out_dir = out_dir / pair['name']
        pair_out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n---> Starting RQ2 for pair: {pair['name']}")

        try:
            model = joblib.load(cache_dir / pair['model_file'])
        except Exception as e:
            print(f"Skipping {pair['name']} due to model load error: {e}")
            continue

        try:
            windows_results = []

            for period in unique_periods:
                period_mask = periods_test == period
                X_period = X_test[period_mask].reset_index(drop=True)

                # Sample up to 100 points from this period
                n_explain = min(100, len(X_period))
                X_explain = X_period.sample(n_explain, random_state=42).reset_index(drop=True)

                print(f"   Period {period} ({len(X_period):,} pts, explaining {n_explain})...")

                # Ground truth
                shap_truth = compute_shap(pair, model, bg_truth, X_explain)
                truth_imp = np.mean(np.abs(shap_truth), axis=0).tolist()

                # CTE
                shap_cte = compute_shap(pair, model, bg_cte, X_explain)
                cte_imp = np.mean(np.abs(shap_cte), axis=0).tolist()
                cte_mae_global = float(mae_shap(shap_truth, shap_cte, global_importance=True))
                cte_re_global = float(relative_error_shap(shap_truth, shap_cte, global_importance=True))

                # Random (×N_REPS)
                rand_imp_all = []
                rand_mae_globals = []
                rand_re_globals = []
                for rep in range(N_REPS):
                    seed = 42 + rep
                    bg_rand = build_iid_background(X_train, size=TARGET_SIZE, seed=seed)
                    shap_rand = compute_shap(pair, model, bg_rand, X_explain)
                    rand_imp_all.append(np.mean(np.abs(shap_rand), axis=0))
                    rand_mae_globals.append(float(mae_shap(shap_truth, shap_rand, global_importance=True)))
                    rand_re_globals.append(float(relative_error_shap(shap_truth, shap_rand, global_importance=True)))

                rand_imp_arr = np.array(rand_imp_all)

                windows_results.append({
                    "period": int(period),
                    "n_samples": len(X_period),
                    "n_explained": n_explain,
                    "truth_importance": truth_imp,
                    "cte_importance": cte_imp,
                    "cte_mae_global": cte_mae_global,
                    "cte_re_global": cte_re_global,
                    "rand_importance_mean": np.mean(rand_imp_arr, axis=0).tolist(),
                    "rand_importance_std": np.std(rand_imp_arr, axis=0).tolist(),
                    "rand_mae_global_mean": float(np.mean(rand_mae_globals)),
                    "rand_mae_global_std": float(np.std(rand_mae_globals)),
                    "rand_re_global_mean": float(np.mean(rand_re_globals)),
                    "rand_re_global_std": float(np.std(rand_re_globals)),
                    "n_reps": N_REPS,
                })

            res = {
                "bg_size": TARGET_SIZE,
                "feature_names": feature_names,
                "periods": [int(p) for p in unique_periods],
                "windows": windows_results,
            }

            with open(pair_out_dir / 'rq2.json', 'w') as f:
                json.dump(res, f, indent=2)
            print(f"   RQ2 finished successfully.")

        except Exception as e:
            import traceback
            print(f"Error computing RQ2 for {pair['name']}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    run_all_rq2()
