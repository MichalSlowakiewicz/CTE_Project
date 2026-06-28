"""
rq3_train_val.py — RQ3: Train vs Validation Feature Drift

Question: Do SHAP values differ between training and validation data,
          and which features drift most?

Design:
  - Explain samples from BOTH train and val sets using the full train background
  - Compare mean(|SHAP|) per feature between train and val → feature drift ranking
  - Repeat with CTE ensemble(×30) and Random(×30) backgrounds:
      Does CTE preserve the train→val importance shift better than Random?
  - Measures: MAE between truth and approximation on both train/val explained samples
              + per-feature drift magnitude
"""

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
from evaluation.metrics import mae_shap, relative_error_shap
from experiments.main_pipeline.common import PAIRS, compute_shap

N_REPS = 30
TARGET_SIZE = 128
N_EXPLAIN = 200


def run_all_rq3():
    print(f"=== RQ3: Train vs Val Feature Drift (CTE Ensemble: {N_REPS} reps, Random: {N_REPS} reps) ===")

    cache_dir = Path(__file__).parent.parent.parent / 'data' / 'cache_full'
    out_dir = Path(__file__).parent.parent.parent / 'results' / 'main_pipeline'

    data = load_dataset(data_dir=Path(__file__).parent.parent.parent / 'data' / 'ecom-offers')
    X_train, X_val = data['X_train'], data['X_val']
    feature_names = list(X_train.columns)

    X_train_explain = X_train.sample(min(N_EXPLAIN, len(X_train)), random_state=42).reset_index(drop=True)
    X_val_explain = X_val.sample(min(N_EXPLAIN, len(X_val)), random_state=42).reset_index(drop=True)

    print(f"\nExplaining {len(X_train_explain)} train + {len(X_val_explain)} val samples")
    print(f"Background: truth=full train ({len(X_train):,}), CTE/Random N={TARGET_SIZE}")

    bg_truth = pd.read_parquet(cache_dir / 'bg_truth_full.parquet')

    print(f"Pre-generating {N_REPS} CTE coresets and {N_REPS} Random backgrounds (N={TARGET_SIZE})...")
    bg_ctes = []
    bg_rands = []
    for rep in tqdm(range(N_REPS), desc="Generating backgrounds"):
        bg_ctes.append(build_cte_background(X_train, target_size=TARGET_SIZE, verbose=False, seed=rep))
        bg_rands.append(build_iid_background(X_train, size=TARGET_SIZE, seed=42 + rep))

    for pair in PAIRS:
        pair_out_dir = out_dir / pair['name']
        pair_out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n---> Starting RQ3 for pair: {pair['name']}")

        try:
            model = joblib.load(cache_dir / pair['model_file'])
        except Exception as e:
            print(f"Skipping {pair['name']} due to model load error: {e}")
            continue

        try:
            # --- GROUND TRUTH ---
            print("   Computing ground truth (full train bg)...")
            shap_truth_train = compute_shap(pair, model, bg_truth, X_train_explain)
            shap_truth_val = compute_shap(pair, model, bg_truth, X_val_explain)

            truth_imp_train = np.mean(np.abs(shap_truth_train), axis=0)
            truth_imp_val = np.mean(np.abs(shap_truth_val), axis=0)

            feature_drift = np.abs(truth_imp_val - truth_imp_train)
            drift_ranking = np.argsort(feature_drift)[::-1].tolist()

            # --- CTE (Ensemble) ---
            print("   Computing CTE Ensemble...")
            cte_train_maes, cte_val_maes = [], []
            cte_train_res, cte_val_res = [], []
            cte_imp_train_all, cte_imp_val_all = [], []

            for rep in range(N_REPS):
                shap_cte_train = compute_shap(pair, model, bg_ctes[rep], X_train_explain)
                shap_cte_val = compute_shap(pair, model, bg_ctes[rep], X_val_explain)

                cte_imp_train_all.append(np.mean(np.abs(shap_cte_train), axis=0))
                cte_imp_val_all.append(np.mean(np.abs(shap_cte_val), axis=0))

                cte_train_maes.append(float(mae_shap(shap_truth_train, shap_cte_train, global_importance=True)))
                cte_val_maes.append(float(mae_shap(shap_truth_val, shap_cte_val, global_importance=True)))
                cte_train_res.append(float(relative_error_shap(shap_truth_train, shap_cte_train, global_importance=True)))
                cte_val_res.append(float(relative_error_shap(shap_truth_val, shap_cte_val, global_importance=True)))

            cte_imp_train_arr = np.array(cte_imp_train_all)
            cte_imp_val_arr = np.array(cte_imp_val_all)

            # --- Random ---
            print("   Computing Random...")
            rand_train_maes, rand_val_maes = [], []
            rand_train_res, rand_val_res = [], []
            rand_imp_train_all, rand_imp_val_all = [], []

            for rep in range(N_REPS):
                shap_rand_train = compute_shap(pair, model, bg_rands[rep], X_train_explain)
                shap_rand_val = compute_shap(pair, model, bg_rands[rep], X_val_explain)

                rand_imp_train_all.append(np.mean(np.abs(shap_rand_train), axis=0))
                rand_imp_val_all.append(np.mean(np.abs(shap_rand_val), axis=0))

                rand_train_maes.append(float(mae_shap(shap_truth_train, shap_rand_train, global_importance=True)))
                rand_val_maes.append(float(mae_shap(shap_truth_val, shap_rand_val, global_importance=True)))
                rand_train_res.append(float(relative_error_shap(shap_truth_train, shap_rand_train, global_importance=True)))
                rand_val_res.append(float(relative_error_shap(shap_truth_val, shap_rand_val, global_importance=True)))

            rand_imp_train_arr = np.array(rand_imp_train_all)
            rand_imp_val_arr = np.array(rand_imp_val_all)

            result = {
                "bg_size": TARGET_SIZE,
                "n_explain": N_EXPLAIN,
                "feature_names": feature_names,
                "drift_ranking": drift_ranking,
                "feature_drift": feature_drift.tolist(),
                "truth": {
                    "train_importance": truth_imp_train.tolist(),
                    "val_importance": truth_imp_val.tolist(),
                },
                "cte": {
                    "train_importance_mean": np.mean(cte_imp_train_arr, axis=0).tolist(),
                    "train_importance_std": np.std(cte_imp_train_arr, axis=0).tolist(),
                    "val_importance_mean": np.mean(cte_imp_val_arr, axis=0).tolist(),
                    "val_importance_std": np.std(cte_imp_val_arr, axis=0).tolist(),
                    "train_mae_mean": float(np.mean(cte_train_maes)),
                    "train_mae_std": float(np.std(cte_train_maes)),
                    "val_mae_mean": float(np.mean(cte_val_maes)),
                    "val_mae_std": float(np.std(cte_val_maes)),
                    "train_re_mean": float(np.mean(cte_train_res)),
                    "train_re_std": float(np.std(cte_train_res)),
                    "val_re_mean": float(np.mean(cte_val_res)),
                    "val_re_std": float(np.std(cte_val_res)),
                    "n_reps": N_REPS,
                },
                "random": {
                    "train_importance_mean": np.mean(rand_imp_train_arr, axis=0).tolist(),
                    "train_importance_std": np.std(rand_imp_train_arr, axis=0).tolist(),
                    "val_importance_mean": np.mean(rand_imp_val_arr, axis=0).tolist(),
                    "val_importance_std": np.std(rand_imp_val_arr, axis=0).tolist(),
                    "train_mae_mean": float(np.mean(rand_train_maes)),
                    "train_mae_std": float(np.std(rand_train_maes)),
                    "val_mae_mean": float(np.mean(rand_val_maes)),
                    "val_mae_std": float(np.std(rand_val_maes)),
                    "train_re_mean": float(np.mean(rand_train_res)),
                    "train_re_std": float(np.std(rand_train_res)),
                    "val_re_mean": float(np.mean(rand_val_res)),
                    "val_re_std": float(np.std(rand_val_res)),
                    "n_reps": N_REPS,
                },
            }

            with open(pair_out_dir / 'rq3.json', 'w') as f:
                json.dump(result, f, indent=2)
            print(f"   RQ3 finished. Top 5 drifted features: "
                  f"{[feature_names[i] for i in drift_ranking[:5]]}")

        except Exception as e:
            import traceback
            print(f"Error computing RQ3 for {pair['name']}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    run_all_rq3()
