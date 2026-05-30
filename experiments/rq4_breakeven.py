"""
RQ4: Break-even analysis for CTE vs Random using KernelExplainer.

KernelExplainer is the "black-box" scenario CTE was designed for —
it works with any model and the background set directly determines
approximation quality. TreeExplainer (used in RQ1-3) is XGBoost-specific
and too fast for meaningful timing comparison.

Question: given that building a CTE coreset is expensive (one-time cost),
how many explanation batches do you need to run before CTE pays off
compared to always using random sampling?

break-even: n* = build_time_cte / (shap_time_rand - shap_time_cte)
"""

import time
import json
import pandas as pd
import shap
import joblib
from pathlib import Path
import warnings

import sys
sys.path.append(str(Path(__file__).parent.parent))

from data.data_loader import load_dataset
from cte_compression.kernel_thinning import build_iid_background


def run_rq4():
    print("Loading dataset and cached model...")
    try:
        data = load_dataset(data_dir=Path(__file__).parent.parent / 'data' / 'ecom-offers')
    except Exception as e:
        warnings.warn(f"Failed to load TabReD dataset, falling back to synthetic. Error: {e}")
        data = load_dataset()

    cache_dir = Path(__file__).parent.parent / 'data' / 'cache'
    if not (cache_dir / 'xgb_model.pkl').exists():
        raise FileNotFoundError("Cache not found! Run `python experiments/prepare_cache.py` first.")

    model = joblib.load(cache_dir / 'xgb_model.pkl')
    X_train = data['X_train']

    def pred_fn(X):
        return model.predict_proba(X)[:, 1]

    # Fixed explain set — 50 samples per batch (keeps timing reasonable)
    n_explain = 50
    X_explain = data['X_val'].sample(n_explain, random_state=42).reset_index(drop=True)

    bg_sizes = [10, 50, 100, 200]
    results = []

    for size in bg_sizes:
        print(f"\n--- Background size: {size} ---")

        # CTE — load coreset from cache, record true KT build time separately
        bg_cte_path = cache_dir / f'bg_cte_{size}.parquet'
        if not bg_cte_path.exists():
            raise FileNotFoundError(
                f"{bg_cte_path} not found. Run prepare_cache.py first."
            )
        bg_cte = pd.read_parquet(bg_cte_path)

        print("  Timing KernelExplainer with CTE background...")
        explainer_cte = shap.KernelExplainer(pred_fn, bg_cte)
        t0 = time.time()
        explainer_cte.shap_values(X_explain)
        t_cte_shap = time.time() - t0

        # Random — build fresh each time (no build cost)
        bg_rand = build_iid_background(X_train, size=size)

        print("  Timing KernelExplainer with Random background...")
        explainer_rand = shap.KernelExplainer(pred_fn, bg_rand)
        t0 = time.time()
        explainer_rand.shap_values(X_explain)
        t_rand_shap = time.time() - t0

        print(f"  CTE shap/batch: {t_cte_shap:.2f}s  |  Rand shap/batch: {t_rand_shap:.2f}s")

        results.append({
            "size":        size,
            "t_cte_shap":  t_cte_shap,
            "t_rand_shap": t_rand_shap,
        })

    # Load KT build times recorded during prepare_cache.py
    build_times_path = cache_dir / 'kt_build_times.json'
    if not build_times_path.exists():
        raise FileNotFoundError(
            "kt_build_times.json not found. Re-run prepare_cache.py to regenerate it."
        )
    with open(build_times_path) as f:
        build_times = json.load(f)

    print("\n=== Break-even summary ===")
    print(f"{'bg':>5} {'build(s)':>10} {'cte_shap':>10} {'rand_shap':>10} {'delta':>8} {'n* batches':>12} {'n* samples':>12}")
    for r in results:
        s = r['size']
        bt = build_times.get(str(s), 0)
        delta = r['t_rand_shap'] - r['t_cte_shap']
        if delta > 0:
            n_star = bt / delta
        else:
            n_star = float('inf')
        r['build_time']  = bt
        r['delta']       = delta
        r['n_star_batches'] = n_star if n_star != float('inf') else None
        r['n_star_samples'] = n_star * n_explain if n_star != float('inf') else None
        print(f"{s:>5} {bt:>10.1f} {r['t_cte_shap']:>10.2f} {r['t_rand_shap']:>10.2f} "
              f"{delta:>8.2f} {n_star:>12.0f} {n_star*n_explain:>12.0f}")

    out_dir = Path(__file__).parent.parent / 'results'
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / 'rq4_breakeven.json', 'w') as f:
        json.dump({
            "n_explain_per_batch": n_explain,
            "experiments": results,
        }, f, indent=2)
    print("\nResults saved to results/rq4_breakeven.json")


if __name__ == "__main__":
    run_rq4()
