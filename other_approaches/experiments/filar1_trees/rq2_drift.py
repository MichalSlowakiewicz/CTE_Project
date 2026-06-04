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
from cte_compression.kernel_thinning import build_cte_background
from evaluation.metrics import mae_shap

def run_filar1_rq2():
    print("=== Filar 1 (TreeExplainer) - RQ2: Concept Drift ===")
    
    cache_dir = Path(__file__).parent.parent.parent / 'data' / 'cache_full'
    out_dir = Path(__file__).parent.parent.parent / 'results' / 'filar1_trees'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    model = joblib.load(cache_dir / 'xgb_model.pkl')
    data = load_dataset(data_dir=Path(__file__).parent.parent.parent / 'data' / 'ecom-offers')
    
    X_train, X_test = data['X_train'], data['X_test']
    
    # Explain recent (test) samples
    X_test_explain = X_test.sample(min(200, len(X_test)), random_state=42).reset_index(drop=True)
    
    # True background for Test samples is the Test set itself
    bg_truth_test = X_test
    print(f"Computing Ground Truth SHAP on Test background (size: {len(bg_truth_test)})...")
    explainer_truth = shap.TreeExplainer(model, data=bg_truth_test, model_output="probability", feature_perturbation="interventional")
    shap_truth = explainer_truth.shap_values(X_test_explain)
    
    bg_sizes = [50, 100, 200]
    results = []
    
    for size in tqdm(bg_sizes, desc="RQ2 (Drift) sizes"):
        # CTE built on Past (Train)
        bg_cte_past = pd.read_parquet(cache_dir / f'bg_cte_{size}.parquet')
        explainer_past = shap.TreeExplainer(model, data=bg_cte_past, model_output="probability", feature_perturbation="interventional")
        shap_past = explainer_past.shap_values(X_test_explain)
        
        # CTE built on Future (Test)
        # We compute it on the fly for RQ2
        bg_cte_future = build_cte_background(X_test, target_size=size, verbose=False)
        explainer_future = shap.TreeExplainer(model, data=bg_cte_future, model_output="probability", feature_perturbation="interventional")
        shap_future = explainer_future.shap_values(X_test_explain)
        
        res = {
            "size": size,
            "mae_past_bg": float(mae_shap(shap_truth, shap_past)),
            "mae_future_bg": float(mae_shap(shap_truth, shap_future))
        }
        results.append(res)
        
    with open(out_dir / 'rq2.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to results/filar1_trees/rq2.json")

if __name__ == "__main__":
    run_filar1_rq2()
