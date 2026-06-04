import time
import json
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from data.data_loader import load_dataset
from cte_compression.kernel_thinning import build_cte_background
from evaluation.metrics import mae_shap

from prepare_mlp import RobustMLP, compute_expected_gradients

def run_filar2_rq2():
    print("=== Filar 2 (Deep Learning) - RQ2: Concept Drift ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = Path(__file__).parent.parent.parent / 'data' / 'cache_full'
    out_dir = Path(__file__).parent.parent.parent / 'results' / 'filar2_deep_learning'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    data = load_dataset(data_dir=Path(__file__).parent.parent.parent / 'data' / 'ecom-offers')
    X_train, X_test = data['X_train'], data['X_test']
    
    model = RobustMLP(X_train.shape[1]).to(device)
    model.load_state_dict(torch.load(cache_dir / 'mlp_model.pt', map_location=device))
    model.eval()
    
    X_test_explain = X_test.sample(min(100, len(X_test)), random_state=42).reset_index(drop=True)
    
    bg_truth_test = X_test.sample(min(4000, len(X_test)), random_state=42)
    print(f"Computing Ground Truth ExpectedGradients on Test background (size: {len(bg_truth_test)})...")
    
    shap_truth = compute_expected_gradients(model, X_test_explain, bg_truth_test, device)
    
    bg_sizes = [50, 100, 200]
    results = []
    
    for size in tqdm(bg_sizes, desc="RQ2 (Filar 2) sizes"):
        bg_cte_past = pd.read_parquet(cache_dir / f'bg_cte_{size}.parquet')
        shap_past = compute_expected_gradients(model, X_test_explain, bg_cte_past, device)
        
        bg_cte_future = build_cte_background(X_test, target_size=size, verbose=False)
        shap_future = compute_expected_gradients(model, X_test_explain, bg_cte_future, device)
        
        res = {
            "size": size,
            "mae_past_bg": float(mae_shap(shap_truth, shap_past)),
            "mae_future_bg": float(mae_shap(shap_truth, shap_future))
        }
        results.append(res)
        
    with open(out_dir / 'rq2.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to results/filar2_deep_learning/rq2.json")

if __name__ == "__main__":
    run_filar2_rq2()
