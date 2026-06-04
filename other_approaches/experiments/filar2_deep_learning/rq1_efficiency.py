import time
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from data.data_loader import load_dataset
from cte_compression.kernel_thinning import build_iid_background
from evaluation.metrics import mae_shap, speedup_ratio

# Import the model architecture
from prepare_mlp import RobustMLP, compute_expected_gradients

def run_filar2_rq1():
    print("=== Filar 2 (Deep Learning) - RQ1: Efficiency vs Fidelity ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = Path(__file__).parent.parent.parent / 'data' / 'cache_full'
    out_dir = Path(__file__).parent.parent.parent / 'results' / 'filar2_deep_learning'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    data = load_dataset(data_dir=Path(__file__).parent.parent.parent / 'data' / 'ecom-offers')
    X_train, X_val = data['X_train'], data['X_val']
    
    model = RobustMLP(X_train.shape[1]).to(device)
    model.load_state_dict(torch.load(cache_dir / 'mlp_model.pt', map_location=device))
    model.eval()
    
    X_val_explain = X_val.sample(min(100, len(X_val)), random_state=42).reset_index(drop=True)
    
    print("Loading Ground Truth Background...")
    bg_truth = pd.read_parquet(cache_dir / 'bg_truth_full.parquet')
    bg_truth_sampled = bg_truth.sample(4000, random_state=42) # Downsample GT to 4000 to match 20x methodology
    
    t0 = time.time()
    shap_truth = compute_expected_gradients(model, X_val_explain, bg_truth_sampled, device)
    time_truth = time.time() - t0
    print(f"Ground Truth computed in {time_truth:.2f} seconds.")
    
    bg_sizes = [10, 50, 100, 200]
    results = []
    
    for size in tqdm(bg_sizes, desc="RQ1 (Filar 2) sizes"):
        bg_cte = pd.read_parquet(cache_dir / f'bg_cte_{size}.parquet')
        
        t0 = time.time()
        shap_cte = compute_expected_gradients(model, X_val_explain, bg_cte, device)
        t_cte = time.time() - t0
        
        bg_rand = build_iid_background(X_train, size=size)
        
        t0 = time.time()
        shap_rand = compute_expected_gradients(model, X_val_explain, bg_rand, device)
        t_rand = time.time() - t0
        
        res = {
            "size": size,
            "cte": {"mae": float(mae_shap(shap_truth, shap_cte)), "speedup": float(speedup_ratio(time_truth, t_cte))},
            "random": {"mae": float(mae_shap(shap_truth, shap_rand)), "speedup": float(speedup_ratio(time_truth, t_rand))}
        }
        results.append(res)
        
    with open(out_dir / 'rq1.json', 'w') as f:
        json.dump({"truth_time": time_truth, "experiments": results}, f, indent=2)
    print("\nResults saved to results/filar2_deep_learning/rq1.json")

if __name__ == "__main__":
    run_filar2_rq1()
