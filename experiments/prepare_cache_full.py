import json
import time
import joblib
import pandas as pd
from pathlib import Path
import warnings
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from data.data_loader import load_dataset
from cte_compression.kernel_thinning import build_cte_background
import xgboost as xgb
from sklearn.metrics import roc_auc_score

def prepare_full_cache():
    print("=== CTE Global Cache Preparation (FULL DATASET) ===")
    
    cache_dir = Path(__file__).parent.parent / 'data' / 'cache_full'
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n1. Loading FULL dataset (this may take a moment)...")
    try:
        data = load_dataset(data_dir=Path(__file__).parent.parent / 'data' / 'ecom-offers')
    except Exception as e:
        warnings.warn(f"Failed to load TabReD dataset, falling back to synthetic. Error: {e}")
        data = load_dataset()
        
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    
    print("\n[NOTE] Model training has been moved to train_xgboost.py")
    
    print(f"\n1. Saving Ground Truth Background (Size: {len(X_train)})...")
    bg_truth_path = cache_dir / 'bg_truth_full.parquet'
    X_train.to_parquet(bg_truth_path)
    print(f"   Ground Truth saved to {bg_truth_path}")
    
    bg_sizes = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    print(f"\n4. Building CTE Backgrounds for sizes: {bg_sizes}")
    
    # Progress bar for background sizes
    for size in tqdm(bg_sizes, desc="Generating Coresets (CTE)"):
        bg_cte_path = cache_dir / f'bg_cte_{size}.parquet'
        
        # We also want to see when it's done for each size
        tqdm.write(f"\n--- Starting Kernel Thinning for Target Size: {size} ---")
        t0 = time.time()
        # verbose=False to keep tqdm clean, but goodpoints might print anyway
        bg_cte = build_cte_background(X_train, target_size=size, verbose=False)
        t1 = time.time()
        
        bg_cte.to_parquet(bg_cte_path)
        tqdm.write(f"   Completed in {t1-t0:.1f} seconds. Saved to {bg_cte_path.name}")
    
    print("\n=== Global Cache Preparation Complete! ===")

if __name__ == "__main__":
    prepare_full_cache()
