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
    
    print("\n=== Global Cache Preparation Complete! ===")

if __name__ == "__main__":
    prepare_full_cache()
