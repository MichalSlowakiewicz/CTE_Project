"""
data/data_loader.py - Dataset loading and preparation.

Supports two modes:
  1. TabReD ecom_offers (real data) - set TABRED_DATA_DIR in config.py
  2. Synthetic fallback  - used when TABRED_DATA_DIR is None

TabReD ecom_offers is a clickthrough dataset from an e-commerce recommendation
system with a temporal train/val split (older months → train, newer → val).
The temporal structure makes concept drift naturally present in the data.

TabReD data is NOT a pip package. Download instructions:
  git clone https://github.com/yandex-research/tabred
  cd tabred
  # ecom_offers is a private Kaggle dataset; download from:
  # https://www.kaggle.com/datasets/pcovkrd84mejm/cooking-time  (similar structure)
  # Follow preprocessing/ecom_offers.py to produce the split files.
"""

import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def load_dataset(data_dir=None, seed=42):
    """
    Load the dataset and return train/val splits with period metadata.

    Args:
        data_dir: Path to preprocessed TabReD ecom_offers directory.
                  If None, generates a synthetic dataset instead.
        seed:     Random seed for the synthetic generator.

    Returns:
        dict with keys:
            X_train (pd.DataFrame), y_train (np.ndarray),
            X_val   (pd.DataFrame), y_val   (np.ndarray),
            X_test  (pd.DataFrame or None), y_test  (np.ndarray or None),
            feature_names (list[str]),
            periods_train (np.ndarray or None),
            periods_val   (np.ndarray or None),
            periods_test  (np.ndarray or None),
            source (str)  -- "tabred" or "synthetic"
    """
    if data_dir is not None:
        return _load_tabred(Path(data_dir))
    return _load_synthetic(seed=seed)


# ---------------------------------------------------------------------------
# TabReD loader
# ---------------------------------------------------------------------------

def _load_tabred(data_dir: Path):
    """
    Load preprocessed TabReD ecom_offers files.

    Expected directory layout (produced by TabReD preprocessing scripts):
        X_train.parquet / X_train.csv
        X_val.parquet   / X_val.csv
        y_train.parquet / y_train.csv / y_train.npy
        y_val.parquet   / y_val.csv   / y_val.npy
        (optionally) X_test, y_test, periods_train, periods_val, periods_test
    """
    if not data_dir.exists():
        raise FileNotFoundError(
            f"TabReD data directory not found: {data_dir}\n"
            "Download instructions: see data/data_loader.py docstring."
        )

    X_train = _read_frame(data_dir, "X_train")
    X_val   = _read_frame(data_dir, "X_val")
    y_train = _read_target(data_dir, "y_train")
    y_val   = _read_target(data_dir, "y_val")

    try:
        X_test = _read_frame(data_dir, "X_test")
        y_test = _read_target(data_dir, "y_test")
        periods_test = _try_read_npy(data_dir / "periods_test.npy")
    except FileNotFoundError:
        X_test, y_test, periods_test = None, None, None

    # Period metadata is optional - TabReD stores it separately
    periods_train = _try_read_npy(data_dir / "periods_train.npy")
    periods_val   = _try_read_npy(data_dir / "periods_val.npy")

    print(f"[data] Loaded TabReD ecom_offers: "
          f"train={len(X_train):,}  val={len(X_val):,}  "
          f"features={X_train.shape[1]}")
    if X_test is not None:
        print(f"       test={len(X_test):,}")

    return {
        "X_train":       X_train,
        "y_train":       y_train,
        "X_val":         X_val,
        "y_val":         y_val,
        "X_test":        X_test,
        "y_test":        y_test,
        "feature_names": X_train.columns.tolist(),
        "periods_train": periods_train,
        "periods_val":   periods_val,
        "periods_test":  periods_test,
        "source":        "tabred",
    }


def _read_frame(directory: Path, name: str) -> pd.DataFrame:
    for ext in (".parquet", ".csv", ".feather"):
        path = directory / (name + ext)
        if path.exists():
            if ext == ".parquet":
                return pd.read_parquet(path)
            if ext == ".csv":
                return pd.read_csv(path)
            if ext == ".feather":
                return pd.read_feather(path)
    raise FileNotFoundError(f"Could not find {name}.[parquet|csv|feather] in {directory}")


def _read_target(directory: Path, name: str) -> np.ndarray:
    npy_path = directory / (name + ".npy")
    if npy_path.exists():
        return np.load(npy_path)
    # Try reading as single-column frame
    frame = _read_frame(directory, name)
    return frame.values.ravel()


def _try_read_npy(path: Path):
    if path.exists():
        return np.load(path)
    return None


# ---------------------------------------------------------------------------
# Synthetic fallback (mirrors the notebook prototype)
# ---------------------------------------------------------------------------

def _load_synthetic(n_samples=20_000, seed=42):
    """
    Generate a synthetic e-commerce clickthrough dataset with injected
    concept drift:
      - Periods 1-6:  discount_offered drives clicks
      - Periods 7-12: past_purchases  drives clicks

    Temporal train/val split: periods 1-8 → train, 9-12 → val.
    This mimics the TabReD time-based split philosophy.
    """
    rng = np.random.default_rng(seed)

    user_age        = rng.normal(35, 10, n_samples)
    past_purchases  = rng.poisson(3, n_samples)
    time_on_site    = rng.exponential(5, n_samples)
    discount        = rng.uniform(5, 50, n_samples)
    periods         = np.repeat(np.arange(1, 13), int(np.ceil(n_samples / 12)))[:n_samples]

    y = np.zeros(n_samples)
    for i in range(n_samples):
        if periods[i] <= 6:
            logit = discount[i] * 0.2 - 5
        else:
            logit = past_purchases[i] * 0.8 - 2
        prob = 1.0 / (1.0 + np.exp(-logit))
        y[i] = rng.binomial(1, prob)

    df = pd.DataFrame({
        "user_age":        user_age,
        "past_purchases":  past_purchases,
        "time_on_site":    time_on_site,
        "discount_offered": discount,
        "period":          periods,
    })

    train_mask = df["period"] <= 8
    X_train = df[train_mask].drop(columns=["period"]).reset_index(drop=True)
    X_val   = df[~train_mask].drop(columns=["period"]).reset_index(drop=True)
    y_train = y[train_mask]
    y_val   = y[~train_mask]

    print(f"[data] Loaded synthetic dataset: "
          f"train={len(X_train):,}  val={len(X_val):,}  "
          f"features={X_train.shape[1]}")

    return {
        "X_train":       X_train,
        "y_train":       y_train,
        "X_val":         X_val,
        "y_val":         y_val,
        "X_test":        None,
        "y_test":        None,
        "feature_names": X_train.columns.tolist(),
        "periods_train": df[train_mask]["period"].values,
        "periods_val":   df[~train_mask]["period"].values,
        "periods_test":  None,
        "source":        "synthetic",
    }
