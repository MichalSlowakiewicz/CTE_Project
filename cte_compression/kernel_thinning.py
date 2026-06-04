"""
compression/kernel_thinning.py - CTE background set construction.

Implements the "Compress" step from:
    Baniecki et al. "Efficient and Accurate Explanation Estimation
    with Distribution Compression", ICLR 2025.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler
from goodpoints import compress
import goodpoints.kt as kt

def rbf_kernel_wrapper(X, Y):
    X2d = np.atleast_2d(X)
    Y2d = np.atleast_2d(Y)

    if (X2d.ndim == 2
            and Y2d.ndim == 2
            and X2d.shape == Y2d.shape
            and X2d.shape[0] > 1
            and np.shares_memory(X, Y)):
        return np.ones(X2d.shape[0])

    K = rbf_kernel(X2d, Y2d)

    if K.shape[0] == 1 or K.shape[1] == 1:
        return K.flatten()

    return K


def build_cte_background(X_data, target_size=50, max_halving_rounds=10, verbose=True):
    """
    Build a CTE background set using Compress++ (KT) for speed.
    """
    X_clean = X_data.copy().reset_index(drop=True)
    
    # 1. Dynamic Compress phase (extract base_size * 2^g points)
    n_prime = 4 ** int(np.floor(np.log(len(X_clean)) / np.log(4)))
    base_size = int(np.sqrt(n_prime))
    
    g = 0
    while base_size * (2**g) < target_size:
        g += 1
        
    output_size = base_size * (2**g)
    
    if verbose:
        print(f"  [COMPRESS] Extracting {output_size} points (g={g})...")
        
    X_np = np.ascontiguousarray(X_clean.values.astype(np.float64))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_np)
    
    # Use proper Gaussian bandwidth (k_params = n_features), otherwise it acts as an Identity Matrix!
    gamma_param = np.array([X_scaled.shape[1]], dtype=np.float64)
    
    # Use compress_kt directly to respect our dynamic g and avoid forced sqrt(N) truncation
    ids = compress.compress_kt(np.ascontiguousarray(X_scaled), kernel_type=b"gaussian", k_params=gamma_param, g=g)
    X_current = X_clean.iloc[ids].reset_index(drop=True)

    # 2. Exact KT phase (standard O(N^2), but N is small now)
    m = int(np.floor(np.log2(len(X_current) / target_size)))
    m = min(m, max_halving_rounds)

    if m > 0:
        if verbose:
            print(f"  [KT] Reducing {len(X_current)} points via {m} halving rounds...")
            
        X_np_curr = np.ascontiguousarray(X_current.values.astype(np.float64))
        X_scaled_curr = scaler.fit_transform(X_np_curr)
        
        indices = kt.thin(
            X=X_scaled_curr,
            m=m,
            split_kernel=rbf_kernel_wrapper,
            swap_kernel=rbf_kernel_wrapper,
            seed=42,
        )
        X_current = X_current.iloc[indices].reset_index(drop=True)

    # 3. No final trimming. We rely on the natural halving sizes (powers of 2)
    # to maintain strict minimax optimality of the Kernel Thinning coreset.
    if verbose:
        print(f"  [CTE] Final coreset size: {len(X_current)}")

    return X_current


def build_iid_background(X_data, size=1000, seed=42):
    n = min(size, len(X_data))
    return X_data.sample(n, random_state=seed).reset_index(drop=True)
