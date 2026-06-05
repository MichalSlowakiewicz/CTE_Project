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


def _median_bandwidth(X_scaled, n_subsample=5000, seed=42):
    """
    Compute the median heuristic bandwidth for a Gaussian kernel.

    Returns λ² such that the kernel k(x,y) = exp(-||x-y||² / λ²)
    has λ² equal to the median of pairwise squared distances.
    This is the standard bandwidth selection in the MMD literature.

    We subsample to avoid O(N²) cost on large datasets.
    """
    rng = np.random.default_rng(seed)
    n = X_scaled.shape[0]
    if n > n_subsample:
        idx = rng.choice(n, n_subsample, replace=False)
        X_sub = X_scaled[idx]
    else:
        X_sub = X_scaled

    # Compute pairwise squared distances on the subsample
    # Using the identity ||x-y||² = ||x||² + ||y||² - 2<x,y>
    norms_sq = np.sum(X_sub ** 2, axis=1)
    dists_sq = norms_sq[:, None] + norms_sq[None, :] - 2.0 * X_sub @ X_sub.T

    # Extract upper triangle (excluding diagonal zeros)
    upper_tri = dists_sq[np.triu_indices_from(dists_sq, k=1)]
    median_sq_dist = float(np.median(upper_tri))

    # Guard against degenerate case (all points identical)
    if median_sq_dist < 1e-10:
        median_sq_dist = 1.0

    return median_sq_dist


def rbf_kernel_wrapper(X, Y):
    X2d = np.atleast_2d(X)
    Y2d = np.atleast_2d(Y)

    # Safe identity check: only return ones when X and Y are the
    # exact same object (diagonal self-kernel evaluation).
    # Previous code used np.shares_memory which could give false
    # positives for different views of the same underlying array.
    if (X2d.ndim == 2
            and Y2d.ndim == 2
            and X2d.shape == Y2d.shape
            and X2d.shape[0] > 1
            and X is Y):
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

    # Bandwidth selection: median heuristic (standard in MMD literature)
    # k_params[j] = λ² for Gaussian kernel k(x,y) = exp(-||x-y||² / λ²)
    bandwidth_sq = _median_bandwidth(X_scaled)
    gamma_param = np.array([bandwidth_sq], dtype=np.float64)

    if verbose:
        print(f"  [BANDWIDTH] Median heuristic: λ² = {bandwidth_sq:.4f}")

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
        # FIX: Use transform() instead of fit_transform() to maintain
        # consistent scaling between Compress++ and KT phases.
        # Previously fit_transform() recomputed mean/std on the compressed
        # subset, breaking the kernel space consistency.
        X_scaled_curr = scaler.transform(X_np_curr)

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
    actual_size = len(X_current)
    if verbose:
        print(f"  [CTE] Final coreset size: {actual_size}"
              + (f" (requested {target_size})" if actual_size != target_size else ""))

    return X_current


def build_iid_background(X_data, size=1000, seed=42):
    n = min(size, len(X_data))
    return X_data.sample(n, random_state=seed).reset_index(drop=True)
