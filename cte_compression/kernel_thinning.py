"""
compression/kernel_thinning.py - CTE background set construction.

Implements the "Compress" step from:
    Baniecki et al. "Efficient and Accurate Explanation Estimation
    with Distribution Compression", ICLR 2025.

The key idea: instead of randomly sampling a background set for SHAP,
use Kernel Thinning (goodpoints library) to find a small coreset that
minimizes the Maximum Mean Discrepancy (MMD) to the full training distribution.
This gives better explanations per background point than i.i.d. sampling.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel
import goodpoints.kt as kt


def rbf_kernel_wrapper(X, Y):
    """
    RBF kernel compatible with the goodpoints calling convention.

    goodpoints calls the kernel function in three patterns:
      (a) kernel(X[i, newaxis], X)          -> expects 1D array of length n
      (b) kernel(X[i, newaxis], X[coreset]) -> expects 1D array of length |C|
      (c) kernel(X, X)                      -> expects 1D diagonal of length n

    Pattern (c) is the problematic one: it happens inside refine_X (kt.py:678)
    where the library needs the diagonal k(xi, xi) for each point.
    Without the fix, rbf_kernel returns an (n, n) matrix here, argmin gives
    a flat index > n, and everything crashes with an IndexError.

    Fix: detect pattern (c) via np.shares_memory and return np.ones(n),
    because for RBF: k(x, x) = exp(-gamma * 0) = 1.0 always.
    """
    X2d = np.atleast_2d(X)
    Y2d = np.atleast_2d(Y)

    # Pattern (c): same object on both sides -> return diagonal
    if (X2d.ndim == 2
            and Y2d.ndim == 2
            and X2d.shape == Y2d.shape
            and X2d.shape[0] > 1
            and np.shares_memory(X, Y)):
        return np.ones(X2d.shape[0])

    K = rbf_kernel(X2d, Y2d)

    # Patterns (a) and (b): one side is a single point -> flatten to 1D
    if K.shape[0] == 1 or K.shape[1] == 1:
        return K.flatten()

    return K


def build_cte_background(X_data, target_size=50, max_halving_rounds=6,
                          verbose=True):
    """
    Build a CTE background set using Kernel Thinning.

    Compresses X_data into a coreset of ~target_size representative points
    by running m rounds of kernel halving, where m = floor(log2(n / target_size)).

    Args:
        X_data:             pd.DataFrame with training features.
        target_size:        Desired number of background points.
        max_halving_rounds: Cap on m to prevent very long runtimes.
        verbose:            Print progress info.

    Returns:
        pd.DataFrame with target_size (or fewer) rows selected from X_data.
    """
    # Reset index so iloc lookups match the numpy array indices
    X_clean = X_data.copy().reset_index(drop=True)

    m = int(np.floor(np.log2(len(X_clean) / target_size)))
    m = min(m, max_halving_rounds)

    if m <= 0:
        if verbose:
            print(f"  [KT] n={len(X_clean)} too small for {target_size} target, "
                  "using random sample")
        return X_clean.sample(min(target_size, len(X_clean)), random_state=42)

    # goodpoints requires a C-contiguous float64 array
    X_np = np.ascontiguousarray(X_clean.values.astype(np.float64))
    n_out = len(X_np) // (2 ** m)

    if verbose:
        print(f"  [KT] n={len(X_clean):,}  m={m} rounds  -> ~{n_out} points")

    indices = kt.thin(
        X=X_np,
        m=m,
        split_kernel=rbf_kernel_wrapper,
        swap_kernel=rbf_kernel_wrapper,
        seed=42,
    )

    coreset = X_clean.iloc[indices]

    if len(coreset) > target_size:
        coreset = coreset.sample(target_size, random_state=42)

    if verbose:
        print(f"  [KT] coreset size: {len(coreset)}")

    return coreset


def build_iid_background(X_data, size=1000, seed=42):
    """
    Build a standard i.i.d. random background set (baseline for comparison).

    Args:
        X_data: pd.DataFrame with training features.
        size:   Number of background points to sample.
        seed:   Random seed.

    Returns:
        pd.DataFrame with 'size' randomly sampled rows.
    """
    n = min(size, len(X_data))
    return X_data.sample(n, random_state=seed).reset_index(drop=True)
