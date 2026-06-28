"""
compression/kernel_thinning.py - CTE background set construction.

Implements the "Compress" step from:
    Baniecki et al. "Efficient and Accurate Explanation Estimation
    with Distribution Compression", ICLR 2025.

Uses the Compress++ algorithm from:
    Shetty, Dwivedi, Mackey. "Distribution Compression in Near-linear Time",
    ICLR 2022.  (arXiv:2111.07941v6)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from goodpoints import compress


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


def build_cte_background(X_data, target_size=50, max_halving_rounds=10, verbose=True, seed=42):
    """
    Build a CTE background set using the native Compress++ (compresspp_kt)
    from the goodpoints library.

    Architecture (based on Shetty et al., ICLR 2022):
    ──────────────────────────────────────────────────
    1. compresspp_kt produces a coreset of size √n' (where n' is the largest
       power of 4 ≤ n). This is the NATIVE output of Algorithm 2 (COMPRESS++).
       It internally runs both the COMPRESS phase (fast, near-linear) and the
       THIN phase (accurate, quadratic on the small intermediate set) using
       optimized C extensions and consistent kernel computations.

    2. If a smaller target_size < √n' is requested, we apply additional
       KT halving rounds to the √n'-point coreset. Each halving is an
       additional round of full-matrix Kernel Thinning (including KT-SWAP)
       that further refines the set. This additional refinement operates
       on O(n²) kernel matrices and can significantly improve quality
       beyond the near-linear Compress++ output.

    Important note on the N=√n' edge case:
       When target_size = √n' (e.g., 256 for M≈100k), NO additional KT
       halving is applied. The returned coreset is the raw Compress++ output.
       When target_size < √n' (e.g., 128), the coreset receives additional
       KT-SWAP refinement steps that directly optimize MMD on the target
       kernel. This means smaller targets can paradoxically have BETTER
       distributional fidelity than √n' itself, because they benefit from
       the extra quadratic-time refinement.

    Args:
        X_data: Input DataFrame of training samples.
        target_size: Desired number of background points (power of 2).
        max_halving_rounds: Safety cap on the number of halving rounds.
        verbose: Print progress information.
        seed: Random seed for reproducibility. The algorithm is randomized
              (KT-SPLIT uses coin flips), but fixing the seed yields
              identical coresets across runs. Different seeds produce
              independent coresets, enabling ensemble averaging.

    Returns:
        DataFrame with the selected background points.
    """
    X_clean = X_data.copy().reset_index(drop=True)

    # ─── Step 0: Standardize and compute bandwidth ───
    X_np = np.ascontiguousarray(X_clean.values.astype(np.float64))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_np)

    # Bandwidth: median heuristic (standard in MMD / kernel thinning literature)
    # k_params[j] = λ² for goodpoints Gaussian kernel: k(x,y) = exp(-||x-y||²/λ²)
    bandwidth_sq = _median_bandwidth(X_scaled)
    k_params = np.array([bandwidth_sq], dtype=np.float64)

    if verbose:
        print(f"  [BANDWIDTH] Median heuristic: λ² = {bandwidth_sq:.4f}")

    # ─── Step 1: Run native COMPRESS++ ───
    # compresspp_kt (Algorithm 2 from the paper) runs:
    #   Phase 1 (COMPRESS): near-linear time, outputs 2^g * √n' points
    #   Phase 2 (THIN): accurate KT on the intermediate set, outputs √n' points
    #
    # The oversampling parameter g controls how much the COMPRESS phase
    # oversamples before THIN refines. g = ceil(log4(log4(n))) is recommended
    # by Remark 5 for near-linear runtime with √2 error inflation.
    n_prime = 4 ** int(np.floor(np.log(len(X_clean)) / np.log(4)))
    native_size = int(np.sqrt(n_prime))  # = √n', the natural output size

    # Use g recommended by the paper: ceil(log4(log4(n)))
    # For n~100k: log4(100000) ≈ 8.3, log4(8.3) ≈ 1.5, ceil = 2
    g = max(0, int(np.ceil(np.log(max(1, np.log(n_prime) / np.log(4))) / np.log(4))))

    if verbose:
        print(f"  [COMPRESS++] n'={n_prime}, native output √n'={native_size}, g={g}")

    ids = compress.compresspp_kt(
        np.ascontiguousarray(X_scaled),
        kernel_type=b"gaussian",
        k_params=k_params,
        g=g,
        seed=seed,
    )

    if verbose:
        print(f"  [COMPRESS++] Produced {len(ids)} points (native COMPRESS++ output)")

    X_current = X_clean.iloc[ids].reset_index(drop=True)

    # ─── Step 2: Additional KT halving if target_size < native_size ───
    if target_size < len(X_current):
        m = int(np.floor(np.log2(len(X_current) / target_size)))
        m = min(m, max_halving_rounds)

        if m > 0:
            if verbose:
                print(f"  [KT-THIN] Reducing {len(X_current)} → {len(X_current) // (2**m)} "
                      f"via {m} additional halving rounds...")

            # Scale the coreset points using the SAME scaler (consistent RKHS space)
            X_np_curr = np.ascontiguousarray(X_current.values.astype(np.float64))
            X_scaled_curr = scaler.transform(X_np_curr)

            # Use goodpoints' own compresspp_kt for the additional thinning too.
            # This keeps all kernel computations inside the library's C extensions,
            # avoiding any numerical mismatch from external kernel implementations.
            #
            # However, compresspp_kt always outputs √n points, so for fine-grained
            # control we use the native kt.thin_K with the library's own kernel matrix.
            import goodpoints.compressc as compressc
            import goodpoints.kt as kt_module

            n_curr = X_scaled_curr.shape[0]
            K = np.empty((n_curr, n_curr))
            compressc.compute_K(
                X_scaled_curr,
                np.arange(n_curr, dtype=int),
                b"gaussian",
                k_params,
                K,
            )

            # kt.thin_K(K_split, K_swap, m) uses the precomputed kernel matrix
            # for both KT-SPLIT and KT-SWAP, exactly as compresspp_kt does internally.
            indices = kt_module.thin_K(K, K, m, delta=0.5, seed=seed)
            X_current = X_current.iloc[indices].reset_index(drop=True)

    actual_size = len(X_current)
    if verbose:
        print(f"  [CTE] Final coreset size: {actual_size}"
              + (f" (requested {target_size})" if actual_size != target_size else ""))

    return X_current


def build_iid_background(X_data, size=1000, seed=42):
    n = min(size, len(X_data))
    return X_data.sample(n, random_state=seed).reset_index(drop=True)
