"""
cte_kernel.py - Moduł CTE (Compress Then Explain) dla projektu Biecek/Baniecki.

Rozwiązuje problem IndexError w bibliotece goodpoints wynikający z tego,
że goodpoints/kt.py linia 678 wywołuje:
    sufficient_stat = kernel(X, X) / coreset_size

gdzie oba argumenty to ten sam obiekt X o kształcie (n, d).
Biblioteka oczekuje wektora 1D długości n (diagonal), ale rbf_kernel()
zwraca macierz (n, n) - stąd argmin() zwraca flat index > n.

Użycie w notebooku:
    from cte_kernel import smart_rbf_kernel, get_cte_background
"""

import numpy as np
import goodpoints.kt as kt
from sklearn.metrics.pairwise import rbf_kernel


def smart_rbf_kernel(X, Y):
    """
    Wrapper dla jądra RBF kompatybilny z goodpoints.

    goodpoints wywołuje kernel w trzech trybach:
      1) kernel(X[i, newaxis], X)         -> wektor (n,)      [split_X, swap_X]
      2) kernel(X[i, newaxis], X[coreset])-> wektor (|C|,)    [refine_X]
      3) kernel(X, X)                     -> DIAGONAL (n,)    [refine_X linia 678]
         ↑ bez fixu rbf_kernel zwraca (n,n) -> argmin daje flat index -> IndexError!

    Fix: wykrywamy tryb 3 przez np.shares_memory i zwracamy diagonal (= np.ones
    dla RBF z domyślnym gamma, bo exp(-gamma * ||x-x||^2) = exp(0) = 1).
    """
    X2d = np.atleast_2d(X)
    Y2d = np.atleast_2d(Y)

    # === KRYTYCZNY FIX: tryb 3 - kernel(X, X) ===
    # goodpoints wywołuje kernel z tym samym obiektem X po obu stronach.
    # np.shares_memory(X, X) = True pozwala nam to wykryć.
    # Dla RBF: k(x, x) = exp(-gamma * 0) = 1.0, więc diagonal = np.ones(n)
    if (X2d.ndim == 2
            and Y2d.ndim == 2
            and X2d.shape == Y2d.shape
            and X2d.shape[0] > 1
            and np.shares_memory(X, Y)):
        return np.ones(X2d.shape[0])

    # Tryby 1 i 2: standardowe wywołanie punktu vs. zbioru punktów
    K = rbf_kernel(X2d, Y2d)

    # Gdy jeden z wymiarów = 1, zwróć wektor 1D (nie macierz 1×n)
    if K.shape[0] == 1 or K.shape[1] == 1:
        return K.flatten()

    return K


def get_cte_background(X_data, target_size=50, verbose=True):
    """
    Zwraca tło (background) dla SHAP obliczone metodą Kernel Thinning (CTE).

    Implementuje algorytm z:
        Baniecki et al. "Efficient and Accurate Explanation Estimation
        with Distribution Compression" (ICLR 2025)

    Args:
        X_data   : pd.DataFrame z danymi treningowymi
        target_size: docelowy rozmiar coresetu (tła SHAP)
        verbose  : drukuj info o postępie

    Returns:
        pd.DataFrame: coreset o rozmiarze <= target_size
    """
    # Oblicz ile rund halvingów (każda dzieli zbiór przez 2)
    m_steps = int(np.floor(np.log2(len(X_data) / target_size)))

    if m_steps <= 0:
        if verbose:
            print(f"  ℹ️  Za mało danych dla KT ({len(X_data)} pt), używam sample({target_size})")
        return X_data.sample(min(target_size, len(X_data)), random_state=42)

    # Konwertuj do contiguous float64 - wymagane przez goodpoints
    X_np = np.ascontiguousarray(X_data.values.astype(np.float64))
    n_output = len(X_np) // (2 ** m_steps)

    if verbose:
        print(f"  → KT: {len(X_data)} pkt, {m_steps} rund → {n_output} pkt w coresecie")

    # Kernel Thinning - zwraca indeksy wybranych punktów
    compressed_indices = kt.thin(
        X=X_np,
        m=m_steps,
        split_kernel=smart_rbf_kernel,
        swap_kernel=smart_rbf_kernel,
        seed=42
    )

    # compressed_indices to indeksy do X_np (tablicy NumPy, 0-based, ciągłe)
    # X_data.iloc[] - używamy iloc bo X_np ma ciągłą indeksację 0..n-1
    coreset = X_data.iloc[compressed_indices]

    # Przytnij jeśli za duże
    if len(coreset) > target_size:
        coreset = coreset.sample(target_size, random_state=42)

    if verbose:
        print(f"  ✅ Coreset gotowy: {len(coreset)} punktów")

    return coreset
