"""
Test diagnostyczny - sprawdza DOKŁADNIE jak goodpoints wywołuje kernel
i weryfikuje że nasz fix działa.
"""

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from numpy import newaxis

print("=" * 60)
print("DIAGNOSTYKA: Jak goodpoints wywołuje kernel?")
print("=" * 60)

# Symulujemy różne wywołania jak robi to goodpoints
X = np.random.randn(10, 4)  # 10 punktów, 4 cechy

calls_log = []

def logging_kernel(A, B):
    """Kernel który loguje każde wywołanie żebyśmy mogli je przeanalizować."""
    calls_log.append({
        'A_shape': A.shape,
        'B_shape': B.shape,
        'shares_memory': np.shares_memory(A, B),
        'same_object': A is B,
        'A_is_X': A is X,
        'B_is_X': B is X,
    })
    # Zwracamy coś co nie spowoduje błędu w tej fazie
    A2d = np.atleast_2d(A)
    B2d = np.atleast_2d(B)
    K = rbf_kernel(A2d, B2d)
    if K.shape[0] == 1 or K.shape[1] == 1:
        return K.flatten()
    return K

# Symulujemy wywołania z split_X
print("\n--- Symulacja wywołań z split_X ---")
i = 5
coreset = np.arange(10)
i_array = coreset[i, newaxis]  # np.array([5])

print(f"i_array = {i_array}, shape = {i_array.shape}")
print(f"X[i_array].shape = {X[i_array].shape}")          # (1, 4)
print(f"X[coreset[:i+1]].shape = {X[coreset[:i+1]].shape}")  # (6, 4)

# Test 1: kernel(X[i_array], X[coreset[:i+1]])  -- tryb normalny
r1 = logging_kernel(X[i_array], X[coreset[:i+1]])
print(f"\nTest 1: kernel(X[i_array], X[:i+1]) -> shape {r1.shape} ✓ (oczekiwany 1D wektor)")

# Symulujemy wywołanie z refine_X linia 678:
# sufficient_stat = kernel(X, X)/coreset_size - 2 * meanK
print(f"\n--- Kluczowe wywołanie z refine_X linia 678 ---")
print(f"kernel(X, X) gdzie X.shape = {X.shape}")
print(f"np.shares_memory(X, X) = {np.shares_memory(X, X)}")

# Test 2: kernel(X, X) -- to jest BUG TRAP
r2_raw = rbf_kernel(X, X)
print(f"\nBEZ fixu: rbf_kernel(X, X).shape = {r2_raw.shape}  ← to jest macierz NxN!")
print(f"np.argmin(r2_raw/1) = {np.argmin(r2_raw/1)}  ← flat index > n, CRASH!")

print("\n" + "=" * 60)
print("TESTUJEMY RÓŻNE WERSJE KERNELA:")
print("=" * 60)

# ============================================================
# WERSJA STARA (buggy)
# ============================================================
def kernel_v1_buggy(X, Y):
    """Stara wersja - powoduje IndexError."""
    K = rbf_kernel(np.atleast_2d(X), np.atleast_2d(Y))
    if K.shape[0] == 1 or K.shape[1] == 1:
        return K.flatten()
    return K

# ============================================================
# WERSJA 2 - shares_memory (mój poprzedni fix)
# Problem: X_data.sample() tworzy kopię, więc shares_memory=False
# dla różnych wywołań split. Ale dla kernel(X_np, X_np) powinno działać?
# ============================================================
def kernel_v2_shares_memory(X, Y):
    """Fix z shares_memory."""
    X2d = np.atleast_2d(X)
    Y2d = np.atleast_2d(Y)
    if (X2d.ndim == 2 and Y2d.ndim == 2
            and X2d.shape == Y2d.shape
            and X2d.shape[0] > 1
            and np.shares_memory(X, Y)):
        return np.ones(X2d.shape[0])  # diagonal RBF = 1
    K = rbf_kernel(X2d, Y2d)
    if K.shape[0] == 1 or K.shape[1] == 1:
        return K.flatten()
    return K

# ============================================================
# WERSJA 3 - PRAWIDŁOWY FIX: unikaj stores_K, użyj store_K=True
# ============================================================
# Zamiast walczyć z kernel wrapper, użyjemy store_K=True w kt.thin
# To sprawi że goodpoints użyje swap_K zamiast swap_X, 
# i unika problematycznego wywołania kernel(X, X)

# ============================================================
# WERSJA 4 - DEFINITYWNY FIX: własna implementacja kernel thinning
# używając store_K=True
# ============================================================

print("\n[TEST] kernel_v2 z kernel(X, X):")
result_v2 = kernel_v2_shares_memory(X, X)
print(f"  shares_memory(X,X) = {np.shares_memory(X, X)}")
print(f"  Wynik shape: {result_v2.shape}")
print(f"  Czy to wektor długości n={len(X)}? {'✅ TAK' if result_v2.shape == (len(X),) else '❌ NIE - to macierz!'}")

# Test z kopią (tak jak robi X_data.values)
X_copy = X.copy()
print(f"\n[TEST] kernel_v2 z kernel(X_copy, X_copy) [KOPIA - jak .values]:")
result_v2_copy = kernel_v2_shares_memory(X_copy, X_copy)
print(f"  shares_memory(X_copy, X_copy) = {np.shares_memory(X_copy, X_copy)}")
print(f"  Wynik shape: {result_v2_copy.shape}")
print(f"  Czy to wektor długości n? {'✅ TAK' if result_v2_copy.shape == (len(X),) else '❌ NIE - PROBLEM!'}")

print("\n" + "=" * 60)
print("ROZWIĄZANIE: Użyć store_K=True w kt.thin")
print("=" * 60)
print("""
Problem: np.shares_memory() działa tylko gdy X i Y są TĄ SAMĄ tablicą.
Ale goodpoints w refine_X wywołuje: kernel(X, X) gdzie X to ten sam obiekt.
np.shares_memory(X, X) = True  ← OK!

ALE: jeśli X_np = X_data.values.astype(np.float64), to X_np is X_np  ← True
Więc fix v2 działa dla wywołania kernel(X_np, X_np).

Dlaczego więc nowy błąd 'index 1 is out of bounds'?

Musi być inny problem. Sprawdźmy z pełnym debuggiem goodpoints...
""")

# Faktyczny test z goodpoints
print("=" * 60)
print("FAKTYCZNY TEST Z GOODPOINTS")
print("=" * 60)

try:
    import goodpoints.kt as kt
    print("✅ goodpoints załadowany")
    
    # Małe dane do testów
    np.random.seed(42)
    X_test = np.random.randn(64, 4).astype(np.float64)  # 64 punkty, power of 2
    target_size = 8
    m = int(np.floor(np.log2(len(X_test) / target_size)))
    print(f"\nTest: {len(X_test)} punktów → {target_size} docelowo, m={m}")
    
    print("\n--- Test z kernel_v1 (stary, buggy) ---")
    try:
        idx = kt.thin(X_test, m=m, split_kernel=kernel_v1_buggy, swap_kernel=kernel_v1_buggy, seed=42)
        print(f"✅ Działa! indices shape: {idx.shape}")
    except IndexError as e:
        print(f"❌ IndexError: {e}")
    
    print("\n--- Test z kernel_v2 (shares_memory fix) ---")
    try:
        idx = kt.thin(X_test, m=m, split_kernel=kernel_v2_shares_memory, swap_kernel=kernel_v2_shares_memory, seed=42)
        print(f"✅ Działa! indices shape: {idx.shape}")
    except IndexError as e:
        print(f"❌ IndexError: {e}")
    
    print("\n--- Test z store_K=True (omija problematyczny kod) ---")
    def simple_kernel(X, Y):
        """Prosty kernel - tylko standardowe wywołania."""
        K = rbf_kernel(np.atleast_2d(X), np.atleast_2d(Y))
        if K.shape[0] == 1 or K.shape[1] == 1:
            return K.flatten()
        return K
    
    try:
        idx = kt.thin(X_test, m=m, split_kernel=simple_kernel, swap_kernel=simple_kernel, seed=42, store_K=True)
        print(f"✅ Działa! indices shape: {idx.shape}")
        print(f"   Pierwsze 5 indeksów: {idx[:5]}")
    except Exception as e:
        print(f"❌ Błąd: {type(e).__name__}: {e}")

except ImportError:
    print("❌ goodpoints nie jest zainstalowany")
