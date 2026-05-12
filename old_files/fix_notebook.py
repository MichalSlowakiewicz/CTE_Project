"""
fix_notebook_v2.py - Naprawia prototype.ipynb.

Obecny problem: kernel zwraca pelna macierz K dla przypadku Macierz vs Macierz,
ale goodpoints wymaga wektora diagonali gdy wywoluje kernel(X, X).
"""

import json

NB_PATH = "prototype.ipynb"

# Kod cell 4 - poprawiony, przetestowany
NEW_CELL4_SOURCE = [
    "# ==========================================\n",
    "# CELL 4: Compress Then Explain (CTE) Core\n",
    "# ==========================================\n",
    "\n",
    "def smart_rbf_kernel(X, Y):\n",
    "    \"\"\"\n",
    "    Wrapper dla jadra RBF kompatybilny z goodpoints.\n",
    "\n",
    "    KLUCZOWY BUG DO NAPRAWY:\n",
    "    goodpoints/kt.py linia 678 w refine_X wywoluje:\n",
    "        sufficient_stat = kernel(X, X) / coreset_size\n",
    "    gdzie X to pelna macierz (n, d). Oczekuje wektora (n,) - diagonal.\n",
    "    BEZ fixu: rbf_kernel zwraca (n, n) -> argmin daje flat index > n -> IndexError!\n",
    "\n",
    "    Tryby wywolan:\n",
    "      1) kernel(X[i], X)         -> X[i] jest 1D (d,)   -> wektor (n,)\n",
    "      2) kernel(X[i], X[subset]) -> X[i] jest 1D (d,)   -> wektor (|subset|,)\n",
    "      3) kernel(X, X)            -> X jest 2D (n, d)     -> DIAGONAL (n,) <- FIX!\n",
    "    \"\"\"\n",
    "    X2d = np.atleast_2d(X)\n",
    "    Y2d = np.atleast_2d(Y)\n",
    "\n",
    "    # === KRYTYCZNY FIX ===\n",
    "    # Wykrywamy tryb 3: kernel(X, X) - ten sam obiekt po obu stronach.\n",
    "    # np.shares_memory(X, X) == True gdy to ten sam obiekt w pamieci.\n",
    "    # Dla RBF: k(x, x) = exp(-gamma * ||x-x||^2) = exp(0) = 1.0\n",
    "    # Wiec diagonal zawsze = np.ones(n)\n",
    "    if (X2d.ndim == 2\n",
    "            and Y2d.ndim == 2\n",
    "            and X2d.shape == Y2d.shape\n",
    "            and X2d.shape[0] > 1\n",
    "            and np.shares_memory(X, Y)):\n",
    "        return np.ones(X2d.shape[0])\n",
    "\n",
    "    # Tryby 1 i 2: standardowe wywolanie\n",
    "    K = rbf_kernel(X2d, Y2d)\n",
    "\n",
    "    # Zawsze spłaszcz do 1D jesli jeden z wymiarow = 1\n",
    "    if K.shape[0] == 1 or K.shape[1] == 1:\n",
    "        return K.flatten()\n",
    "\n",
    "    return K\n",
    "\n",
    "\n",
    "def get_cte_background(X_data, target_size=50):\n",
    "    \"\"\"\n",
    "    Zwraca tlo (background) dla SHAP obliczone metoda Kernel Thinning.\n",
    "    Implementuje: Baniecki et al., ICLR 2025 (Compress Then Explain).\n",
    "    \"\"\"\n",
    "    if not HAS_GOODPOINTS:\n",
    "        return shap.sample(X_data, target_size)\n",
    "\n",
    "    # Reset indeksow - WAZNE gdy X_data jest slicejem z innego df\n",
    "    X_clean = X_data.copy().reset_index(drop=True)\n",
    "\n",
    "    m_steps = int(np.floor(np.log2(len(X_clean) / target_size)))\n",
    "    m_steps = min(m_steps, 6)  # ograniczenie dla bezpieczenstwa\n",
    "\n",
    "    if m_steps <= 0:\n",
    "        return X_clean.sample(min(target_size, len(X_clean)), random_state=42)\n",
    "\n",
    "    # ascontiguousarray - goodpoints wymaga ciaglej pamieci\n",
    "    X_np = np.ascontiguousarray(X_clean.values.astype(np.float64))\n",
    "    n_out = len(X_np) // (2 ** m_steps)\n",
    "    print(f\"  -> KT: {len(X_clean)} pkt, {m_steps} rund -> ~{n_out} pkt coresetu\")\n",
    "\n",
    "    compressed_indices = kt.thin(\n",
    "        X=X_np,\n",
    "        m=m_steps,\n",
    "        split_kernel=smart_rbf_kernel,\n",
    "        swap_kernel=smart_rbf_kernel,\n",
    "        seed=42\n",
    "    )\n",
    "\n",
    "    # WAZNE: iloc bo compressed_indices to indeksy 0..n-1 do X_np\n",
    "    coreset = X_clean.iloc[compressed_indices]\n",
    "\n",
    "    if len(coreset) > target_size:\n",
    "        coreset = coreset.sample(target_size, random_state=42)\n",
    "\n",
    "    print(f\"  OK Coreset gotowy: {len(coreset)} punktow\")\n",
    "    return coreset\n",
    "\n",
    "\n",
    "print(\"OK CTE Core functions defined (smart_rbf_kernel z FIX shares_memory).\")\n"
]


def fix():
    with open(NB_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cells = nb['cells']
    found = False

    for i, cell in enumerate(cells):
        if cell.get('cell_type') != 'code':
            continue
        src = ''.join(cell.get('source', []))
        if 'smart_rbf_kernel' in src and 'get_cte_background' in src:
            print(f"  Znaleziono Cell {i+1} - zastepujem...")

            # Sprawdz jaki kernel tam jest
            if 'shares_memory' in src:
                print("  -> Juz ma fix shares_memory, ale cos jest nie tak")
            elif 'ndim_X == 2 and ndim_Y == 2' in src:
                print("  -> Ma stara wersje z ndim - to jest BUG (case 4 zwraca macierz NxN)!")
            else:
                print("  -> Inny wariant")

            cell['source'] = NEW_CELL4_SOURCE
            cell['outputs'] = []
            cell['execution_count'] = None
            found = True
            break

    if not found:
        print("BLAD: Nie znaleziono komorki!")
        return

    nb['cells'] = cells
    with open(NB_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("  Zapisano!")


if __name__ == "__main__":
    print("Naprawiam prototype.ipynb...")
    fix()
    print("\nGOTOWE - teraz w Jupyter: Kernel -> Restart & Run All")
