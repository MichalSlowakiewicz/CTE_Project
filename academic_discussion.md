# Coreset Tree Explainer (CTE): Academic Discussion & Findings

This document synthesizes the overarching scientific conclusions drawn from the `v15` experimental pipeline. Our investigation evaluates the efficacy of Coreset Tree Explainer (CTE) against Random sampling across two distinct model families: Tree-based ensembles (XGBoost, Random Forest, LightGBM, CatBoost) and Linear models (Logistic Regression, Ridge, Lasso, SVC).

## Methodological Pivot: Oracle Evaluation

A significant divergence in our methodology from classical CTE implementations is the abandonment of the computationally prohibitive `KernelExplainer` in favor of mathematically exact, highly optimized C++ native explainers (`TreeExplainer` and `LinearExplainer`).

By doing so, we essentially conduct an **Oracle Evaluation**:
1. The exact Ground Truth (computed on the full 100,000 background points) can be calculated in fractions of a second.
2. This allows us to rigorously isolate and evaluate the *approximation error* introduced by the compression algorithms (CTE vs Random) without it being confounded by the inaccuracy or sampling noise of the Ground Truth itself.

If we can mathematically prove that CTE reconstructs the expected value distribution better than Random on these "fast" model classes, this supremacy theoretically transfers to "slow" model classes (e.g., Deep Neural Networks with `KernelExplainer`) where the exact Ground Truth is intractable but the mathematical nature of the expected-value approximation remains identical.

---

## RQ1: Efficiency and Convergence
**Hypothesis:** *CTE converges to the true SHAP distribution with lower approximation error than uniform Random sampling across various background sizes.*

**Findings:**
Across both Global Feature Importance (Global MAE) and Per-Sample Explanation (Local MAE), CTE consistently demonstrates lower error compared to Random sampling. This superiority is clearly visible on the horizontal bar charts for all background sizes ($N \in \{4, \dots, 1024\}$). 
The Kernel Thinning algorithm effectively constructs a representative coreset that minimizes the Maximum Mean Discrepancy (MMD) to the original distribution, yielding more accurate marginals than naive sampling.

## RQ2: Temporal Drift and Feature Tracking
**Hypothesis:** *CTE accurately tracks temporal shifts in feature importance over consecutive time windows.*

**Findings:**
In dynamic environments (e.g., e-commerce data with continuous concept drift), feature importance is not static. Our 5-window temporal analysis proves that a CTE background of size $N=128$ tightly mimics the Ground Truth's feature importance trajectories. Random sampling exhibits visible variance and struggles to maintain the exact temporal ranking of the Top 5 features, whereas CTE preserves the ordinality and magnitude of the drift.

## RQ3: Generalization Gap (Train vs Validation)
**Hypothesis:** *CTE reliably exposes the discrepancy in feature reliance between training and validation data.*

**Findings:**
Comparing feature importance generated on training versus validation sets is a critical technique for diagnosing model overfitting. Our analysis confirms that CTE accurately replicates the $Validation / Train$ importance multipliers observed in the Ground Truth. By capturing the core distribution of the training set, CTE provides a reliable surrogate for auditing models without deploying the full 100k background dataset.

## RQ4: Architecture Agnosticism
**Hypothesis:** *Does the superiority of CTE hold consistently across fundamentally different model families (Tree-based ensembles vs. Linear models)?*

**Findings:**
By aggregating the learning curves (MAE vs. Background Size) across the 4 Tree models and 4 Linear models, we observed a unified trend: CTE consistently outperforms Random sampling regardless of the underlying algorithm. This proves that the compression is truly **architecture-agnostic**.

**Crucial Caveat on Breakeven Time:** 
Because `TreeExplainer` evaluates 100,000 samples in milliseconds, the fixed $O(N^2)$ build time of the CTE kernel (approx. 3 seconds in Python) means that CTE **never strictly pays off in execution time** for these native explainers. The evaluation speed is too fast for the compression overhead to amortize. 
Therefore, CTE is mathematically superior in *accuracy per sample*, but from an engineering perspective, it should strictly be utilized as an accelerator for computationally expensive model-agnostic explainers (like `KernelExplainer` or `DeepExplainer`), while native `TreeExplainer` should simply ingest the full dataset.
