import numpy as np
from scipy.stats import spearmanr, pearsonr

def mean_absolute_shap(shap_values):
    """
    Computes global feature importances as the mean absolute SHAP values
    across all samples.
    """
    return np.mean(np.abs(shap_values), axis=0)

def spearman_rank_correlation(shap_a, shap_b, global_importance=True):
    """
    Computes the Spearman rank correlation between two sets of SHAP values.
    
    Args:
        shap_a, shap_b: Arrays of shape (n_samples, n_features)
        global_importance: If True, computes correlation on global feature importances.
                           If False, computes correlation per sample and averages.
    """
    if global_importance:
        imp_a = mean_absolute_shap(shap_a)
        imp_b = mean_absolute_shap(shap_b)
        corr, _ = spearmanr(imp_a, imp_b)
        return corr
    else:
        corrs = []
        for i in range(len(shap_a)):
            corr, _ = spearmanr(np.abs(shap_a[i]), np.abs(shap_b[i]))
            if not np.isnan(corr):
                corrs.append(corr)
        return np.mean(corrs)

def pearson_correlation(shap_a, shap_b, global_importance=True):
    """
    Computes the Pearson correlation between two sets of SHAP values.
    """
    if global_importance:
        imp_a = mean_absolute_shap(shap_a)
        imp_b = mean_absolute_shap(shap_b)
        corr, _ = pearsonr(imp_a + 1e-10, imp_b + 1e-10)
        return corr
    else:
        corrs = []
        for i in range(len(shap_a)):
            corr, _ = pearsonr(np.abs(shap_a[i]) + 1e-10, np.abs(shap_b[i]) + 1e-10)
            if not np.isnan(corr):
                corrs.append(corr)
        return np.mean(corrs)

def mae_shap(shap_a, shap_b, global_importance=True):
    """
    Computes the Mean Absolute Error (MAE) between two sets of SHAP values.
    Lower is better. Represents the approximation error.
    """
    if global_importance:
        imp_a = mean_absolute_shap(shap_a)
        imp_b = mean_absolute_shap(shap_b)
        return np.mean(np.abs(imp_a - imp_b))
    else:
        maes = []
        for i in range(len(shap_a)):
            maes.append(np.mean(np.abs(shap_a[i] - shap_b[i])))
        return np.mean(maes)

def top_k_overlap(shap_a, shap_b, k=10, global_importance=True):
    """
    Computes the overlap of the top-k features between two sets of SHAP values.
    """
    if global_importance:
        imp_a = mean_absolute_shap(shap_a)
        imp_b = mean_absolute_shap(shap_b)
        top_k_a = set(np.argsort(imp_a)[-k:])
        top_k_b = set(np.argsort(imp_b)[-k:])
        return len(top_k_a.intersection(top_k_b)) / k
    else:
        overlaps = []
        for i in range(len(shap_a)):
            top_k_a = set(np.argsort(np.abs(shap_a[i]))[-k:])
            top_k_b = set(np.argsort(np.abs(shap_b[i]))[-k:])
            overlaps.append(len(top_k_a.intersection(top_k_b)) / k)
        return np.mean(overlaps)

def speedup_ratio(baseline_time, optimized_time):
    return baseline_time / optimized_time
