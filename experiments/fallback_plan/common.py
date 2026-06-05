import numpy as np
import pandas as pd
import shap
import joblib

PAIRS = [
    # Tree models (TreeExplainer — exact, fast)
    {"family": "tree", "name": "xgb_tree", "model_file": "xgb_model.pkl", "explainer": "TreeExplainer"},
    {"family": "tree", "name": "rf_tree", "model_file": "rf_model.pkl", "explainer": "TreeExplainer"},
    {"family": "tree", "name": "lgb_tree", "model_file": "lgbm_model.pkl", "explainer": "TreeExplainer"},
    {"family": "tree", "name": "catboost_tree", "model_file": "catboost_model.pkl", "explainer": "TreeExplainer"},
    # Linear models (LinearExplainer — closed-form, fast)
    {"family": "linear", "name": "logreg_linear", "model_file": "logreg_model.pkl", "explainer": "LinearExplainer"},
    {"family": "linear", "name": "ridge_linear", "model_file": "ridge_model.pkl", "explainer": "LinearExplainer"},
    {"family": "linear", "name": "lasso_linear", "model_file": "lasso_model.pkl", "explainer": "LinearExplainer"},
    {"family": "linear", "name": "svc_linear", "model_file": "svc_model.pkl", "explainer": "LinearExplainer"},
]


def compute_shap(pair, model, bg_data, X_explain):
    """Compute SHAP values for a given model-explainer pair.
    
    Args:
        pair: dict from PAIRS with 'family' and 'explainer' keys
        model: fitted model object
        bg_data: background DataFrame used for feature marginalization
        X_explain: DataFrame of samples to explain
    
    Returns:
        np.ndarray of shape (n_samples, n_features) with SHAP values
    """
    if pair['family'] == 'linear':
        # Pipeline: scaler → linear model
        step_names = list(model.named_steps.keys())
        linear_model = model.named_steps[step_names[1]]
        scaler = model.named_steps[step_names[0]]
        bg_scaled = scaler.transform(bg_data)
        X_scaled = scaler.transform(X_explain)
        
        explainer = shap.LinearExplainer(linear_model, bg_scaled)
        return explainer.shap_values(X_scaled)
        
    elif pair['family'] == 'tree':
        # check_additivity=False to prevent tiny float inaccuracy failures
        explainer = shap.TreeExplainer(
            model, data=bg_data, feature_perturbation="interventional"
        )
        shap_vals = explainer.shap_values(X_explain, check_additivity=False)
        
        # Handle multi-output (e.g. binary classification returns list of 2)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
        elif len(shap_vals.shape) == 3:
            shap_vals = shap_vals[:, :, 1] if shap_vals.shape[2] > 1 else shap_vals[:, :, 0]
        return shap_vals
    
    else:
        raise ValueError(f"Unknown family: {pair['family']}")
