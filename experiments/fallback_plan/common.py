import torch
import torch.nn as nn
import shap
import joblib

# Dynamiczna definicja MLP by móc załadować wagi z PyTorcha
class RobustMLP(nn.Module):
    def __init__(self, input_dim, hidden1=128, hidden2=64, dropout=0.3):
        super(RobustMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

def load_mlp_model(cache_dir, input_dim):
    # Load architecture parameters selected by Optuna
    params_path = cache_dir / 'mlp_best_params.pkl'
    if params_path.exists():
        best_params = joblib.load(params_path)
    else:
        # Fallback if Optuna wasn't run yet
        best_params = {'hidden1': 128, 'hidden2': 64, 'dropout': 0.3}
        
    model = RobustMLP(
        input_dim, 
        hidden1=best_params['hidden1'], 
        hidden2=best_params['hidden2'], 
        dropout=best_params['dropout']
    )
    # The model might be saved as .pth or .pt
    model_file = cache_dir / 'mlp_model.pt'
    if not model_file.exists():
        model_file = cache_dir / 'mlp_model.pth'
        
    model.load_state_dict(torch.load(model_file, weights_only=True))
    model.eval()
    return model

PAIRS = [
    # Filar 1 (Drzewa)
    {"family": "tree", "name": "xgb_tree", "model_file": "xgb_model.pkl", "explainer": "TreeExplainer"},
    {"family": "tree", "name": "rf_tree", "model_file": "rf_model.pkl", "explainer": "TreeExplainer"},
    {"family": "tree", "name": "lgb_tree", "model_file": "lgbm_model.pkl", "explainer": "TreeExplainer"},
    # Filar 2 (Sieci Neuronowe - PyTorch) - Wyrzucone, wyrzucają SegFault z biblioteki SHAP na 100k próbek
    # Filar 3 (EBM Kernel) - Wyrzucony zgodnie z poleceniem (nie radzi sobie ze 100k tła)
    # Nowe zoptymalizowane modele (Fallback Plan)
    {"family": "tree", "name": "catboost_tree", "model_file": "catboost_model.pkl", "explainer": "TreeExplainer"},
    # ExtraTrees wyrzucony - model zajmuje 867MB i przy 100k tła TreeExplainer robi SegFault (brak RAM na samo wczytanie modelu w silniku C++)
    # Modele Liniowe (LogReg, Ridge, Lasso)
    {"family": "linear", "name": "logreg_linear", "model_file": "logreg_model.pkl", "explainer": "LinearExplainer"},
    {"family": "linear", "name": "ridge_linear", "model_file": "ridge_model.pkl", "explainer": "LinearExplainer"},
    {"family": "linear", "name": "lasso_linear", "model_file": "lasso_model.pkl", "explainer": "LinearExplainer"},
    {"family": "linear", "name": "svc_linear", "model_file": "svc_model.pkl", "explainer": "LinearExplainer"}
]

def compute_shap(pair, model, bg_data, X_explain):
    explainer_type = pair['explainer']
    
    if pair['family'] == 'neural':
        # Re-enable CUDA but use background chunking to prevent Segfaults / OOM on massive 100k backgrounds
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        bg_tensor_full = torch.tensor(bg_data.values, dtype=torch.float32).to(device)
        X_tensor = torch.tensor(X_explain.values, dtype=torch.float32).to(device)
        
        chunk_size = 2000
        all_shap_vals = []
        
        for i in range(0, len(bg_tensor_full), chunk_size):
            bg_chunk = bg_tensor_full[i:i+chunk_size]
            if explainer_type == 'DeepExplainer':
                explainer = shap.DeepExplainer(model, bg_chunk)
            elif explainer_type == 'GradientExplainer':
                explainer = shap.GradientExplainer(model, bg_chunk)
            else:
                raise ValueError(f"Unknown explainer {explainer_type}")
                
            chunk_shap_vals = explainer.shap_values(X_tensor)
            if isinstance(chunk_shap_vals, list): 
                chunk_shap_vals = chunk_shap_vals[0]
            
            # Weight by the exact chunk size since the last chunk might be smaller
            all_shap_vals.append(chunk_shap_vals * (len(bg_chunk) / len(bg_tensor_full)))
            
        # Sum the weighted SHAP values (which effectively calculates the mean across all bg samples)
        final_shap_vals = np.sum(all_shap_vals, axis=0)
        return final_shap_vals
            
    elif pair['family'] == 'linear':
        linear_model = model.named_steps[list(model.named_steps.keys())[1]]
        scaler = model.named_steps[list(model.named_steps.keys())[0]]
        bg_scaled = scaler.transform(bg_data)
        X_scaled = scaler.transform(X_explain)
        
        explainer = shap.LinearExplainer(linear_model, bg_scaled)
        return explainer.shap_values(X_scaled)
        
    elif pair['family'] == 'tree':
        if explainer_type == 'TreeExplainer':
            # check_additivity=False to prevent Random Forest failures on tiny float inaccuracies
            explainer = shap.TreeExplainer(model, data=bg_data, feature_perturbation="interventional")
            shap_vals = explainer.shap_values(X_explain, check_additivity=False)
        elif explainer_type == 'GPUTreeExplainer':
            explainer = shap.GPUTreeExplainer(model, data=bg_data, feature_perturbation="interventional")
            shap_vals = explainer.shap_values(X_explain, check_additivity=False)
        
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
        elif len(shap_vals.shape) == 3:
            shap_vals = shap_vals[:,:,1] if shap_vals.shape[2] > 1 else shap_vals[:,:,0]
        return shap_vals
        
    elif pair['family'] == 'additive':
        if explainer_type == 'KernelExplainer':
            # EBM expects dataframe, we pass a subset of predict_proba returning 1D per class
            def predict_fn(x):
                return model.predict_proba(pd.DataFrame(x, columns=bg_data.columns))[:, 1]
            
            # KernelExplainer scales terribly with large backgrounds. Downsample to 1000 for Ground Truth
            if len(bg_data) > 1000:
                print(f"      [Additive] Downsampling background from {len(bg_data)} to 1000 for KernelExplainer")
                bg_data = bg_data.sample(1000, random_state=42)
                
            # shap.kmeans can be used if bg is too large, but we use the exact bg_data points
            explainer = shap.KernelExplainer(predict_fn, bg_data)
            # max_evals restricts exponential explosion
            shap_vals = explainer.shap_values(X_explain)
            return shap_vals
