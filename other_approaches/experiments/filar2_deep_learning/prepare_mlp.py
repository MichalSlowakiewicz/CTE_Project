import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import joblib
import sys
import optuna

sys.path.append(str(Path(__file__).parent.parent.parent))

from data.data_loader import load_dataset
from sklearn.metrics import log_loss

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

def objective(trial, X_train, y_train, X_val, y_val, device, input_dim):
    hidden1 = trial.suggest_categorical("hidden1", [64, 128, 256])
    hidden2 = trial.suggest_categorical("hidden2", [32, 64, 128])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    
    model = RobustMLP(input_dim, hidden1=hidden1, hidden2=hidden2, dropout=dropout).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).unsqueeze(1))
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    model.train()
    epochs = 5  # Quick evaluation for optuna
    for epoch in range(epochs):
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
    model.eval()
    val_tensor = torch.FloatTensor(X_val).to(device)
    with torch.no_grad():
        preds = model(val_tensor).cpu().numpy().squeeze()
        
    return log_loss(y_val, preds)

def train_and_cache_mlp():
    print("=== PyTorch MLP & Optuna Optimization ===")
    cache_dir = Path(__file__).parent.parent.parent / 'data' / 'cache_full'
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    data = load_dataset(data_dir=Path(__file__).parent.parent.parent / 'data' / 'ecom-offers')
    X_train, y_train = data['X_train'].values, data['y_train']
    X_val, y_val = data['X_val'].values, data['y_val']
    
    input_dim = X_train.shape[1]
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    
    n_trials = 10
    print(f"\n1. Starting Optuna optimization ({n_trials} trials)...")
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val, device, input_dim), n_trials=n_trials, show_progress_bar=True)
    
    print("\n   Best parameters:")
    best_params = study.best_params
    for k, v in best_params.items():
        print(f"      {k}: {v}")
        
    print("\n2. Training final MLP model with best parameters...")
    final_model = RobustMLP(
        input_dim, 
        hidden1=best_params['hidden1'], 
        hidden2=best_params['hidden2'], 
        dropout=best_params['dropout']
    ).to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(final_model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).unsqueeze(1))
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    epochs = 15
    final_model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = final_model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"   Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(train_loader):.4f}")
        
    final_model.eval()
    
    model_path = cache_dir / 'mlp_model.pt'
    torch.save(final_model.state_dict(), model_path)
    # Save the parameters as well so Fallback Plan can instantiate the correct architecture!
    joblib.dump(best_params, cache_dir / 'mlp_best_params.pkl')
    
    print(f"\n=== Model saved to {model_path} ===")

if __name__ == "__main__":
    train_and_cache_mlp()
