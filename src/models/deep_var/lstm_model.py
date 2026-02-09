
import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd
import numpy as np
from src.models.deep_var.parametric_model import priori_value_at_risk

# 1. The Single Neuron Model
class SimpleQuantileNeuron(nn.Module):
    def __init__(self, input_size):
        super(SimpleQuantileNeuron, self).__init__()
        # A single linear layer: y = wx + b
        self.linear = nn.Linear(input_size, 1)
        
    def forward(self, x):
        return self.linear(x)

# 2. The Math: Quantile Loss Function
class QuantileLoss(nn.Module):
    def __init__(self, alpha=0.05):
        super(QuantileLoss, self).__init__()
        self.alpha = alpha

    def forward(self, preds, target, weight=0, preds_prior=0):
        errors = target - preds
        # The "Pinball" Logic:
        # max( (alpha-1)*error, alpha*error )
        loss = torch.max((self.alpha - 1) * errors, self.alpha * errors) + weight * (preds - preds_prior)** 2
        return torch.mean(loss)

# --- B. The LSTM Model ---
class QuantileLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2):
        super(QuantileLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        
        last_step_output = out[:, -1, :]
        prediction = self.fc(last_step_output)
        return prediction

def train_model(data, model_type='SimpleQuantileNeuron', alpha=0.05, epochs=500, lr=0.01, rolling=22, 
    split_type={'percentage': 0.8}, regularization_pm=None):

    print(f"--- 🧠 Training Single Neuron Quantile Regressor (alpha={alpha}) ---")

    X = data['X']
    y = data['y']
    dates = data['dates']
    
    num_samples = len(X)

    if 'percentage' in split_type:
        split_idx = int(num_samples * split_type['percentage'])
        window_size = num_samples - split_idx
    else:
        # split_idx = split_type['index']
        date_rep = split_type['date']
        split_idx = len(dates[dates <= pd.to_datetime(date_rep)])
        window_size = 22
    test_size = num_samples - split_idx
    iterate_number = test_size // window_size
    # split_type={'percentage': 0.8}

    if regularization_pm is not None:
        weight = regularization_pm['weight']
        df = regularization_pm['df']
        regularization_pm = regularization_pm
        df_value_at_risk = priori_value_at_risk(df, rolling=rolling, alpha=alpha)
        
    X_test_aux = []
    y_test_aux = [] 
    preds_test_aux = []
    dates_test_aux = []

    for i in range(iterate_number):
        # Slicing tensors
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_test, y_test = X[split_idx:split_idx + window_size], y[split_idx:split_idx + window_size]
        if split_idx >= len(dates):
            break
        
        if regularization_pm is not None:
            date_var = dates[split_idx]
            y_priori = df_value_at_risk.loc[date_var, 'value_at_risk_param']
        else:
            weight = 0
            y_priori = 0
        # Keep dates for plotting later
        dates_train = dates[:split_idx]
        dates_test = dates[split_idx:split_idx + window_size]

        print(f"    Train size: {len(X_train)} | Test size: {len(X_test)}")

        # 2. Init Model & Optimizer
        if model_type == 'SimpleQuantileNeuron':
            model = SimpleQuantileNeuron(input_size=X_train.shape[1])
        elif model_type == 'QuantileLSTM':
            model = QuantileLSTM(input_size=X_train.shape[1])
        
        criterion = QuantileLoss(alpha=alpha)
        optimizer = optim.SGD(model.parameters(), lr=lr) 
    
        # History to track both curves
        history = {"train_loss": [], "test_loss": []}
        
        loss_prev = np.inf

        # 3. Training Loop
        for epoch in range(epochs):
            
            # --- A. Training Step (Update Weights) ---
            model.train() # Set mode to train
            
            # Forward pass on TRAIN data only
            preds_train = model(X_train.unsqueeze(1))
                      
            loss_train = criterion(preds_train, y_train, weight, preds_prior=y_priori)
            
            # Backward pass
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            
            # --- B. Validation Step (Check Performance) ---
            model.eval() # Set mode to evaluation (disables dropout, etc.)
            with torch.no_grad(): # No gradient calculation needed for test
                # Forward pass on TEST data
                preds_test = model(X_test.unsqueeze(1))
                # Calculate Functional (Loss) on Test
                loss_test = criterion(preds_test, y_test, weight, preds_prior=y_priori)
            
            # Store results
            history["train_loss"].append(loss_train.item())
            history["test_loss"].append(loss_test.item())
            
            loss = loss_train.item()
            # if loss == loss_prev:
            if np.abs(1 - loss / loss_prev) < 0.0000001:
                print(f"Early stopping at epoch {epoch}")
                print(f"Loss {loss:.5f}, prev {loss_prev:.5f}")
                break
            else:
                loss_prev = loss
            # Print progress
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Train Loss={loss_train.item():.5f} | Test Loss={loss_test.item():.5f}")
        X_test_aux.append(X_test)
        y_test_aux.append(y_test)
        preds_test_aux.append(preds_test)
        dates_test_aux.append(dates_test)
        split_idx += window_size
    return model, history, (X_train, y_train), (X_test_aux, y_test_aux, preds_test_aux, dates_test_aux)

def analyze_var_functional(df, rolling=132, alpha=0.05, z0=1.645, 
                              std_range=(0, 2), mean_range=(-1, 2), n_points=31, weight=1):
    """
    Analyze the quantile loss functional for different actual mean and std values.
    
    The theoretical optimal VaR is: mu - z0 * sigma
    We test combinations to see which minimizes the loss.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'log_ret' column
    rolling : int
        Rolling window size for calculating mean and std
    alpha : float
        Significance level (e.g., 0.05 for 95% confidence)
    z0 : float
        Z-score for the given alpha (e.g., 1.645 for alpha=0.05)
    std_range : tuple
        Range for std multipliers (min, max). Default: (0, 2)
    mean_range : tuple
        Range for mean multipliers (min, max). Default: (-1, 2)
    n_points : int
        Number of points to test in each dimension. Default: 31
    
    Returns:
    --------
    tuple : results_df, loss_matrix, actual_std_values, actual_mean_values, 
            optimal_params, theoretical_loss, avg_mean, avg_std
    """
    # Prepare data
    df_toy = df.copy()
    df_toy['std'] = df_toy['log_ret'].rolling(rolling).std()
    df_toy['mean'] = df_toy['log_ret'].rolling(rolling).mean()
    df_toy['log_ret_p1'] = df_toy['log_ret'].shift(-1)
    df_toy = df_toy.dropna()
    
    # Calculate average mean and std to create meaningful ranges
    avg_mean = df_toy['mean'].mean()
    avg_std = df_toy['std'].mean()
    var_priori = avg_mean - z0 * avg_std
    
    print(f"Average rolling mean: {avg_mean:.6f}")
    print(f"Average rolling std: {avg_std:.6f}")
    print(f"Testing std multipliers from {std_range[0]} to {std_range[1]}")
    print(f"Testing mean multipliers from {mean_range[0]} to {mean_range[1]}")
    print(f"Grid size: {n_points} × {n_points} = {n_points**2} points")
    
    # Define grid of multipliers using the provided ranges
    std_multipliers = np.linspace(std_range[0], std_range[1], n_points)
    mean_multipliers = np.linspace(mean_range[0], mean_range[1], n_points)
    
    # Convert to actual values for plotting
    actual_std_values = avg_std * std_multipliers
    actual_mean_values = avg_mean * mean_multipliers
    
    # Store results
    results = []
    loss_matrix = np.zeros((len(std_multipliers), len(mean_multipliers)))
    
    # Calculate loss for each combination
    for i, aux_std in enumerate(std_multipliers):
        for j, aux_mean in enumerate(mean_multipliers):
            # Calculate VaR estimate using the multipliers
            var_estimate = -z0 * df_toy['std'] * aux_std + df_toy['mean'] * aux_mean
            
            # Calculate quantile loss
            loss = calculate_quantile_loss(df_toy['log_ret_p1'].values, 
                                          var_estimate.values, 
                                          alpha, weight, var_priori)
            
            loss_matrix[i, j] = loss
            results.append({
                'aux_std': aux_std,
                'aux_mean': aux_mean,
                'actual_std': avg_std * aux_std,
                'actual_mean': avg_mean * aux_mean,
                'loss': loss
            })
    
    results_df = pd.DataFrame(results)
    
    # Find optimal values
    optimal_idx = results_df['loss'].idxmin()
    optimal_params = results_df.loc[optimal_idx]
    
    # Find loss at theoretical optimal (aux_std=1, aux_mean=1)
    theoretical_idx = results_df[(results_df['aux_std'] == 1.0) & 
                                  (results_df['aux_mean'] == 1.0)].index[0]
    theoretical_loss = results_df.loc[theoretical_idx, 'loss']
    
    return results_df, loss_matrix, actual_std_values, actual_mean_values, optimal_params, theoretical_loss, avg_mean, avg_std

def calculate_quantile_loss(returns, var_estimates, alpha=0.05, weight=1, var_priori=0):
    """
    Calculate the quantile loss (pinball loss) for VaR estimates.
    
    Loss = alpha * (y - var) if y >= var
    Loss = (alpha - 1) * (y - var) if y < var
    """
    errors = returns - var_estimates
    loss = np.where(errors >= 0, 
                    alpha * errors, 
                    (alpha - 1) * errors)
    
    return np.mean(loss + weight * (var_priori - var_estimates)**2)
