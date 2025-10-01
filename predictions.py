import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error


### 1. PREPROCESSING DATA
# Download data from yfinance
spy_df = yf.download(
    "SPY",
    start="2020-01-01",
    end="2025-08-31",
    progress=False,
)

spy_df['Close'].plot(title="S&P 500")


# DF to NP
spy_np = spy_df['Close'].to_numpy()

# NP to PT tensor
spy_tensor = torch.tensor(spy_np)
date_values = torch.arange(1, spy_tensor.size(0) + 1, dtype=torch.int32).unsqueeze(dim=1)


## Train-test split
train_size = int(0.8*len(spy_tensor))
train_prices = spy_tensor[:train_size]
test_prices = spy_tensor[train_size:]
train_dates = date_values[:train_size]
test_dates = date_values[train_size:]


## Normalisation
scaler = MinMaxScaler()
train_prices_np = train_prices.detach().cpu().numpy().reshape(-1, 1)
test_prices_np = test_prices.detach().cpu().numpy().reshape(-1, 1)
scaler.fit(train_prices_np)

train_scaled = torch.tensor(scaler.transform(train_prices_np), dtype=torch.float32)
test_scaled = torch.tensor(scaler.transform(test_prices_np), dtype=torch.float32)


## Time Periods
# note to self: mistake to avoid -- not factoring in time periods in the correct order
def time_periods(data, window_size=30):
    X, y = [], []
    for i in range(len(data)-window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return torch.stack(X).float(), torch.stack(y).float()

X_train, y_train = time_periods(train_scaled)
X_test, y_test = time_periods(test_scaled)


### 2. TRAINING & EVALUATING
# Model
class StockPricePredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=10, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, out_features=1)

    def forward(self, x):
        output, _ = self.lstm(x)
        output = self.fc(output[:, -1, :])
        return output
    
modelV0 = StockPricePredictor()
loss_fn = nn.MSELoss()
optimiser = torch.optim.Adam(modelV0.parameters(), lr=0.001)


## Training Loop
epochs = 1001
total_train_loss, total_train_mae, batches = 0.0, 0, 0.0
for epoch in range(epochs):
    modelV0.train()
    optimiser.zero_grad()
    train_preds = modelV0(X_train)
    train_loss = loss_fn(train_preds, y_train)
    train_loss.backward()
    optimiser.step()

    # Metrics
    total_train_loss += train_loss
    batches += 1
    avg_train_loss = total_train_loss / batches
    
    y_train_true = y_train.detach().cpu().numpy().flatten()
    y_train_preds = train_preds.detach().cpu().numpy().flatten()
    train_mae = mean_absolute_error(y_train_true, y_train_preds)


    ## Testing Loop
    modelV0.eval()
    with torch.inference_mode():
        total_test_loss, total_test_mae = 0.0, 0.0

        test_preds = modelV0(X_test)
        test_loss = loss_fn(test_preds, y_test)
        
        # Metrics
        total_test_loss += test_loss
        avg_test_loss = total_test_loss / batches

        y_test_true = y_test.detach().cpu().numpy().flatten()
        y_test_preds = test_preds.detach().cpu().numpy().flatten()
        test_mae = mean_absolute_error(y_test_true, y_test_preds)
    
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Train Loss: {avg_train_loss:.5f} | Train MAE: {train_mae:.5f} | Test Loss: {avg_test_loss:.5f} | Test MAE: {test_mae:.5f}")



### 3. SCALING BACK & VISUALISATION
# Scale back to actual price values with inverse_transform
y_train_true_unscaled = scaler.inverse_transform(y_train.detach().cpu().numpy().reshape(-1, 1))
y_train_preds_unscaled = scaler.inverse_transform(y_train_preds.reshape(-1, 1))
y_test_true_unscaled = scaler.inverse_transform(y_test.detach().cpu().numpy().reshape(-1, 1))
y_test_preds_unscaled = scaler.inverse_transform(y_test_preds.reshape(-1, 1))


## Values of time
'''
Sample of simple integer values:
train_dates = range(len(y_train_true_unscaled))
test_dates  = range(len(y_train_true_unscaled), 
                   len(y_train_true_unscaled) + len(y_test_true_unscaled))
'''

# For clarity's sake, I'll use the date_range function from pd
dates = pd.date_range(start="2020-01-01", periods=len(y_train_true_unscaled) + len(y_test_true_unscaled), freq="D")
train_dates = dates[:len(y_train_true_unscaled)]
test_dates = dates[len(y_train_true_unscaled):]


## Visualisation
plt.figure(figsize=(8,8))

plt.plot(train_dates, y_train_true_unscaled, label="Actual Train Prices", color="blue")
plt.plot(train_dates, y_train_preds_unscaled, label="Predicted Train Prices", color="cyan")
plt.plot(test_dates, y_test_true_unscaled, label="Actual Test Prices", color="green")
plt.plot(test_dates, y_test_preds_unscaled, label="Predicted Test Prices", color="red")

plt.xlabel("Date")
plt.ylabel("Price of $SPY (USD)")
plt.title("Predicted vs Actual Prices of $SPY against Time")
plt.legend()
plt.show()
plt.savefig("spy_preds.png")