# StockPricePredictor
This passion project of mine is an attempt to tackle the prediction of $SPY prices, which follow the S&P 500 Index Fund. The workflow of this project involves:
* Preprocessing data:
  - Downloading data with `yfinance` & making a train-test split.
  - Normalisation with `MinMaxScaler()` from `sklearn.preprocessing` for training and testing.
* Training & Evaluating
  - Using an LSTM model for `torch.nn` alongisde an `MSELoss()` for the loss function and `Adam` for the optimiser
  - Metrics: Loss, MAE
* Scaling Back & Visualisation
  - Scaling back normalised values to actual price values with `inverse_transform()`
  - Visualisation: Matplotlib -- displays predicted vs actual train and test prices for $SPY
 
The workflow above successfully achieved a Train MAE of 0.01614 and a Test MAE of 0.03837 after 1000 epochs of training
