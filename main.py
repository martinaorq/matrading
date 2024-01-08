import numpy as np
import pandas as pd
import requests
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Constants
API_KEY = 'RS9RLRIIJ4I9E1YN'
SYMBOL = 'AAPL'

class CustomScaler(TransformerMixin, BaseEstimator):
    def __init__(self):
        self.scalers = []

    def fit(self, X, y=None):
        self.scalers = []
        for i in range(X.shape[2]):
            scaler = StandardScaler()
            self.scalers.append(scaler.fit(X[:, :, i]))
        return self

    def transform(self, X):
        X_scaled = np.zeros(X.shape)
        for i in range(X.shape[2]):
            X_scaled[:, :, i] = self.scalers[i].transform(X[:, :, i])
        return X_scaled

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

# Fetch stock data from Alpha Vantage
def fetch_stock_data(symbol, api_key):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={API_KEY}&outputsize=full'
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data['Time Series (Daily)']).T
    df = df.rename(columns={'1. open': 'open', '2. high': 'high', '3. low': 'low', '4. close': 'close',
                            '5. volume': 'volume'})
    df = df.apply(pd.to_numeric)
    df.index = pd.to_datetime(df.index)
    return df


# Preprocess the data
def preprocess_data(df, n_lags):
    X, y = [], []
    for i in range(n_lags, len(df)):
        X.append(df.iloc[i - n_lags:i].values)
        y.append(df.iloc[i]['close'])
    return np.array(X), np.array(y)


# Function to create the LSTM model
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# Fetch and preprocess data
df = fetch_stock_data(SYMBOL, API_KEY)
n_lags = 60  # number of days used to predict the next day
X, y = preprocess_data(df, n_lags)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=False)

# Create a pipeline
pipeline = Pipeline([
    ('scaler', CustomScaler()),
    ('model',
     KerasRegressor(build_fn=create_lstm_model, input_shape=(n_lags, X.shape[2]), epochs=50, batch_size=32, verbose=0))
])

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Make predictions
predictions = pipeline.predict(X_test)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(y_test, color='blue', label='Actual Stock Price')
plt.plot(predictions, color='red', label='Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()


# Function to predict the next days
def predict_next_days(model, last_60_days, days=30, n_features=X.shape[2]):
    predictions = []
    current_batch = last_60_days.reshape((1, n_lags, n_features))

    for i in range(days):
        # Get the prediction value for the first instance
        current_pred = model.predict(current_batch)

        # Ensure current_pred is an array (in case it's a scalar)
        current_pred = np.array([current_pred]).reshape(1, 1, 1)

        # Append the prediction into the array
        predictions.append(current_pred[0, 0, 0])

        # Use the prediction to update the batch and remove the first value
        # Replace the last feature of the last timestep with the predicted value
        new_timestep = current_batch[:, -1:, :].copy()
        new_timestep[:, :, -1] = current_pred[:, :, :]
        current_batch = np.append(current_batch[:, 1:, :], new_timestep, axis=1)

    return np.array(predictions)

# Preparing the last 60 days from the test data
last_60_days = X_test[-1]

# Predict the next 30 days
predicted_prices = predict_next_days(pipeline, last_60_days, 30)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(predicted_prices, color='red', label='Predicted Stock Prices for the Next 30 Days')
plt.title('Future Stock Price Prediction')
plt.xlabel('Days')
plt.ylabel('Predicted Price')
plt.legend()
plt.show()