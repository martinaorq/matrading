import numpy as np
import pandas as pd
import requests
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Constants
API_KEY = 'RS9RLRIIJ4I9E1YN'
SYMBOL = 'AAPL'


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


# Rolling window normalization
def rolling_window_normalization(data, window_size=20):
    rolling_mean = data.rolling(window=window_size).mean()
    rolling_std = data.rolling(window=window_size).std()
    normalized_data = (data - rolling_mean) / rolling_std
    normalized_data = normalized_data.dropna()
    return normalized_data


# Preprocess data with rolling window normalization
def preprocess_data(data, window_size=20):
    normalized_data = rolling_window_normalization(data, window_size)
    return normalized_data


# Inverse transform for predictions (modified to handle the lack of a scaler object)
def inverse_transform_predictions(predictions, original_data, window_size=20):
    # Get the last 'len(predictions)' rolling means and standard deviations
    rolling_mean = original_data.rolling(window=window_size).mean().iloc[-len(predictions):]
    rolling_std = original_data.rolling(window=window_size).std().iloc[-len(predictions):]

    # Convert the rolling mean and standard deviation to NumPy arrays
    rolling_mean = rolling_mean.to_numpy()
    rolling_std = rolling_std.to_numpy()

    # Ensure predictions is a flat array
    predictions = predictions.ravel()

    # Element-wise operation to inverse the normalization
    inversed_predictions = (predictions * rolling_std) + rolling_mean
    return inversed_predictions


# Create dataset for LSTM
def create_dataset(data, time_step=60):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), :])
        Y.append(data[i + time_step, 0])  # Predicting next 'open' value
    return np.array(X), np.array(Y)


# Build LSTM model
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# Main program
if __name__ == "__main__":
    # Fetch and preprocess data
    stock_data = fetch_stock_data(SYMBOL, API_KEY)
    window_size = 20  # Define the window size for rolling normalization

    # Apply preprocessing to the entire dataset
    processed_data = preprocess_data(stock_data[['open', 'high', 'low', 'close', 'volume']], window_size)

    # Split data into training and testing before normalization
    train_size = int(len(processed_data) * 0.67)
    train_data, test_data = processed_data[:train_size], processed_data[train_size:]

    time_step = 60
    # Create train and test datasets
    X_train, y_train = create_dataset(train_data.values, time_step)
    X_test, y_test = create_dataset(test_data.values, time_step)

    # Build and train the model
    model = build_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=2,
              callbacks=[EarlyStopping(monitor='loss', patience=10)], validation_data=(X_test, y_test))

    # Predict
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Inverse transform for predictions
    train_predict = inverse_transform_predictions(train_predict, stock_data['open'][:train_size], window_size)
    test_predict = inverse_transform_predictions(test_predict, stock_data['open'][train_size:], window_size)

    # Plotting
    plt.figure(figsize=(12, 6))

    # Since we're predicting the 'open' prices, we will inverse transform only this column
    actual_data = stock_data['open']

    # Plot actual data
    plt.plot(actual_data.index, actual_data.values, label='Actual Data')

    # Prepare the training predictions for plotting
    train_predict_plot = np.empty_like(actual_data)
    train_predict_plot[:] = np.nan
    train_predict_plot[window_size + time_step:window_size + len(train_predict) + time_step] = train_predict.ravel()

    # Prepare the test predictions for plotting
    test_predict_plot = np.empty_like(actual_data)
    test_predict_plot[:] = np.nan

    # Calculate the exact indices where test predictions should start and end
    test_predict_start = window_size + len(train_predict) + (time_step * 2)
    test_predict_end = test_predict_start + len(test_predict)

    # Ensure that the test predictions fit exactly into the allocated space
    test_predict_plot[test_predict_start:test_predict_end] = test_predict.ravel()

    # Plot train prediction and test prediction
    plt.plot(actual_data.index, train_predict_plot, label='Train Prediction')
    plt.plot(actual_data.index, test_predict_plot, label='Test Prediction')

    plt.legend()
    plt.show()

