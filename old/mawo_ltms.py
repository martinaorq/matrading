import requests
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Your AlphaVantage API Key
API_KEY = 'RS9RLRIIJ4I9E1YN'


def fetch_stock_data(symbol):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={API_KEY}&outputsize=full'
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data['Time Series (Daily)']).T
    df = df.rename(columns={'1. open': 'open', '2. high': 'high', '3. low': 'low', '4. close': 'close',
                            '5. volume': 'volume'})
    df = df.apply(pd.to_numeric)
    df.index = pd.to_datetime(df.index)
    return df


def preprocess_data(df):
    # Handling missing values - if any
    df = df.dropna()

    # Normalization - can be important for some models
    df = (df - df.mean()) / df.std()

    return df


def predict_future(model, df, days=30):
    future_predictions = []
    recent_data = df[-5:]  # Last 5 days of data

    for _ in range(days):
        # Create lag features from recent_data
        input_features = []
        for i in range(1, 6):
            input_features.append(recent_data['close'].shift(i).iloc[-1])
        input_features.extend(recent_data.iloc[-1][['open', 'high', 'low', 'volume']].tolist())  # Include other features

        # Predict the next day
        next_day_prediction = model.predict([input_features])[0]
        future_predictions.append(next_day_prediction)

        # Update recent_data for the next prediction
        next_day_row = recent_data.iloc[-1].copy()
        next_day_row['close'] = next_day_prediction
        recent_data = recent_data.append(next_day_row, ignore_index=True).iloc[-5:]

    return future_predictions


def feature_engineering(df):
    # Use past values as features - Lag Features
    for i in range(1, 6):  # Using last 5 days of prices as features
        df[f'lag_{i}'] = df['close'].shift(i)

    df.dropna(inplace=True)  # Drop rows with NaN values
    return df


def train_model(df):
    # Splitting the data - Time series split
    train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)

    # Define and train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(train_df.drop('close', axis=1), train_df['close'])

    # Evaluate the model
    predictions = model.predict(test_df.drop('close', axis=1))
    mse = mean_squared_error(test_df['close'], predictions)

    return model, mse, test_df, predictions


# Main Execution
symbol = 'AAPL'  # Example with Apple Inc.
df = fetch_stock_data(symbol)
df = preprocess_data(df)
df = feature_engineering(df)

model, mse, test_df, predictions = train_model(df)
print(f'Model MSE: {mse}')

# Predicting future values
future_days = 30  # Number of days you want to predict
recent_data = df['close'].iloc[-5:]  # Recent data to base predictions on
future_predictions = predict_future(model, recent_data, days=future_days)

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(range(len(df)), df['close'], label='Historical Data')
plt.plot(range(len(df), len(df) + future_days), future_predictions, label='Future Predictions', linestyle='--')
plt.title(f'{symbol} Stock Price Prediction')
plt.xlabel('Days')
plt.ylabel('Normalized Price')
plt.legend()
plt.show()