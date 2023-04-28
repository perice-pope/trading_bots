import os
import requests
import pandas as pd
import numpy as np
import talib as ta
import json
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import EarlyStopping
from keras.regularizers import l1_l2
from keras.activations import relu
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
import tempfile
import boto3
from sklearn.metrics import mean_absolute_error

# Load configuration from file
with open('config.json') as config_file:
    config = json.load(config_file)

# Data source object for fetching historical data
class DataSource:
    def fetch_historical_data(self, symbol):
        pass

class TDAmeritradeDataSource(DataSource):
    def fetch_historical_data(self, symbol):
        try:
            endpoint = f'https://api.tdameritrade.com/v1/marketdata/{symbol}/pricehistory'
            params = {
                'apikey': config['api_key'],
                'periodType': 'day',
                'frequencyType': 'minute',
                'frequency': 1
            }
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            if 'candles' in data:
                return data['candles']
            else:
                return None
        except requests.exceptions.RequestException as e:
            # Handle network connectivity issues or other errors
            print(f"Error fetching historical data: {e}")
            return None

def preprocess_data(data):
    df = pd.DataFrame(data)
    df['SMA'] = ta.SMA(df['close'])
    df['RSI'] = ta.RSI(df['close'])
    df['BB_UPPER'], df['BB_MIDDLE'], df['BB_LOWER'] = ta.BBANDS(df['close'])
    df.dropna(inplace=True)
    return df

def feature_scaling(df):
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)
    return pd.DataFrame(df_scaled, columns=df.columns, index=df.index)

def create_dataset(df, lookback=config['lookback']):
    X, y = [], []
    for i in range(lookback, len(df)):
        X.append(df.iloc[(i - lookback):i, :-1].values)
        y.append(df.iloc[i, -1])
    return np.array(X), np.array(y)

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape, kernel_regularizer=l1_l2(l1=0.01, l2=0.01), activation=relu))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True, kernel_regularizer=l1_l2(l1=0.01, l2=0.01), activation=relu))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, kernel_regularizer=l1_l2(l1=0.01, l2=0.01), activation=relu))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(X_train, y_train, input_shape):
    model = build_model(input_shape)

    # Early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(X_train, y_train, epochs=100, batch_size=32, callbacks=[early_stopping], validation_split=0.1)

    return model

def lambda_handler(event, context):
    symbol = config['symbol']
    data_source = TDAmeritradeDataSource()
    data = data_source.fetch_historical_data(symbol)
    df = preprocess_data(data['candles'])
    df = feature_scaling(df)
    lookback = config['lookback']
    min_mae = float('inf')

    # Split data into time-series cross-validation using TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=config['time_series_splits'])

    ensemble_models = []

    for i, (train_index, test_index) in enumerate(tscv.split(df)):
        train_df = df.iloc[train_index, :]
        test_df = df.iloc[test_index, :]

        X_train, y_train = create_dataset(train_df, lookback)
        X_test, y_test = create_dataset(test_df, lookback)

        # Train the model
        model = train_model(X_train, y_train, (X_train.shape[1], X_train.shape[2]))

        # Evaluate the model
        mae = mean_absolute_error(y_test, model.predict(X_test))

        # Save and upload the model if the mean absolute error is lower than the previous minimum
        if mae < min_mae:
            min_mae = mae

            # Save the model to a temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            model.save(temp_file.name)

            # Upload the file to S3
            bucket_name = config['s3_bucket_name']
            key = f'path/to/model_{i}.h5'
            s3 = boto3.client('s3')
            s3.upload_file(temp_file.name, bucket_name, key)


            # Delete the temporary file
            temp_file.close()
            os.unlink(temp_file.name)

        ensemble_models.append(model)

    # Ensemble averaging
    predictions = [model.predict(X_test) for model in ensemble_models]
    ensemble_prediction = np.mean(predictions, axis=0)

    # Return a success message
    return {'statusCode': 200, 'body': 'Best model saved to S3'}
