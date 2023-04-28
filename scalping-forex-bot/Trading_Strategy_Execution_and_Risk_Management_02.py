import os
import json
import boto3
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
from sklearn.externals import joblib
from my_broker_api import BrokerAPI
from tensorflow import lite
import talib as ta

# Load configuration from file
with open('config.json') as config_file:
    config = json.load(config_file)

# Load the best model from S3 and convert it to TensorFlow Lite
def load_best_model():
    bucket_name = config['s3_bucket_name']
    key = 'path/to/best_model.h5'
    local_path = '/tmp/best_model.h5'

    s3 = boto3.client('s3', region_name=config['aws_region'], aws_access_key_id=config['aws_access_key_id'], aws_secret_access_key=config['aws_secret_access_key'])
    s3.download_file(bucket_name, key, local_path)

    # Load the Keras model
    keras_model = tf.keras.models.load_model(local_path)

    # Convert the Keras model to TFLite format
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_model = converter.convert()

    # Load the TFLite model using the Interpreter
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    return interpreter

model = load_best_model()

# Set up AWS services
# Set up Kinesis Data Stream
kinesis = boto3.client('kinesis', region_name=config['aws_region'], aws_access_key_id=config['aws_access_key_id'], aws_secret_access_key=config['aws_secret_access_key'])
stream_name = 'market-data-stream'
shard_id = 'shardId-000000000000'
shard_iterator_type = 'LATEST'
shard_iterator = kinesis.get_shard_iterator(StreamName=stream_name, ShardId=shard_id, ShardIteratorType=shard_iterator_type)['ShardIterator']

# Set up SNS
sns = boto3.client('sns', region_name=config['aws_region'], aws_access_key_id=config['aws_access_key_id'], aws_secret_access_key=config['aws_secret_access_key'])
topic_arn = 'arn:aws:sns:us-east-1:123456789012:market-data-alerts'

# Load the scaler from S3
bucket_name = config['s3_bucket_name']
key = 'path/to/scaler.pkl'
local_path = '/tmp/scaler.pkl'
s3 = boto3.client('s3', region_name=config['aws_region'], aws_access_key_id=config['aws_access_key_id'], aws_secret_access_key=config['aws_secret_access_key'])
s3.download_file(bucket_name, key, local_path)
scaler = joblib.load(local_path)

# Preprocess market data for prediction
def preprocess_live_data(data):
    try:
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        if len(df.columns) != 6:
            raise ValueError(f"Input data has {len(df.columns)} columns, but expected 6 columns")
        df.set_index('timestamp', inplace=True)
        df['SMA'] = ta.SMA(df['close'])
        df['RSI'] = ta.RSI(df['close'])
        df['BB_UPPER'], df['BB_MIDDLE'], df['BB_LOWER'] = ta.BBANDS(df['close'])
        df.dropna(inplace=True)
        df = df.tail(config['lookback'])
        df_scaled = scaler.transform(df)
        return np.array([df_scaled])
    except (ValueError, KeyError) as e:
        print(f"Error preprocessing live data: {e}")
        return None


# Execute trades
def execute_trades(predictions, risk_reward_ratio, max_trades):
    broker_api = BrokerAPI(config['broker_api_key'], config['broker_api_secret'], config['broker_api_endpoint'])
    symbol = config['symbol']
    stop_loss = config['stop_loss']
    take_profit = stop_loss * risk_reward_ratio

    # Set up S3
    s3 = boto3.client('s3', region_name=config['aws_region'], aws_access_key_id=config['aws_access_key_id'], aws_secret_access_key=config['aws_secret_access_key'])
    bucket_name = config['s3_bucket_name']

    # Get the latest price and cache it
    latest_price_key = f"{symbol}-latest-price"
    latest_price = s3.get_object(Bucket=bucket_name, Key=latest_price_key)['Body'].read().decode('utf-8')
    if latest_price == '':
        latest_price = broker_api.get_last_price(symbol)
        s3.put_object(Bucket=bucket_name, Key=latest_price_key, Body=str(latest_price))
    else:
        latest_price = float(latest_price)

    # Cache market data for future use
    market_data = json.dumps(data)
    s3.put_object(Bucket=bucket_name, Key=symbol, Body=market_data)

    # Cache the latest price
    s3.put_object(Bucket=bucket_name, Key=latest_price_key, Body=str(broker_api.get_last_price(symbol)))

    # Check if there are any open positions for the symbol
    open_positions = broker_api.get_open_positions(symbol)
    if not open_positions:
        message = f"No open positions for {symbol}"
        sns.publish(TopicArn=topic_arn, Message=message)
        return 0
    
    # Open a new position if the predicted price change is positive
    if predictions[0][0] > 0 and len(open_positions) < max_trades:
        position_size = 0.01 # You may need to adjust the position size based on your account balance and risk tolerance
        market_price = latest_price
        stop_loss_price = market_price - stop_loss
        take_profit_price = market_price + take_profit
        broker_api.open_position(symbol, position_size, 'buy', stop_loss_price, take_profit_price)
    
    # Close existing positions if the predicted price change is negative
    elif predictions[0][0] < 0:
        for position in open_positions:
            market_price = latest_price
            stop_loss_price = market_price + stop_loss
            take_profit_price = market_price - take_profit
            broker_api.close_position(position['id'], stop_loss_price, take_profit_price)

    # Return the number of open positions
    return len(open_positions)

# Define the main Lambda function handler
def lambda_handler(event, context):
    try:
        # Get the latest market data from the Kinesis Data Stream
        shard_iterator = kinesis.get_shard_iterator(StreamName=stream_name, ShardId=shard_id, ShardIteratorType=shard_iterator_type)['ShardIterator']
        records_response = None
        retry_count = 0
        while not records_response and retry_count < config['max_retries']:
            records_response = kinesis.get_records(ShardIterator=shard_iterator, Limit=config['batch_size'])
            if not records_response['Records']:
                time.sleep(config['retry_delay'])
                retry_count += 1
        if not records_response:
            raise Exception("No data available to process")

        market_data = pd.concat([pd.read_json(record['Data']) for record in records_response['Records']], ignore_index=True)

        # Preprocess the market data using the preprocess_live_data() function
        live_data = preprocess_live_data(market_data)

        # Use the TensorFlow Lite interpreter for prediction
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        model.set_tensor(input_details[0]['index'], live_data.astype(np.float32))
        model.invoke()
        predictions = model.get_tensor(output_details[0]['index'])

        # Execute trades based on the predictions using the execute_trades() function
        execute_trades(predictions)

        # Publish a message to the SNS topic to notify the user of the executed trades
        message = f"Scalping strategy executed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        sns.publish(TopicArn=topic_arn, Message=message)

        # Return a success message
        return {'statusCode': 200, 'body': 'Scalping strategy executed'}
    except Exception as e:
        # Handle any exceptions that occur during the preprocessing or prediction steps
        error_message = f"An error occurred during strategy execution: {e}"
        sns.publish(TopicArn=topic_arn, Message=error_message)
        raise e
