import boto3
import json
import pandas as pd
from datetime import datetime
from textblob import TextBlob
import requests

# Load configuration from file
with open('config.json') as config_file:
    config = json.load(config_file)

# Set up SNS
sns = boto3.client('sns', region_name=config['aws_region'], aws_access_key_id=config['aws_access_key_id'], aws_secret_access_key=config['aws_secret_access_key'])
topic_arn = 'arn:aws:sns:us-east-1:123456789012:news-sentiment-alerts'

def fetch_news_data():
    # Fetch news data from Bloomberg News API
    url = 'https://newsapi.bloomberg.com/bqfeed'
    params = {
        'search': config['symbol'],
        'limit': 100,
        'apikey': config['news_api_key']
    }
    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        raise ValueError(f"Error fetching news data: {response.content.decode('utf-8')}")

    data = response.json()
    articles = data['items']
    
    if not articles:
        raise ValueError(f"No news articles found for {config['symbol']}")
    
    # Create DataFrame from news data
    df = pd.DataFrame(articles, columns=['storyId', 'title', 'publishedAt', 'body'])
    df.rename(columns={'publishedAt': 'timestamp', 'body': 'article'}, inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    return df

def analyze_sentiment(news_data):
    sentiment_scores = []

    for index, row in news_data.iterrows():
        article = row['article']
        sentiment = TextBlob(article).sentiment.polarity
        sentiment_scores.append(sentiment)

    news_data['sentiment'] = sentiment_scores
    return news_data

def generate_sentiment_based_signals(news_sentiment_data, market_data):
    # Calculate the 10-period moving average of the close prices
    market_data['MA'] = market_data['close'].rolling(window=10).mean()

    # Merge the news sentiment data and market data on the timestamp column
    merged_data = pd.merge(news_sentiment_data, market_data, on='timestamp')

    # Generate signals based on the sentiment score and the market condition
    signals = []
    for index, row in merged_data.iterrows():
        sentiment_score = row['sentiment']
        ma = row['MA']
        close_price = row['close']
        signal = None

        # If the sentiment score is positive and the close price is above the moving average, buy signal
        if sentiment_score > 0 and close_price > ma:
            signal = 'buy'

        # If the sentiment score is negative and the close price is below the moving average, sell signal
        elif sentiment_score < 0 and close_price < ma:
            signal = 'sell'

        signals.append(signal)

    merged_data['signal'] = signals
    return merged_data


def news_sentiment_analysis_lambda_handler(event, context):
    try:
        news_data = fetch_news_data()
        news_sentiment_data = analyze_sentiment(news_data)
        generate_sentiment_based_signals(news_sentiment_data)

        message = f"News sentiment analysis executed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        sns.publish(TopicArn=topic_arn, Message=message)
        
        return {'statusCode': 200, 'body': 'News sentiment analysis executed'}
    except Exception as e:
        error_message = f"An error occurred during news sentiment analysis: {str(e)}"
        sns.publish(TopicArn=topic_arn, Message=error_message)
        print(error_message)
        raise e
