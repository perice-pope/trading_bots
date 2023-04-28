# Trading Bot Scalping

This repository contains a trading bot that uses machine learning and reinforcement learning techniques to execute trades in a scalping strategy. The bot is designed to adapt to market conditions by periodically updating its data, model, and strategies. The bot utilizes AWS services such as Lambda, S3, and SNS for data storage, notifications, and managing the overall process.

## AWS Services

- **Lambda**: Used to run the four main functions of the trading bot (data retrieval, training and deployment, trade execution, monitoring and optimization) as serverless functions.
- **S3**: Utilized for storing market data, trained models, and other relevant files. The bot reads and writes to the specified S3 bucket.
- **SNS**: Employed for sending notifications about the bot's performance metrics, optimization results, and any errors that occur during the process.

## Lambda Functions

The trading bot consists of four AWS Lambda functions, each serving a specific purpose in the overall process:

1. **Data Retrieval (`data_retrieval_lambda_handler`)**: Responsible for retrieving market data and storing it in a CSV file in the S3 bucket. This function is triggered periodically (e.g., every minute) to ensure the trading bot has the latest market data.

2. **Training and Deployment (`training_and_deployment_lambda_handler`)**: Responsible for training the Keras model, Bayesian optimization, and deploying the updated model. It trains the Keras model using the latest market data retrieved by the `data_retrieval_lambda_handler`. It uses Bayesian optimization to find optimal parameters and strategies for the trading bot. After training and optimization, the updated model is saved to the S3 bucket for use by the trading bot. This function can be triggered periodically (e.g., every day) to ensure the model is up-to-date with the latest market data and optimized parameters.

3. **Trade Execution (`execution_lambda_handler`)**: Responsible for executing trades based on the current market state and the trained model's recommendations. It retrieves the current market state and uses the trained model stored in the S3 bucket to predict optimal actions (e.g., position size, stop loss, take profit). Based on the predicted actions, the function executes trades on the desired platform. This function can be triggered frequently (e.g., every minute) to ensure the trading bot is responsive to market changes.

4. **Monitoring and Optimization (`monitoring_and_optimization_lambda_handler`)**: Responsible for monitoring the trading bot's performance and making necessary adjustments. It calculates performance metrics (e.g., returns, win rate, risk-reward ratio, max drawdown, return on capital) for the past week. It retrains the model using online learning algorithms and new market data. It adjusts the trading bot's parameters and strategies using reinforcement learning. Notifications about the bot's performance, optimization results, and any errors are sent using SNS. This function can be triggered periodically (e.g., every week) to ensure the trading bot is performing well and making adjustments as needed.

## Overall Process

1. **Data retrieval**: The trading bot starts by retrieving the latest market data using the `data_retrieval_lambda_handler`.
2. **Training and deployment**: The `training_and_deployment_lambda_handler` trains the model, optimizes parameters, and saves the updated model to the S3 bucket.
3. **Trade execution**: The `execution_lambda_handler` executes trades based on the model's recommendations and the current market state.
4. **Monitoring and optimization**: The `monitoring_and_optimization_lambda_handler` periodically monitors the bot's performance, makes adjustments to improve its performance, and sends notifications using SNS.

The trading bot operates in a continuous loop, periodically updating its data, model, and strategies to adapt to the ever-changing market conditions. By leveraging AWS services such as Lambda, S3, and SNS, the bot is able to efficiently manage its resources, send notifications, and store necessary information. This ensures the bot remains responsive to market changes and consistently seeks to improve its performance.
