import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models import MLP, LSTM
from dataset import TimeSeriesDataset
import pandas as pd
from trainers import BaseTrainer
import plotly.graph_objects as go
from evaluation import MeanAbsoluteError
import random
import numpy as np
from utils import MinMaxScaler

seed = 8989898
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

np.random.seed(seed)

random.seed(seed)

if __name__ == "__main__":

    # Import Data
    df = pd.read_csv('./data/data_daily.csv')
    df = df.rename(columns={'# Date': 'Date'})
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[['Receipt_Count', 'Date']]  

    # Hyperparameters
    train_size = int(len(df) * 0.8)
    predict_on_all_data = True
    model = 'MLP'

    sequence_length = 30
    hidden_size = 64
    output_size = 1
    lstm_num_layers = 1

    learning_rate = 0.0001
    if model=='MLP':
        learning_rate = 0.001

    batch_size = 16


    # Normalize the data
    scaler = MinMaxScaler()
    df[['Receipt_Count']] = scaler.fit_transform(df[['Receipt_Count']])

    # Split the data
    train_data, test_data = df.iloc[:train_size], df.iloc[train_size:]

    # DataLoader

    train_dataset_model = TimeSeriesDataset(train_data, sequence_length)
    train_loader_model = DataLoader(train_dataset_model, batch_size=batch_size, shuffle=True)

   
    if predict_on_all_data==True:

        test_dataset_model = TimeSeriesDataset(df, sequence_length)
        test_loader_model = DataLoader(test_dataset_model, batch_size=1, shuffle=False)
        test_data = df
    else:
        test_dataset_model = TimeSeriesDataset(test_data, sequence_length)
        test_loader_model = DataLoader(test_dataset_model, batch_size=1, shuffle=False)

    if model == 'MLP':
        model = MLP(input_size=sequence_length, hidden_size=hidden_size, output_size=output_size)
    else:
        model = LSTM(input_size=sequence_length, hidden_size=hidden_size, output_size=output_size, num_layers=lstm_num_layers)

    # Loss and optimizer
    criterion_model = nn.MSELoss()
    optimizer_model = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize and train the BaseTrainer
    trainer = BaseTrainer(model=model, criterion=criterion_model, optimizer=optimizer_model, train_loader=train_loader_model, test_loader=test_loader_model)
    trainer.train()

    # Evaluate the model
    test_targets, predictions = trainer.evaluate(test_data=test_data, scaler=scaler)

    # Calculate Mean Absolute Error
    mae = MeanAbsoluteError()
    mae_value = mae(test_targets, predictions)

    # Predict 365 days into the future
    future_predictions = trainer.predict_future(test_data=test_data, n_steps=365, scaler=scaler)

    # Plot the results for MLP using Plotly
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=test_data['Date'].iloc[sequence_length:], y=test_targets, mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=test_data['Date'].iloc[sequence_length:], y=predictions, mode='lines', name='Predicted (MLP)'))
    fig.add_trace(go.Scatter(x=pd.date_range(test_data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=365), y=future_predictions, mode='lines', name='Future Predictions'))

    fig.update_layout(xaxis_title='Date', yaxis_title='Receipt_Count', title=f'MLP Model Evaluation\nMean Absolute Error: {mae_value:.2f}')
    fig.show()
