import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from models import MLP
from dataset import TimeSeriesDataset
import pandas as pd
from trainers import BaseTrainer
import plotly.graph_objects as go
from evaluation import MeanAbsoluteError

if __name__ == "__main__":
    df = pd.read_csv('./data/data_daily.csv')
    df = df.rename(columns={'# Date': 'Date'})
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[['Receipt_Count', 'Date']]  

    # Normalize the data
    scaler = MinMaxScaler()
    df[['Receipt_Count']] = scaler.fit_transform(df[['Receipt_Count']])

    # Split the data
    train_size = int(len(df) * 0.8)
    train_data, test_data = df.iloc[:train_size], df.iloc[train_size:]
    sequence_length = 10  
    hidden_size = 64
    output_size = 1
    learning_rate = 0.001
    batch_size = 16

    # DataLoader
    train_dataset_mlp = TimeSeriesDataset(train_data, sequence_length)
    train_loader_mlp = DataLoader(train_dataset_mlp, batch_size=batch_size, shuffle=True)
    test_dataset_mlp = TimeSeriesDataset(test_data, sequence_length)
    test_loader_mlp = DataLoader(test_dataset_mlp, batch_size=1, shuffle=False)
    model_mlp = MLP(input_size=sequence_length, hidden_size=hidden_size, output_size=output_size)

    # Loss and optimizer
    criterion_mlp = nn.MSELoss()
    optimizer_mlp = torch.optim.Adam(model_mlp.parameters(), lr=learning_rate)

    # Initialize and train the BaseTrainer
    trainer = BaseTrainer(model=model_mlp, criterion=criterion_mlp, optimizer=optimizer_mlp, train_loader=train_loader_mlp, test_loader=test_loader_mlp)
    trainer.train()

    # Evaluate the model
    test_targets, predictions = trainer.evaluate(test_data=test_data, scaler=scaler)


    # Calculate Mean Absolute Error
    mae = MeanAbsoluteError()
    mae_value = mae(test_targets, predictions)

    # Plot the results for MLP using Plotly
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=test_data['Date'].iloc[sequence_length:], y=test_targets, mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=test_data['Date'].iloc[sequence_length:], y=predictions, mode='lines', name='Predicted (MLP)'))

    fig.update_layout(xaxis_title='Date', yaxis_title='Receipt_Count', title=f'MLP Model Evaluation\nMean Absolute Error: {mae_value:.2f}')
    fig.show()
