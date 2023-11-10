import streamlit as st
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from models import CustomLinearRegression, MLP, LSTM
from dataset import TimeSeriesDataset
from trainers import BaseTrainer
from evaluation import MeanAbsoluteError
import plotly.express as px
import plotly.graph_objects as go

# Set random seed for reproducibility
seed = 8989898
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import numpy as np
np.random.seed(seed)

# Streamlit app
def main():
    st.title("Time Series Prediction App")

    # Upload data
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        st.success("File successfully uploaded!")

        # Load data
        df = pd.read_csv(uploaded_file)
        # df = pd.read_csv('./data/data_daily.csv')
        df = df.rename(columns={'# Date': 'Date'})
        df['Date'] = pd.to_datetime(df['Date'])
        df = df[['Receipt_Count', 'Date']]  

        st.write("Preview of the uploaded data:")
        st.write(df.head())

        # Normalize the data
        scaler = MinMaxScaler()
        df[['Receipt_Count']] = scaler.fit_transform(df[['Receipt_Count']])

        # Split the data
        train_size = int(len(df) * 0.8)
        train_data, test_data = df.iloc[:train_size], df.iloc[train_size:]

        sequence_length = 30

        hidden_size = 64
        output_size = 1

        lstm_num_layers = 1

        batch_size = 16


        predict_on_all_data = True
        

        # Create DataLoader
        train_dataset = TimeSeriesDataset(train_data, sequence_length)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if predict_on_all_data==True:
            test_dataset_model = TimeSeriesDataset(df, sequence_length)
            test_loader_model = DataLoader(test_dataset_model, batch_size=1, shuffle=False)
            test_data = df
        else:
            test_dataset_model = TimeSeriesDataset(test_data, sequence_length)
            test_loader_model = DataLoader(test_dataset_model, batch_size=1, shuffle=False)

        # Model selection
        model_option = st.selectbox("Select a Model", ["Linear Regression", "MLP", "LSTM"])

        if model_option == "Linear Regression":
            model = CustomLinearRegression(input_size=sequence_length, output_size=output_size)
        elif model_option == "MLP":
            model = MLP(input_size=sequence_length, hidden_size=hidden_size, output_size=output_size)
        elif model_option == "LSTM":
            model = LSTM(input_size=sequence_length, hidden_size=hidden_size, output_size=output_size, num_layers=lstm_num_layers)

        # Loss and optimizer
        learning_rate = 0.0001 if model_option == 'LSTM' else 0.001
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Create BaseTrainer instance
        trainer = BaseTrainer(model, torch.nn.MSELoss(), optimizer, train_loader, test_loader_model)  # You can provide test_loader here if needed

        # Train the model
        st.subheader("Training the Model")
        trainer.train()  # Adjust num_epochs as needed

        # Evaluation
        st.subheader("Model Evaluation")
        test_targets, predictions = trainer.evaluate(test_data=test_data, scaler=scaler)


        future_predictions = trainer.predict_future(test_data=test_data, n_steps=365, scaler=scaler)
        # Plot results using Plotly

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=test_data['Date'].iloc[sequence_length:], y=test_targets, mode='lines', name='Actual'))
        fig.add_trace(go.Scatter(x=test_data['Date'].iloc[sequence_length:], y=predictions, mode='lines', name='Predicted (MLP)'))
        fig.add_trace(go.Scatter(x=pd.date_range(test_data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=365), y=future_predictions, mode='lines', name='Future Predictions'))


        st.plotly_chart(fig)

        # Calculate Mean Absolute Error
        mae_calculator = MeanAbsoluteError()
        mae_value = mae_calculator(test_targets, predictions)
        st.write(f"Mean Absolute Error: {mae_value:.4f}")
        

if __name__ == "__main__":
    main()
