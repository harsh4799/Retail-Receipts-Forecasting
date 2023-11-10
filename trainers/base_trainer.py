import torch
import numpy as np
class BaseTrainer:
    def __init__(self, model, criterion, optimizer, train_loader, test_loader, num_epochs=100):
        """
        Initialize the BaseTrainer.

        Parameters
        ----------
        model : torch.nn.Module
            The neural network model.
        criterion : torch.nn.Module
            The loss function.
        optimizer : torch.optim.Optimizer
            The optimizer.
        train_loader : torch.utils.data.DataLoader
            DataLoader for training data.
        test_loader : torch.utils.data.DataLoader
            DataLoader for training data.
        num_epochs : int, optional
            Number of training epochs, default is 100.
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs

    def train(self):
        """
        Train the model.
        """
        for epoch in range(self.num_epochs):
            for batch_x, batch_y in self.train_loader:
                # Forward pass
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {loss.item():.4f}')

    def evaluate(self,test_data,scaler=None):
        """
        Evaluate the model.
        """
        self.model.eval()

        predictions = []

        with torch.no_grad():
            for batch_x, batch_y in self.test_loader:
                # Reshape the input for the MLP
                batch_x = batch_x.view(-1, self.model.input_size)

                # Forward pass
                output = self.model(batch_x)
                predictions.append(output)
        print(len(predictions))
        # Reverse scaling for evaluation
        test_targets = test_data['Receipt_Count'].iloc[self.model.input_size:].values.reshape(-1, 1)
        predictions = np.array(predictions).reshape(-1, 1)
        if scaler != None:
            test_targets = scaler.inverse_transform(test_targets).flatten()
            predictions = scaler.inverse_transform(predictions).flatten()
        test_targets = test_targets.squeeze()
        predictions = predictions.squeeze()

        return test_targets, predictions