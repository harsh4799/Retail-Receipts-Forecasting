from torch.utils.data import Dataset
import torch
import numpy as np
class TimeSeriesDataset(Dataset):
    def __init__(self, data, sequence_length=1):
        self.data = data.values # Assumes Date, Values
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        x = np.float64(self.data[idx:idx + self.sequence_length, 0])  
        y = np.float64(self.data[idx + self.sequence_length, 0])     
        return torch.FloatTensor(x), torch.FloatTensor([y])