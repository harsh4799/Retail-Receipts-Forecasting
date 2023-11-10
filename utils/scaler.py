import numpy as np
class MinMaxScaler:
    def __init__(self):
        self.min_values = None
        self.max_values = None

    def fit(self, data):
        self.min_values = np.min(data, axis=0)
        self.max_values = np.max(data, axis=0)

    def transform(self, data):
        scaled_data = (data - self.min_values) / (self.max_values - self.min_values)
        return scaled_data

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, scaled_data):
        min_values = np.full_like(scaled_data, self.min_values)
        max_values = np.full_like(scaled_data, self.max_values)
        original_data = scaled_data * (max_values - min_values) + min_values

        return original_data