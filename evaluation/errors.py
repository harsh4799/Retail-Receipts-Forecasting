import numpy as np
class MeanAbsoluteError:
    def __init__(self):
        self.errors = None

    def __call__(self, actual, predicted):
        """
        Calculate Mean Absolute Error.

        Parameters
        ----------
        actual : numpy array or list
            The actual values.
        predicted : numpy array or list
            The predicted values.

        Returns
        -------
        float
            Mean Absolute Error.
        """
        if len(actual) != len(predicted):
            raise ValueError("Input arrays must have the same length.")

        self.errors = np.abs(np.subtract(actual, predicted))
        mae = np.mean(self.errors)
        return mae


if __name__ == "__main__":
    mae = MeanAbsoluteError()

    actual_values = np.array([5, 8, 12, 15, 20])
    predicted_values = np.array([6, 9, 11, 14, 18])

    mae_value = mae(actual_values, predicted_values)

    print(f"Mean Absolute Error: {mae_value}")