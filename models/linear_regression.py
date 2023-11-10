import numpy as np
class CustomLinearRegression:
    def __init__(self):
        """
        Initialize a CustomLinearRegression object.

        Attributes
        ----------
        coefficients : numpy array
            Coefficients of the linear regression model.
        """
        self.coefficients = None

    def fit(self, X, y):
        """
        Fit the linear regression model using the method of least squares.

        Parameters
        ----------
        X : numpy array
            Input features.
        y : numpy array
            Target variable.
        """
        # Intercept term
        ones_column = np.ones(X.shape[0]).reshape(-1, 1)
        X = np.concatenate((ones_column, X), axis=1)

        # Coefficients using the least squares formula
        self.coefficients = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, X):
        """
        Make predictions using the trained linear regression model.

        Parameters
        ----------
        X : numpy array
            Input features for prediction.

        Returns
        -------
        numpy array
            Predicted values.
        """
        # Intercept Term
        ones_column = np.ones(X.shape[0]).reshape(-1, 1)
        X = np.concatenate((ones_column, X), axis=1)

        predictions = X @ self.coefficients
        return predictions

if __name__ == '__main__': 
    # Testing our model out
    regression_model = CustomLinearRegression()
    X_train = np.array([[1], [2], [3]])
    y_train = np.array([2, 4, 5])
    regression_model.fit(X_train, y_train)
    X_test = np.array([[4], [5]])
    predictions = regression_model.predict(X_test)
    print(predictions)
    
    # Comparing this to Numpy's implmentation (np.polyfit)
    degree = 1  
    coefficients_test = np.polyfit(X_train.flatten(), y_train, deg=degree)
    predictions_test = np.polyval(coefficients_test, X_test.flatten())
    print(predictions_test)
