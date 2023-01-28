import numpy as np


class LinearRegression:

    w: np.ndarray
    b: float

    def __init__(self):
        self.w = 0
        self.b = 0


    def fit(self, X, y):
        if np.linalg.det(X.T@X) != 0:
            self.w = np.linalg.inv(X.T@X) @X.T@y
        else:
            print("LinAlgError. Matrix is Singular. No analytical solution.")

    def predict(self, X):
        return self.w @ X + self.b


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:
        for i in range(epochs):
            grad =  2 * (self.w.T @ x+self.b -y) * x
            self.w -= lr * grad # dL / dw = 2 * (w.T*x+b -y) * x
 
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        return self.w @ X + self.b
#         raise NotImplementedError()

