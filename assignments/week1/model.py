import numpy as np
from matplotlib import pyplot as plt


class LinearRegression:
    """
    A linear regression model that uses the closed form solution to fit the model.
    """

    w: np.ndarray
    b: float

    def __init__(self):
        self.w = 0
        self.b = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the weights for the given input using closed form solution.

        Arguments:
            X (np.ndarray): The input data NxF: number of examples x features
            y (np.ndarray): The output data.

        """
        # First append the bias to the features
        X_new = np.hstack((X, np.ones((X.shape[0], 1))))
        # make sure that y is 2D
        if np.linalg.det(X_new.T @ X_new) != 0:
            self.w = np.linalg.inv(X_new.T @ X_new) @ X_new.T @ y.reshape(-1, 1)
        else:
            print("LinAlgError. Matrix is Singular. No analytical solution.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output by forward pass the input to the system

        Arguments:
            X (np.ndarray): The input data NxF: number of examples x features

        """

        # First append the bias to the features
        X_new = np.hstack((X, np.ones((X.shape[0], 1))))
        return X_new @ self.w


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    w: np.ndarray
    b: float

    def __init__(self):
        self.w = 0
        self.b = 0

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 1e-8, epochs: int = 1000
    ) -> None:
        """
        Fit the weights for the given input using closed form solution.

        Arguments:
            X (np.ndarray): The input data.
            y (np.ndarray): The output data.

        """
        # First append the bias to the features
        X_new = np.hstack((X, np.ones((X.shape[0], 1))))
        # initialize the weights
        self.w = np.random.normal(0, 1, (X_new.shape[1], 1))
        losses = []
        for i in range(epochs):
            grad = 2 * ((self.w.T @ X_new.T) - y) @ X_new / X_new.shape[0]
            # grad =  (2 * (X_new@self.w-y) @ X_new).mean(axis=0).reshape(-1,1)
            # print('grad.shape',grad.shape, grad[:5])
            # print('X_new.shape',X_new.shape)
            # print('y.shape',y.shape)
            self.w -= lr * grad.T  # dL / dw = 2 * (w.T*x+b -y) * x
            # losses.append((((self.w.T @ X_new.T).T-y)**2).mean())
        # plt.plot(losses)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        X_new = np.hstack((X, np.ones((X.shape[0], 1))))
        return (self.w.T @ X_new.T).T  # + self.b
