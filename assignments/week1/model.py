import numpy as np


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
        if np.linalg.det(X_new.T @ X_new) != 0:
            sol = np.linalg.inv(X_new.T @ X_new) @ X_new.T @ y.reshape(-1, 1)
            self.w = sol[:-1, :]
            self.b = sol[-1].item()
        else:
            print("LinAlgError. Matrix is Singular. No analytical solution.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output by forward pass the input to the system

        Arguments:
            X (np.ndarray): The input data NxF: number of examples x features

        """

        # First append the bias to the features
        return X @ self.w + self.b


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
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.0001, epochs: int = 1000
    ) -> None:
        """
        Fit the weights for the given input using closed form solution.

        Arguments:
            X (np.ndarray): The input data.
            y (np.ndarray): The output data.

        """
        # First append the bias to the features
        n_samples, n_features = X.shape
        # initialize the weights
        self.w = np.random.normal(0, 1, (n_features, 1))
        self.b = 0
        # losses = []
        for i in range(epochs):
            y_pred = X @ self.w + self.b
            err = y_pred - y
            grad_w = (2 * err @ X).mean(axis=0).reshape(-1, 1)
            grad_b = 2 * (y_pred.squeeze() - y.squeeze()).mean()
            self.w -= lr * grad_w
            self.b -= lr * grad_b
        #     losses.append(((y_pred.squeeze()-y.squeeze())**2).mean())
        # plt.plot(losses)
        # plt.show()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        X_new = np.hstack((X, np.ones((X.shape[0], 1))))
        return X @ self.w + self.b  # + self.b
