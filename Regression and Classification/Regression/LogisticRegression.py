import numpy as np
from matplotlib import pyplot as plt


class MyLogisticRegressionClassifier:
    def __init__(self):
        self.learning_rate = 0.01
        self.num_iterations = 1000
        self.coef_ = None
        self.intercept_ = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.coef_ = np.zeros(num_features)
        self.intercept_ = 0
        error = []
        # Gradient descent
        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.coef_) + self.intercept_
            predictions = self.sigmoid(linear_model)

            # Compute gradients
            dw = (1 / num_samples) * np.dot(X.T, (predictions - y))
            db = (1 / num_samples) * np.sum(predictions - y)

            # Update parameters
            self.coef_ -= self.learning_rate * dw
            self.intercept_ -= self.learning_rate * db

            for t1, t2 in zip(predictions, y):
                error.append((t1 - t2) ** 2)

        plt.plot(error)
        plt.show()

    def predict(self, X):
        linear_model = np.dot(X, self.coef_) + self.intercept_
        predictions = self.sigmoid(linear_model)
        return [1 if p >= 0.5 else 0 for p in predictions]


