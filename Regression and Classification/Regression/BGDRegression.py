import numpy as np
import matplotlib.pyplot as plt


class MyBGDRegression:
    # learning_rate: cât de mult modificăm parametrii modelului în fiecare pas de antrenare
    # num_iterations: numărul de iterații (epoci)
    # batch_size: câte exemple de antrenament sunt folosite pentru a actualiza parametrii modelului în fiecare pas.

    def __init__(self, learning_rate=0.0001, num_iterations=10000, batch_size=32):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X = np.array(X)
        num_samples, num_features = X.shape
        self.coef_ = np.zeros(num_features)
        self.intercept_ = 0

        y = np.array(y).flatten()  # Convert y to 1D array
        error = []
        for _ in range(self.num_iterations):
            # Shuffle the data
            permutation = np.random.permutation(num_samples)
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]

            for i in range(0, num_samples, self.batch_size):
                # Get batch
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]
                # print(X_batch)

                # Compute gradients
                gradient_weights, gradient_bias = self.compute_gradients(X_batch, y_batch)

                # Update weights and bias
                self.coef_ -= self.learning_rate * gradient_weights
                self.intercept_ -= self.learning_rate * gradient_bias

            computedTestOutputs = self.predict([[x] for x in X_shuffled])
            # print(computedTestOutputs)

            for t1, t2 in zip(computedTestOutputs, y):
                error.append((t1 - t2) ** 2)

        plt.plot(error)
        plt.show()

    def compute_gradients(self, X_batch, y_batch):
        num_batch_samples = X_batch.shape[0]

        # np.dot = produs scalar intre 2 matrici
        y_predicted = np.dot(X_batch, self.coef_) + self.intercept_

        # Compute gradients
        gradient_weights = -(2 / num_batch_samples) * np.dot(X_batch.T, (y_batch - y_predicted))
        gradient_bias = -(2 / num_batch_samples) * np.sum(y_batch - y_predicted)

        return gradient_weights, gradient_bias

    def predict(self, X):
        return np.dot(X, self.coef_) + self.intercept_
