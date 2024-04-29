import numpy as np
from matplotlib import pyplot as plt


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.02):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.bias_input_hidden = np.zeros((1, self.hidden_size))
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_hidden_output = np.zeros((1, self.output_size))

    def sigmoid(self, x):
        clipped_x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-clipped_x))


    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward_propagation(self, X):
        # Input to hidden layer
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_input_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)

        # Hidden to output layer
        self.output = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_hidden_output
        self.predicted_output = self.sigmoid(self.output)

        return self.predicted_output

    def backward_propagation(self, X, y, predicted_output):
        # Calculate output layer error
        output_error = y - predicted_output
        output_delta = output_error * self.sigmoid_derivative(predicted_output)

        # Calculate hidden layer error
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)

        # Update weights and biases
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * self.learning_rate
        self.bias_hidden_output += np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate
        self.weights_input_hidden += X.T.dot(hidden_delta) * self.learning_rate
        self.bias_input_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * self.learning_rate

    def train(self, X, y, epochs):
        losses = []
        for epoch in range(epochs):
            # Forward propagation
            predicted_output = self.forward_propagation(X)

            # Backward propagation
            self.backward_propagation(X, y, predicted_output)

            # Calculate and print loss
            loss = np.mean(np.abs(y - predicted_output))
            losses.append(loss)

        plt.plot(losses)
        plt.title("Loss curve for ANN")
        plt.show()

    def predict(self, X):
        # Forward propagation for prediction
        return self.forward_propagation(X)
