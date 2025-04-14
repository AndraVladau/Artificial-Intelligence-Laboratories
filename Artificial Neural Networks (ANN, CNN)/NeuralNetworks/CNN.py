import numpy as np


class NeuralNetwork:
    def __init__(self):
        # Initialize parameters
        np.random.seed(1)
        self.parameters = {}
        self.parameters['W1'] = np.random.randn(3, 3, 3, 8) * 0.1
        self.parameters['b1'] = np.zeros((1, 1, 1, 8))
        self.parameters['W2'] = np.random.randn(3, 3, 8, 16) * 0.1
        self.parameters['b2'] = np.zeros((1, 1, 1, 16))
        self.parameters['W3'] = np.random.randn(8 * 8 * 16, 1) * 0.1
        self.parameters['b3'] = np.zeros((1, 1))

    def convolution_forward(self, A_prev, W, b, stride=1, padding=1):
        # Retrieve dimensions from A_prev shape
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

        # Retrieve dimensions from W shape
        (f, f, n_C_prev, n_C) = W.shape

        # Compute the dimensions of the CONV output volume
        n_H = int((n_H_prev - f + 2 * padding) / stride) + 1
        n_W = int((n_W_prev - f + 2 * padding) / stride) + 1

        # Initialize the output volume Z with zeros
        Z = np.zeros((m, n_H, n_W, n_C))

        # Create A_prev_pad by padding A_prev
        A_prev_pad = np.pad(A_prev, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode='constant',
                            constant_values=(0, 0))

        # Convolution operation
        for i in range(m):  # Loop over the batch of training examples
            a_prev_pad = A_prev_pad[i]  # Select ith training example's padded activation
            for h in range(n_H):  # Loop over vertical axis of the output volume
                for w in range(n_W):  # Loop over horizontal axis of the output volume
                    for c in range(n_C):  # Loop over channels (= #filters) of the output volume
                        # Find the corners of the current "slice"
                        vert_start = h * stride
                        vert_end = vert_start + f
                        horiz_start = w * stride
                        horiz_end = horiz_start + f

                        # Use the corners to define the (3D) slice of a_prev_pad
                        a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                        # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron
                        Z[i, h, w, c] = np.sum(a_slice_prev * W[:, :, :, c]) + b[:, :, :, c]

        # Making sure output shape is correct
        assert (Z.shape == (m, n_H, n_W, n_C))

        return Z

    def pooling_forward(self, A_prev, mode='max', f=2, stride=2):
        # Retrieve dimensions from the input shape
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

        # Define the dimensions of the output
        n_H = int(1 + (n_H_prev - f) / stride)
        n_W = int(1 + (n_W_prev - f) / stride)
        n_C = n_C_prev

        # Initialize output matrix A
        A = np.zeros((m, n_H, n_W, n_C))

        for i in range(m):  # Loop over the training examples
            for h in range(n_H):  # Loop on the vertical axis of the output volume
                for w in range(n_W):  # Loop on the horizontal axis of the output volume
                    for c in range(n_C):  # Loop over the channels of the output volume
                        # Find the corners of the current "slice"
                        vert_start = h * stride
                        vert_end = vert_start + f
                        horiz_start = w * stride
                        horiz_end = horiz_start + f

                        # Use the corners to define the current slice on the ith training example of A_prev, channel c.
                        a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]

                        # Compute the pooling operation on the slice.
                        if mode == 'max':
                            A[i, h, w, c] = np.max(a_prev_slice)
                        elif mode == 'average':
                            A[i, h, w, c] = np.mean(a_prev_slice)

        # Making sure output shape is correct
        assert (A.shape == (m, n_H, n_W, n_C))

        return A

    def flatten_forward(self, A):
        # Retrieve dimensions from the input shape
        (m, n_H, n_W, n_C) = A.shape

        # Reshape A to shape (m, n_H * n_W * n_C)
        A_flatten = A.reshape(m, -1)

        return A_flatten

    def forward_propagation(self, X):
        # Retrieve parameters
        W1 = self.parameters['W1']
        b1 = self.parameters['b1']
        W2 = self.parameters['W2']
        b2 = self.parameters['b2']
        W3 = self.parameters['W3']
        b3 = self.parameters['b3']

        # Convolutional Layer 1
        Z1 = self.convolution_forward(X, W1, b1, stride=1, padding=1)
        A1 = np.maximum(0, Z1)  # ReLU activation

        # Pooling Layer 1
        P1 = self.pooling_forward(A1, mode='max', f=2, stride=2)

        # Convolutional Layer 2
        Z2 = self.convolution_forward(P1, W2, b2, stride=1, padding=1)
        A2 = np.maximum(0, Z2)  # ReLU activation

        # Pooling Layer 2
        P2 = self.pooling_forward(A2, mode='max', f=2, stride=2)

        # Flatten Layer
        F = self.flatten_forward(P2)

        # Fully Connected Layer
        Z3 = np.dot(F, W3) + b3

        # Applying sigmoid activation to the final layer
        A3 = 1 / (1 + np.exp(-Z3))

        return A3

    def backward_propagation(self, X, Y, AL):
        # Retrieve parameters
        W1 = self.parameters['W1']
        b1 = self.parameters['b1']
        W2 = self.parameters['W2']
        b2 = self.parameters['b2']
        W3 = self.parameters['W3']
        b3 = self.parameters['b3']

        # Compute gradients
        dZ3 = AL - Y
        dW3 = 1 / m * np.dot(dZ3.T, self.flatten_forward(
            self.pooling_forward(np.maximum(0, self.convolution_forward(X, W2, b2, stride=1, padding=1)), mode='max',
                                 f=2, stride=2)).T)
        db3 = 1 / m * np.sum(dZ3, axis=0, keepdims=True)
        dZ2 = np.dot(dZ3, W3.T) * np.where(np.maximum(0, self.convolution_forward(X, W2, b2, stride=1, padding=1)) > 0,
                                           1, 0)
        dW2 = 1 / m * np.dot(dZ2.T, self.flatten_forward(
            self.pooling_forward(np.maximum(0, self.convolution_forward(X, W1, b1, stride=1, padding=1)), mode='max',
                                 f=2, stride=2)).T)
        db2 = 1 / m * np.sum(dZ2, axis=0, keepdims=True)
        dZ1 = np.dot(dZ2, W2.T) * np.where(np.maximum(0, self.convolution_forward(X, W1, b1, stride=1, padding=1)) > 0,
                                           1, 0)
        dW1 = 1 / m * np.dot(dZ1.T, X.reshape(m, -1))
        db1 = 1 / m * np.sum(dZ1, axis=0, keepdims=True)

        # Update parameters
        self.parameters['W1'] -= self.learning_rate * dW1
        self.parameters['b1'] -= self.learning_rate * db1
        self.parameters['W2'] -= self.learning_rate * dW2
        self.parameters['b2'] -= self.learning_rate * db2
        self.parameters['W3'] -= self.learning_rate * dW3
        self.parameters['b3'] -= self.learning_rate * db3

    def train(self, X, Y, epochs=100, learning_rate=0.01):
        self.learning_rate = learning_rate
        m = X.shape[0]
        for i in range(epochs):
            # Forward propagation
            AL = self.forward_propagation(X)

            # Compute cost
            cost = -1 / m * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))

            # Print the cost every 10 epochs
            if i % 10 == 0:
                print(f'Cost after epoch {i}: {cost}')

            # Backward propagation
            self.backward_propagation(X, Y, AL)