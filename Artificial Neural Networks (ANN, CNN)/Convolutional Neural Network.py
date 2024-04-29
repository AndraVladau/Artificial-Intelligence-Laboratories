"""
Social Network
What kind of pictures have you posted?
You have just started your first day of work as a software developer at Facebook in the team that deals with
the content part of the platform.
The team of analysts noticed that a lot of people use filters over their photos, so in the hope of promoting less
edited content, and more real photos, they want to implement a new functionality to show users
whether a photo was or not edited. To test this idea, and to see if users would find such functionality useful,
they decided to test the idea on pictures that have sepia filters. Your first task is to implement
a photo classification algorithm that will tell us if a photo has a sepia filter added or not.
The team leader of the ML team proposes the following work plan:

    - developing, training and testing a classifier based on neural networks using simpler data,
      such as numerical characteristics - for example data with irises (demo)
    - developing, training and testing a classifier based on neural networks using more complex,
      image-type data - for example, a database with numbers, for each example considering the pixel matrix (demo)
    - creating a base with images (with and without sepia filter) and the corresponding labels
    - training and testing the classifier (based on artificial neural networks - tool or the developed ANN)
      for classifying images with and without filter

Specify, design, implement and test a classification algorithm based on Convolutional Neural Networks.
Check the influence of (hyper)parameters on the quality of the trained classifier.
"""

import numpy as np
import os
import cv2
from NeuralNetworks.CNN import NeuralNetwork


# Define the function to load the dataset
def load_dataset(directory):
    X = []
    y = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = cv2.imread(os.path.join(directory, filename))
            img = cv2.resize(img, (64, 64))  # Resize images to a fixed size
            X.append(img)
            if "sepia" in filename:
                y.append(1)  # Label 1 for images with sepia filter
            else:
                y.append(0)  # Label 0 for images without sepia filter
    return np.array(X), np.array(y)


# Assuming you have a single directory containing all images
data_directory = "data"

# Load the entire dataset
X, y = load_dataset(data_directory)

# Normalize pixel values to be between 0 and 1
X = X / 255.0

# Reshape labels for binary classification
y = y.reshape(-1, 1)

# Split the dataset into training and testing sets
test_size = 0.2
split_index = int(len(X) * (1 - test_size))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Shuffle the training dataset
shuffle_index = np.random.permutation(len(X_train))
X_train_shuffled = X_train[shuffle_index]
y_train_shuffled = y_train[shuffle_index]

# Example usage:
cnn = NeuralNetwork()
cnn.train(X_train_shuffled, y_train_shuffled, epochs=100, learning_rate=0.01)

# Evaluate the trained model on the test dataset
predictions = cnn.forward_propagation(X_test)
predictions_binary = np.where(predictions > 0.5, 1, 0)
accuracy = np.mean(predictions_binary == y_test)
print("Accuracy on test set:", accuracy)
