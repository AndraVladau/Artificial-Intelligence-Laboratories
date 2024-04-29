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

Specify, design, implement and test a classification algorithm based on Artificial Neural Networks.
Check the influence of (hyper)parameters on the quality of the trained classifier.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import os
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import itertools
import random
from NeuralNetworks.ANN import NeuralNetwork


def loadImageData(data_dir, target_resolution=(128, 128)):
    images = []
    labels = []
    label_names = os.listdir(data_dir)
    for label, label_name in enumerate(label_names):
        label_dir = os.path.join(data_dir, label_name)
        for img_name in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, target_resolution)
                images.append(img)
                labels.append(label)  # 0 for normal and 1 for sepia
    return np.array(images), np.array(labels), label_names


def convert_outputs(labels, num_classes):
    """
    Convert integer labels to one-hot encoded vectors
    """
    converted = np.zeros((len(labels), num_classes))
    for i, label in enumerate(labels):
        converted[i, label] = 1
    return converted


def preprocessImages(images):
    # Flatten each image
    flattened_images = [img.flatten() for img in images]
    return np.array(flattened_images)


def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def train_ann(train_inputs, train_outputs, hidden_layer_sizes, activation, solver, alpha,
              learning_rate, max_iter):
    ann = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, activation=activation, solver=solver,
                        alpha=alpha, learning_rate=learning_rate)
    ann.fit(train_inputs, train_outputs)
    return ann


def evaluate_ann(ann, test_inputs, test_outputs, label_names):
    predicted_outputs = ann.predict(test_inputs)
    accuracy = accuracy_score(test_outputs, predicted_outputs)
    cm = confusion_matrix(test_outputs, predicted_outputs)
    plot_confusion_matrix(cm, label_names, title='Confusion Matrix')
    plt.show()
    return accuracy


def plot_sample_images(images, labels, label_names, num_samples=5):
    fig, axes = plt.subplots(1, num_samples,
                             figsize=(15, 3))  # Creăm o figură și un set de axe pentru a plasa imaginile

    for i in range(num_samples):
        index = random.randint(0, len(images) - 1)  # Alegem un index aleator
        img = images[index]
        label = labels[index]
        ax = axes[i]
        ax.imshow(img)  # Afișăm imaginea
        ax.set_title(label_names[label])  # Afișăm eticheta sub imagine
        ax.axis('off')  # Ascundem axele

    plt.show()  # Afișăm figură


def main(data_dir, hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.0001,
         learning_rate='constant', max_iter=500, batch_size='auto'):
    images, labels, label_names = loadImageData(data_dir)

    # Preprocess images
    processed_images = preprocessImages(images)

    # Split data into train and test sets
    train_inputs, test_inputs, train_outputs, test_outputs = train_test_split(processed_images, labels, test_size=0.2,
                                                                              random_state=42)

    scaler = StandardScaler()
    scaler.fit(train_inputs)
    train_inputs = scaler.transform(train_inputs)
    test_inputs = scaler.transform(test_inputs)

    # Train ANN
    ann = train_ann(train_inputs, train_outputs, hidden_layer_sizes, activation, solver, alpha,
                    learning_rate, max_iter)

    # Evaluate ANN
    accuracy = evaluate_ann(ann, test_inputs, test_outputs, label_names=["Normal", "Sepia"])
    print("Accuracy:", accuracy)


# print("Valori standard:")
# main("data")
# print("hidden_layer_sizes=(100,50):")
# main("data",hidden_layer_sizes=(100,50))
# print("activation=logistic:")
# main("data",activation="logistic")
# print("solver=sgd:")
# main("data",solver="sgd")
# print("alpha=0.001:")
# main("data",alpha=0.001)
# print("learning_rate=adaptive:")
# main("data",learning_rate="adaptive")
# print("max_iter=200:")
# main("data",max_iter=200)
#
# print("alpha=0.1:")
# main("data",alpha=0.1)
# main("data",alpha=0.1)
# main("data",alpha=0.1)
#
# print("alpha=0.001,hidden_layer_sizes=(50,50,50):")
# main("data",alpha=0.001,hidden_layer_sizes=(50,50,50))
# main("data",alpha=0.001,hidden_layer_sizes=(200,))
# main("data",alpha=0.001,hidden_layer_sizes=(100,100))
#
# print("alpha=0.001,activation=tanh:")
# main("data",alpha=0.001,activation="tanh")
# main("data",alpha=0.001,activation="tanh")
# main("data",alpha=0.001,activation="tanh")
#
# print("alpha=0.001,activation=tanh,max_iter=1000:")
# main("data",alpha=0.001,activation="tanh",max_iter=1000)
# main("data",alpha=0.001,activation="tanh",max_iter=1000)
# main("data",alpha=0.001,activation="tanh",max_iter=1000)


def main_ann(data_dir, hidden_size, epochs):
    # Load data
    inputs, outputs, output_names = loadImageData(data_dir)

    # Flatten images
    inputs = inputs.reshape(inputs.shape[0], -1)  # Flatten each image

    # Convert outputs to one-hot encoding
    n_classes = len(output_names)
    train_outputs = to_one_hot(outputs, n_classes)

    # Split data into train and test sets
    train_inputs, test_inputs, train_outputs, test_outputs = train_test_split(inputs, train_outputs)

    # Normalize inputs (optional)
    # train_inputs, test_inputs = normalize_data(train_inputs, test_inputs)

    # Create and train neural network
    input_size = train_inputs.shape[1]
    output_size = len(output_names)
    nn = NeuralNetwork(input_size, hidden_size, output_size)
    nn.train(train_inputs, train_outputs, epochs)

    # Evaluate neural network
    predicted_outputs = nn.predict(test_inputs)
    predicted_classes = np.argmax(predicted_outputs, axis=1)
    true_classes = np.argmax(test_outputs, axis=1)
    accuracy = accuracy_score(true_classes, predicted_classes)
    print("Accuracy:", accuracy)


def to_one_hot(labels, num_classes):
    """
    Convert integer labels to one-hot encoded vectors
    """
    one_hot = np.zeros((len(labels), num_classes))
    for i, label in enumerate(labels):
        one_hot[i, label] = 1
    return one_hot


main_ann("data", 100, 1000)
