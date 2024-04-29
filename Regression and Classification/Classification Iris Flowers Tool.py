"""
What kind of flower do you prefer?

Consider the problem of classifying iris flowers into different species such as: setosa, versicolor and virginica.
For each flower, characteristics such as: sepal length and width, petal length and width are known.
Using this information, decide which species a particular flower belongs to.

A classification algorithm based on logistic regression to determine:
    - the species of a flower that has a sepal 5.35 cm long and 3.85 cm wide, and a petal 1.25 cm long and 0.4 cm wide
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def load_data(filename):
    data_read = pd.read_csv(filename, header=None)
    data_read = data_read.dropna()
    inputs = data_read[0]
    inputs = pd.concat([inputs, data_read[1]], axis=1)
    inputs = pd.concat([inputs, data_read[2]], axis=1)
    inputs = pd.concat([inputs, data_read[3]], axis=1)
    inputs = inputs.to_numpy()
    outputs = data_read[4]
    outputs_before = outputs
    outputs = pd.get_dummies(outputs)
    outputs = np.asarray(outputs)
    return inputs, outputs


def normalizeData(inputsToNormalize):
    mean, deviation = np.mean(inputsToNormalize, axis=0), np.std(inputsToNormalize, axis=0)
    inputsToNormalize = (inputsToNormalize - mean) / deviation
    return inputsToNormalize, mean, deviation


def get_tip_iris(predictions):
    try:
        indexes = np.argmax(predictions, axis=1)
        lista = []
        for index in indexes:
            if index == 0:
                lista.append('Iris-setosa')
            elif index == 1:
                lista.append('Iris-versicolor')
            else:
                lista.append('Iris-virginica')
        return lista
    except:
        if predictions[0] == 0:
            return 'Iris-setosa'
        elif predictions[1] == 1:
            return 'Iris-versicolor'
        else:
            return 'Iris-virginica'


def splitData(inputs, outputs):
    np.random.seed(5)
    indexes = [i for i in range(len(inputs))]
    trainSample = np.random.choice(indexes, int(0.8 * len(inputs)), replace=False)
    testSample = [i for i in indexes if not i in trainSample]

    trainInputs = [inputs[i] for i in trainSample]
    trainOutputs = [outputs[i] for i in trainSample]

    testInputs = [inputs[i] for i in testSample]
    testOutputs = [outputs[i] for i in testSample]
    return trainInputs, trainOutputs, testInputs, testOutputs


def getColor(outputs):
    colors = []
    for output in outputs:
        if output[0] == 1:
            colors.append('r')
        elif output[1] == 1:
            colors.append('g')
        else:
            colors.append('b')
    return colors


def trainModel(trainInputs, trainOutputs):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(3),
    ])
    model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    history = model.fit(trainInputs, trainOutputs, epochs=100, batch_size=16)
    return model, history


def plotDataHistogram(inputs, outputs):
    fig, axs = plt.subplots(2, 2, figsize=(15, 5))
    axs[0][0].scatter(inputs[:, 0], inputs[:, 1], c=getColor(outputs))
    axs[0][1].scatter(inputs[:, 2], inputs[:, 3], c=getColor(outputs))
    axs[1][0].scatter(inputs[:, 0], inputs[:, 2], c=getColor(outputs))
    axs[1][1].scatter(inputs[:, 1], inputs[:, 3], c=getColor(outputs))
    plt.legend(
        handles=[plt.Line2D([0], [0], marker='o', color='w', label='Iris-setosa', markerfacecolor='r', markersize=10),
                 plt.Line2D([0], [0], marker='o', color='w', label='Iris-versicolor', markerfacecolor='g',
                            markersize=10),
                 plt.Line2D([0], [0], marker='o', color='w', label='Iris-virginica', markerfacecolor='b',
                            markersize=10)])
    plt.show()


def learnCurve(history):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(history.history['loss'])
    axs[0].set_title('Loss Curve')
    axs[1].plot(history.history['accuracy'])
    axs[1].set_title('Accuracy Curve')
    plt.show()


def test(model, testInputs, testOutputs):
    loss, accuracy = model.evaluate(testInputs, testOutputs)
    print(f'Loss: {loss}, Accuracy: {accuracy}')


def predictionError(model, mean, deviation, testOutputs):
    initialTest = np.array([[5.35, 3.85, 1.25, 0.4]])
    normalizedTest = (initialTest - mean) / deviation

    predictions = model.predict(normalizedTest).squeeze()
    error = mean_squared_error(testOutputs, predictions)
    print('Prediction Error: ', error)


def main():
    inputs, outputs = load_data('data/iris.csv')
    inputs, mean, deviation = normalizeData(inputs)

    plotDataHistogram(inputs, outputs)

    trainInputs, trainOutputs, testInputs, testOutputs = splitData(inputs, outputs)

    model, history = trainModel(trainInputs, trainOutputs)

    learnCurve(history)

    test(model, testInputs, testOutputs)

    predictionError(model, mean, deviation, testOutputs)


main()
