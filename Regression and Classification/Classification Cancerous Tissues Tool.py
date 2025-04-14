"""
Classification of cancerous tissues

We consider information about breast cancer in women, information extracted from breast ultrasounds such as:

- The type of malformation identified (benign tissue or malignant tissue)
- Numerical characteristics of the nucleus of the cells in these tissues:
- the radius (average of the distances between the center and the points from the contour)
- the texture (measured by the standard deviation of the gray levels in the image associated with the analyzed tissue)

Using these data, to decide whether the tissue from a new ultrasound (for which the 2 numerical characteristics are known
- radius and texture –) will be labeled as malignant or benign.

A classification algorithm based on logistic regression to determine:
    - if a lesion characterized by a texture of value 10 and a radius of value 18 is a malignant or benign lesion

"""

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def loadData(filename):
    data_read = pd.read_csv(filename, header=None)
    unusedColumns = [0] + [x for x in range(4, 32)]
    data_read.drop(data_read.columns[unusedColumns], axis=1, inplace=True)
    inputs = data_read[2]
    inputs = pd.concat([inputs, data_read[3]], axis=1)
    inputs = inputs.to_numpy()
    outputs = data_read[1]
    outputs = list(outputs)
    outputs = [0 if x == 'M' else 1 for x in outputs]
    outputs = np.asarray(outputs)
    outputs = outputs.reshape(len(outputs), 1)
    return inputs, outputs


def normalizeData(inputToNormalize):
    mean, deviation = np.mean(inputToNormalize, axis=0), np.std(inputToNormalize, axis=0)
    inputToNormalize = (inputToNormalize - mean) / deviation
    return inputToNormalize, mean, deviation


def plotDataHistogram(inputs, outputs):
    plt.scatter(inputs[:, 0], inputs[:, 1], c=outputs)
    plt.show()


def splitTheData(inputs, outputs):
    np.random.seed(5)
    indexes = [i for i in range(len(inputs))]
    trainSample = np.random.choice(indexes, int(0.8 * len(inputs)), replace=False)
    testSample = [i for i in indexes if not i in trainSample]

    trainInputs = [inputs[i] for i in trainSample]
    trainOutputs = [outputs[i] for i in trainSample]

    testInputs = [inputs[i] for i in testSample]
    testOutputs = [outputs[i] for i in testSample]
    return trainInputs, trainOutputs, testInputs, testOutputs


def learnedCurve(history):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(history.history['loss'])
    axs[0].set_title('Loss Curve')
    axs[1].plot(history.history['accuracy'])
    axs[1].set_title('Accuracy Curve')
    plt.show()


def trainModel(trainInputs, trainOutputs):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(trainInputs, trainOutputs, epochs=50, batch_size=8)
    return model, history


def plotLearnedModel(X, y, model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    Z = Z > 0.5

    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, marker='o', edgecolors='black')
    plt.xlabel('Radius')
    plt.ylabel('Texture')
    plt.title('Decision Boundary for Model')
    plt.show()


def testData(model, testInputs, testOutputs):
    loss, accuracy = model.evaluate(testInputs, testOutputs)
    print(f'Loss: {loss}, Accuracy: {accuracy}')


def predictionError(model, mean, deviation, testOutputs):
    initialTest = np.array([[18, 10]])
    normalizedTest = (initialTest - mean) / deviation
    predictions = model.predict(normalizedTest).squeeze()

    error = mean_squared_error(testOutputs, predictions)
    print('Prediction Error: ', error)


def main():
    inputs, outputs = loadData('data/breast-cancer-wisconsin-diagnostic.csv')
    inputs, mean, deviation = normalizeData(inputs)

    plotDataHistogram(inputs, outputs)
    trainInputs, trainOutputs, testInputs, testOutputs = splitTheData(inputs, outputs)

    model, history = trainModel(trainInputs, trainOutputs)

    learnedCurve(history)

    testData(model, testInputs, testOutputs)

    plotLearnedModel(testInputs, testOutputs, model)

    predictionError(model, mean, deviation, testOutputs)


main()
