"""
What can make people happy?
It considers the problem of predicting the degree of happiness of the population of the globe using information about
various characteristics of the well-being of the respective population, such as the gross domestic product of the country
in which they live (GBP), the degree of happiness, etc.

Using the data related to the year 2017, make a prediction of the degree of happiness according to only by "Family".
"""

import csv
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error


def loadData1(fileName, inputName, outputName):
    data = []
    dataNames = []
    with open(fileName) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                dataNames = row
            else:
                data.append(row)
            line_count += 1
    selectedVariable = dataNames.index(inputName)
    inputs = [float(data[i][selectedVariable]) for i in range(len(data))]
    selectedOutput = dataNames.index(outputName)
    outputs = [float(data[i][selectedOutput]) for i in range(len(data))]

    return inputs, outputs


def plotDataHistogram(x, variableName):
    n, bins, patches = plt.hist(x, 10, color='lightblue', edgecolor='black')
    plt.title('Histogram of ' + variableName)
    plt.show()


def plotSpecificHistogram(inputs, outputs, label):
    plt.plot(inputs, outputs, 'ro')
    plt.xlabel('GDP capita')
    plt.ylabel('happiness')
    plt.title('GDP capita vs. happiness')
    plt.show()


def trainAndValidationSample(inputs, outputs, label):
    np.random.seed(155)
    indexes = [i for i in range(len(inputs))]
    trainSample = np.random.choice(indexes, int(0.8 * len(inputs)), replace=False)
    validationSample = [i for i in indexes if not i in trainSample]

    trainInputs = [inputs[i] for i in trainSample]
    trainOutputs = [outputs[i] for i in trainSample]

    validationInputs = [inputs[i] for i in validationSample]
    validationOutputs = [outputs[i] for i in validationSample]

    plt.plot(trainInputs, trainOutputs, 'ro', label='Training Data')
    plt.plot(validationInputs, validationOutputs, 'g^', label='Validation Data')
    plt.title('Train and Validation data')
    plt.xlabel(label)
    plt.ylabel('Happiness')
    plt.legend()
    plt.show()
    return trainInputs, trainOutputs, validationInputs, validationOutputs


def trainTheModel(trainInputs, trainOutputs):
    xx = [[el] for el in trainInputs]

    # model initialisation
    regressor = linear_model.LinearRegression()
    # training the model by using the training inputs and known training outputs
    regressor.fit(xx, trainOutputs)
    # save the model parameters
    w0, w1 = regressor.intercept_, regressor.coef_[0]
    print('the learnt model: f(x) = ', w0, ' + ', w1, ' * x')

    return w0, w1, regressor


def plotTheTrainedModel(validationInputs, validationOutputs, label, w0, w1):
    noOfPoints = 1000
    xref = []
    val = min(validationInputs)
    step = (max(validationInputs) - min(validationOutputs)) / noOfPoints
    for i in range(1, noOfPoints):
        xref.append(val)
        val += step
    yref = [w0 + w1 * el for el in xref]

    plt.plot(validationInputs, validationOutputs, 'ro', label='Training data')
    plt.plot(xref, yref, 'b-', label='Learnt model')
    plt.title('Train Data and the Learnt Model')
    plt.xlabel(label)
    plt.ylabel('Happiness')
    plt.legend()
    plt.show()


def plotThePredictedAndValidationData(regressor, validationInputs, validationOutputs, label):
    computedValidationOutputs = regressor.predict([[x] for x in validationInputs])

    # plot the computed outputs (see how far they are from the real outputs)
    plt.plot(validationInputs, computedValidationOutputs, 'yo',
             label='Predicted Validation Data')  # computed test data are plotted yellow red and circle sign
    plt.plot(validationInputs, validationOutputs, 'g^',
             label='Real Validation Data')  # real test data are plotted by green triangles
    plt.title('Predicted and Real Validation data')
    plt.xlabel(label)
    plt.ylabel('Happiness')
    plt.legend()
    plt.show()
    return computedValidationOutputs


def predictionError(predictedData, outputs):
    error = 0.0
    for t1, t2 in zip(predictedData, outputs):
        error += (t1 - t2) ** 2
    error = error / len(outputs)
    print('Prediction error (manual): ', error)

    error = mean_squared_error(outputs, predictedData)
    print('Prediction error (tool):  ', error)


def main():
    crtDir = os.getcwd()
    filePath = os.path.join(crtDir, 'data', 'v1_world-happiness-report-2017.csv')
    # Load Data for the feature Family
    inputs, outputs = loadData1(filePath, 'Family', 'Happiness.Score')
    # print(inputs[:155])
    # print(outputs[:155])

    plotDataHistogram(inputs, 'Family')
    plotDataHistogram(outputs, 'Happiness')

    plotSpecificHistogram(inputs, outputs, 'Family vs Happiness')

    trainInputs, trainOutputs, validationInputs, validationOutputs = trainAndValidationSample(inputs, outputs, 'Family')

    w0, w1, regression = trainTheModel(trainInputs, trainOutputs)

    plotTheTrainedModel(validationInputs, validationOutputs, 'Family', w0, w1)

    predictedData = plotThePredictedAndValidationData(regression, validationInputs, validationOutputs, 'Family')

    predictionError(predictedData, validationOutputs)

    crtDir = os.getcwd()
    filePath = os.path.join(crtDir, 'data', 'v2_world-happiness-report-2017.csv')

    inputs, outputs = loadData1(filePath, 'Family', 'Happiness.Score')
    # print(inputs[:155])
    # print(outputs[:155])

    plotDataHistogram(inputs, 'Family')
    plotDataHistogram(outputs, 'Happiness')

    plotSpecificHistogram(inputs, outputs, 'Family vs Happiness')

    trainInputs, trainOutputs, validationInputs, validationOutputs = trainAndValidationSample(inputs, outputs, 'Family')

    w0, w1, regression = trainTheModel(trainInputs, trainOutputs)

    plotTheTrainedModel(validationInputs, validationOutputs, 'Family', w0, w1)

    predictedData = plotThePredictedAndValidationData(regression, validationInputs, validationOutputs, 'Family')

    predictionError(predictedData, validationOutputs)

main()
