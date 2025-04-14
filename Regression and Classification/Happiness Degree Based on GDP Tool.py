"""
What can make people happy?
It considers the problem of predicting the degree of happiness of the population of the globe
using information about various characteristics of the well-being of the respective population,
such as the gross domestic product of the country in which they live (GBP), the degree of happiness, etc.

To make a prediction of the degree of happiness according to:
    - only by the gross domestic product
    - of the gross domestic product and the degree of freedom.

A regression algorithm for the first problem based on:
    - stochastic descending gradient method
    - descending gradient method based on batches, with tool/API and/or own code.
"""

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Regression.BGDRegression import MyBGDRegression


def loadData(filename: str):
    data = pd.read_csv(filename, delimiter=',', header='infer')
    data = data.dropna()
    return data


def plotDataHistogram(x, variableName):
    n, bins, patches = plt.hist(x, 10, color='lightblue', edgecolor='black')
    plt.title('Histogram of ' + variableName)
    plt.show()


def checkLinearRelationship(inputs, outputs):
    plt.plot(inputs, outputs, 'bo')
    plt.xlabel('Gross Domestic Product')
    plt.ylabel('Happiness')
    plt.title('Gross Domestic Product vs. Happiness')
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

    plt.plot(trainInputs, trainOutputs, 'bo', label='Training Data')  # train data are plotted by red and circle sign
    plt.plot(testInputs, testOutputs, 'g^', label='Testing Data')  # test data are plotted by green and a triangle sign
    plt.title('Training and Testing Data')
    plt.xlabel('Gross Domestic Product')
    plt.ylabel('Happiness')
    plt.legend()
    plt.show()

    return trainInputs, trainOutputs, testInputs, testOutputs


def trainSGDRegressionModel(trainInputs, trainOutputs):
    xx = [[el] for el in trainInputs]

    regressor = SGDRegressor(learning_rate='constant', eta0=0.01, max_iter=1000, tol=1e-3, random_state=5)

    regressor.fit(xx, trainOutputs)

    w0, w1 = regressor.intercept_, regressor.coef_[0]
    print('The Learnt Model: f(x) = ', w0, ' + ', w1, ' * x')

    return w0, w1, regressor


def trainBGDRegressionModel(trainInputs, trainOutputs):
    xx = [[el] for el in trainInputs]

    regressor = MyBGDRegression()

    regressor.fit(xx, trainOutputs)

    w0, w1 = regressor.intercept_, regressor.coef_[0]
    print('The Learnt Model: f(x) = ', w0, ' + ', w1, ' * x')

    return w0, w1, regressor


def plotTheLearntModel(trainInputs, trainOutputs, w0, w1):
    noOfPoints = 1000
    xref = []
    val = min(trainInputs)
    step = (max(trainInputs) - min(trainInputs)) / noOfPoints
    for i in range(1, noOfPoints):
        xref.append(val)
        val += step
    yref = [w0 + w1 * el for el in xref]

    plt.plot(trainInputs, trainOutputs, 'go', label='Training Data')  # train data are plotted by red and circle sign
    plt.plot(xref, yref, 'b-', label='Learnt Model')  # model is plotted by a blue line
    plt.title('Train Data and the Learnt Model')
    plt.xlabel('Gross Domestic Product')
    plt.ylabel('Happiness')
    plt.legend()
    plt.show()


def plotComputedOutputs(regressor, testInputs, testOutputs):
    computedTestOutputs = regressor.predict([[x] for x in testInputs])

    # plot the computed outputs (see how far they are from the real outputs)
    plt.plot(testInputs, computedTestOutputs, 'bo', label='Computed Test Data')  # computed test data are plotted yellow red and circle sign
    plt.plot(testInputs, testOutputs, 'g^', label='Real Test Data')  # real test data are plotted by green triangles
    plt.title('Computed Test and Real Test Data')
    plt.xlabel('Gross Domestic Product')
    plt.ylabel('Happiness')
    plt.legend()
    plt.show()

    return computedTestOutputs


def predictionError(computedTestOutputs, testOutputs):
    error = 0.0
    for t1, t2 in zip(computedTestOutputs, testOutputs):
        # print((t1-t2)**2)
        error += (t1 - t2) ** 2
    error = error / len(testOutputs)
    print('Prediction Error (manual): ', error)

    error = mean_squared_error(testOutputs, computedTestOutputs)
    print('Prediction Error (tool):  ', error)


def main():
    world_happiness = loadData("data/world-happiness-report-2017.csv")
    # print(world_happiness["Economy..GDP.per.Capita."][0])

    plotDataHistogram(world_happiness["Economy..GDP.per.Capita."], "Gross Domestic Product")
    plotDataHistogram(world_happiness["Happiness.Score"], "Happiness")

    checkLinearRelationship(world_happiness["Economy..GDP.per.Capita."], world_happiness["Happiness.Score"])

    trainInputs, trainOutputs, testInputs, testOutputs = splitTheData(world_happiness["Economy..GDP.per.Capita."], world_happiness["Happiness.Score"])

    print("Stochastic Descending Gradient Regression\n")
    w0, w1, regressor = trainSGDRegressionModel(trainInputs, trainOutputs)

    plotTheLearntModel(trainInputs, trainOutputs, w0, w1)

    computedTestOutputs = plotComputedOutputs(regressor, testInputs, testOutputs)

    predictionError(computedTestOutputs, testOutputs)

    print("\nBatch Descending Gradient Regression\n")
    w2, w3, regressor1 = trainBGDRegressionModel(trainInputs, trainOutputs)

    plotTheLearntModel(trainInputs, trainOutputs, w2, w3)

    computedTestOutputs = plotComputedOutputs(regressor1, testInputs, testOutputs)

    predictionError(computedTestOutputs, testOutputs)


main()
