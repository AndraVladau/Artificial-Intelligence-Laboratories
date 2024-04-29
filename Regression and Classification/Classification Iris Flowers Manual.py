"""
What kind of flower do you prefer?

Consider the problem of classifying iris flowers into different species such as: setosa, versicolor and virginica.
For each flower, characteristics such as: sepal length and width, petal length and width are known.
Using this information, decide which species a particular flower belongs to.

A classification algorithm based on logistic regression to determine:
    - the species of a flower that has a sepal 5.35 cm long and 3.85 cm wide, and a petal 1.25 cm long and 0.4 cm wide
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from Regression.LogisticRegression import MyLogisticRegressionClassifier


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


def splitTheDataSepal(inputs1, inputs2, inputs3, inputs4, outputs):
    np.random.seed(155)
    inputsSepalLength = []
    inputsSepalWidth = []
    outputsType = []
    for i in range(len(inputs1)):
        if 5.30 <= float(inputs1[i]) <= 5.40 and 3.5 <= float(inputs2[i]) <= 4:
            inputsSepalLength.append(inputs1[i])
            inputsSepalWidth.append(inputs2[i])
            outputsType.append(outputs[i])

    indexesPetals = [i for i in range(len(inputsSepalLength))]
    trainSamplePetals = np.random.choice(indexesPetals, int(0.5 * len(inputsSepalLength)), replace=False)
    validationSampleTexture = [i for i in indexesPetals if not i in trainSamplePetals]

    trainInputsTexture = [indexesPetals[i] for i in trainSamplePetals]
    trainOutputsTexture = [outputsType[i] for i in trainSamplePetals]

    validationInputsTexture = [indexesPetals[i] for i in validationSampleTexture]
    validationOutputsTexture = [outputsType[i] for i in validationSampleTexture]

    indexesRadius = [i for i in range(len(inputsSepalWidth))]
    trainSampleRadius = np.random.choice(indexesRadius, int(0.5 * len(inputsSepalWidth)), replace=False)
    validationSampleRadius = [i for i in indexesRadius if not i in trainSampleRadius]

    trainInputsRadius = [indexesRadius[i] for i in trainSampleRadius]

    validationInputsRadius = [indexesRadius[i] for i in validationSampleRadius]

    return trainInputsTexture, trainOutputsTexture, validationInputsTexture, validationOutputsTexture, trainInputsRadius, validationInputsRadius

# def trainedModel(inputs1, inputs2, outputs):
#     x = np.column_stack((inputs1, inputs2))
#     # print(x)
#     # model initialisation
#     regressor = MyLogisticRegressionClassifier()
#     # training the model by using the training inputs and known training outputs
#     regressor.fit(x, outputs)
#     # save the model parameters
#     w0, w1, w2 = regressor.intercept_, regressor.coef_[0], regressor.coef_[1]
#     print('The Learnt Model: f(x) = ', w0, ' + ', w1, ' * x', ' + ', w2, ' * y')
#     return w0, w1, w2, regressor
#
#
# def plotTheLearntModel(trainInputs, trainOutputs, w0, w1):
#     noOfPoints = 1000
#     xref = []
#     val = min(trainInputs)
#     step = (max(trainInputs) - min(trainInputs)) / noOfPoints
#     for i in range(1, noOfPoints):
#         xref.append(val)
#         val += step
#     yref = [w0 + w1 * el for el in xref]
#
#     plt.plot(trainInputs, trainOutputs, 'go', label='Training Data')  # train data are plotted by red and circle sign
#     plt.plot(xref, yref, 'b-', label='Learnt Model')  # model is plotted by a blue line
#     plt.title('Train Data and the Learnt Model')
#     plt.xlabel('Gross Domestic Product')
#     plt.ylabel('Happiness')
#     plt.legend()
#     plt.show()
#
#
# def plotLearnedModel(inputs1, inputs2, regressor, w0, w1, w2, label):
#     prediction = regressor.predict(np.column_stack((inputs1, inputs2)))
#     noOfPoints = 1000
#     minGDP, maxGDP = min(inputs1), max(inputs1)
#     minFreedom, maxFreedom = min(inputs2), max(inputs2)
#     xx, yy = np.meshgrid(np.linspace(minGDP, maxGDP, noOfPoints), np.linspace(minFreedom, maxFreedom, noOfPoints))
#     zz = np.array([w0 + w1 * el1 + w2 * el2 for el1, el2 in zip(xx, yy)])
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(inputs1, inputs2, prediction, color='green', marker='o', label=label)
#     ax.plot_surface(xx, yy, zz, alpha=0.5)
#
#     ax.set_xlabel('Gross Domestic Product')
#     ax.set_ylabel('Freedom')
#     ax.set_zlabel('Happiness')
#     string = '3D Linear Regression for ' + label
#     plt.title(string)
#     plt.legend()
#     plt.show()
#
#     return prediction
#
#
# def predictionError(outputs, prediction):
#
#     error = 0.0
#     for t1, t2 in zip(prediction, outputs):
#         error += (t1 - t2) ** 2
#     error = error / len(outputs)
#     print('Prediction Error (manual)', error)
#
#     # by using sklearn
#     error = mean_squared_error(prediction, outputs)
#     print('Prediction Error (tool)', error)


def main():
    iris_data = loadData("data/iris.csv")

    trainInputsTexture, trainOutputs, validationInputsTexture, validationOutputs, trainInputsRadius, validationInputsRadius =\
        splitTheDataSepal(iris_data['Sepal.Length'], iris_data['Sepal.Width'], iris_data['Iris.Species'])

    # w0, w1, w2, regressor = trainedModel(trainInputsTexture, trainInputsRadius, trainOutputs)
    #
    # # Plot the Learn Model with validation data
    # prediction = plotLearnedModel(validationInputsTexture, validationInputsRadius, regressor, w0, w1, w2, 'Validation Data')
    #
    # # Prediction Error for the Train Data
    # print("Training Data")
    # predictionError(validationOutputs, prediction)
    #
    # # Prediction Error for the Validation Data
    # print("\nValidation Data")
    # predictionError(validationOutputs, prediction)


main()
