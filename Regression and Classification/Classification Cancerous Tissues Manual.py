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


def splitTheDataTexture(inputs1, inputs2, out):
    np.random.seed(155)
    inputsTexture = []
    inputsRadius = []
    outputs = []
    for i in range(len(inputs1)):
        if 10 <= float(inputs1[i]) < 11 and 18 <= float(inputs2[i]) < 19:
            inputsTexture.append(inputs1[i])
            inputsRadius.append(inputs2[i])
            outputs.append(out[i])

    outputsType = []
    for i in outputs:
        if i == 'B':
            outputsType.append(1)
        else:
            outputsType.append(0)

    indexesTexture = [i for i in range(len(inputsTexture))]
    trainSampleTexture = np.random.choice(indexesTexture, int(0.5 * len(inputsTexture)), replace=False)
    validationSampleTexture = [i for i in indexesTexture if not i in trainSampleTexture]

    trainInputsTexture = [indexesTexture[i] for i in trainSampleTexture]
    trainOutputsTexture = [outputsType[i] for i in trainSampleTexture]

    validationInputsTexture = [indexesTexture[i] for i in validationSampleTexture]
    validationOutputsTexture = [outputsType[i] for i in validationSampleTexture]

    indexesRadius = [i for i in range(len(inputsRadius))]
    trainSampleRadius = np.random.choice(indexesRadius, int(0.5 * len(inputsTexture)), replace=False)
    validationSampleRadius = [i for i in indexesRadius if not i in trainSampleRadius]

    trainInputsRadius = [indexesRadius[i] for i in trainSampleRadius]

    validationInputsRadius = [indexesRadius[i] for i in validationSampleRadius]

    return trainInputsTexture, trainOutputsTexture, validationInputsTexture, validationOutputsTexture, trainInputsRadius, validationInputsRadius


def trainedModel(inputs1, inputs2, outputs):
    x = np.column_stack((inputs1, inputs2))
    # print(x)
    # model initialisation
    regressor = MyLogisticRegressionClassifier()
    # training the model by using the training inputs and known training outputs
    regressor.fit(x, outputs)
    # save the model parameters
    w0, w1, w2 = regressor.intercept_, regressor.coef_[0], regressor.coef_[1]
    print('The Learnt Model: f(x) = ', w0, ' + ', w1, ' * x', ' + ', w2, ' * y')
    return w0, w1, w2, regressor


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


def plotLearnedModel(inputs1, inputs2, regressor, w0, w1, w2, label):
    prediction = regressor.predict(np.column_stack((inputs1, inputs2)))
    noOfPoints = 1000
    minGDP, maxGDP = min(inputs1), max(inputs1)
    minFreedom, maxFreedom = min(inputs2), max(inputs2)
    xx, yy = np.meshgrid(np.linspace(minGDP, maxGDP, noOfPoints), np.linspace(minFreedom, maxFreedom, noOfPoints))
    zz = np.array([w0 + w1 * el1 + w2 * el2 for el1, el2 in zip(xx, yy)])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(inputs1, inputs2, prediction, color='green', marker='o', label=label)
    ax.plot_surface(xx, yy, zz, alpha=0.5)

    ax.set_xlabel('Gross Domestic Product')
    ax.set_ylabel('Freedom')
    ax.set_zlabel('Happiness')
    string = '3D Linear Regression for ' + label
    plt.title(string)
    plt.legend()
    plt.show()

    return prediction


def predictionError(outputs, prediction):

    error = 0.0
    for t1, t2 in zip(prediction, outputs):
        error += (t1 - t2) ** 2
    error = error / len(outputs)
    print('Prediction Error (manual)', error)

    # by using sklearn
    error = mean_squared_error(prediction, outputs)
    print('Prediction Error (tool)', error)


def main():
    world_happiness = loadData("data/breast-cancer-wisconsin-diagnostic.csv")

    trainInputsTexture, trainOutputs, validationInputsTexture, validationOutputs, trainInputsRadius, validationInputsRadius =\
        splitTheDataTexture(world_happiness['Texture'], world_happiness['Radius'], world_happiness['Malformation.Type'])

    w0, w1, w2, regressor = trainedModel(trainInputsTexture, trainInputsRadius, trainOutputs)

    # Plot the Learn Model with validation data
    prediction = plotLearnedModel(validationInputsTexture, validationInputsRadius, regressor, w0, w1, w2, 'Validation Data')

    # Prediction Error for the Train Data
    print("Training Data")
    predictionError(validationOutputs, prediction)

    # Prediction Error for the Validation Data
    print("\nValidation Data")
    predictionError(validationOutputs, prediction)


main()
