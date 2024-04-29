"""
What can make people happy?
It considers the problem of predicting the degree of happiness of the population of the globe using information about
various characteristics of the well-being of the respective population, such as the gross domestic product of the country
in which they live (GBP), the degree of happiness, etc.

Using the data related to the year 2017 (file v1_world-happiness-report-2017.csv),
make a prediction of the degree of happiness according to of Gross Domestic Product and degree of freedom.
"""

import csv
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import pandas as pd


# Load data from file
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


def loadData2(filename: str):
    data = pd.read_csv(filename, delimiter=',', header='infer')
    data = data.dropna()
    return data


# see how the data looks (plot the histograms associated to input data
# - GDP feature - and output data - happiness)

# Plots a Histogram
def plotDataHistogram(x, y, z, label1, label2, label3):
    n, bins, patches = plt.hist(x, 10, color='lightblue', edgecolor='black')
    plt.title('Histogram of ' + label1)
    plt.show()

    n, bins, patches = plt.hist(y, 10, color='lightblue', edgecolor='black')
    plt.title('Histogram of ' + label2)
    plt.show()

    n, bins, patches = plt.hist(z, 10, color='lightblue', edgecolor='black')
    plt.title('Histogram of ' + label3)
    plt.show()


def plotSpecificHistogram(x, y, label1, label2):
    plt.plot(x, y, 'ro', color='lightblue')
    plt.xlabel(label1)
    plt.ylabel(label2)
    label = label1 + ' vs. ' + label2
    plt.title(label)
    plt.show()


def plot3DHistogram(x, y, z):
    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(111, projection='3d')
    labels = ["GDP.Per.Capita", "Freedom", "Happiness"]

    for l in labels:
        gdpPerCapita = x

        freedom = y

        happiness = z

        ax.scatter(xs=gdpPerCapita, ys=freedom, zs=happiness, label=l)

    ax.set_title("GDP.Per.Capita Freedom Happiness Distribution")

    ax.set_xlabel("GDP.Per.Capita")

    ax.set_ylabel("Freedom")

    ax.set_zlabel("Happiness")

    ax.legend(loc="best")

    plt.show()


def trainAndValidationSample(inputs, outputs):
    np.random.seed(155)
    indexes = [i for i in range(len(inputs))]
    trainSample = np.random.choice(indexes, int(0.8 * len(inputs)), replace=False)
    validationSample = [i for i in indexes if not i in trainSample]

    trainInputs = [inputs[i] for i in trainSample]
    trainOutputs = [outputs[i] for i in trainSample]

    validationInputs = [inputs[i] for i in validationSample]
    validationOutputs = [outputs[i] for i in validationSample]

    return [trainInputs, trainOutputs, validationInputs, validationOutputs]


def plot3dTrainAndValidation(trainInputs1, trainInputs2, trainOutputs, validationInputs1, validationInputs2, validationOutputs):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(trainInputs1, trainInputs2, trainOutputs, color='blue', label='Training Data')
    ax.scatter(validationInputs1, validationInputs2, validationOutputs, color='green', label='Validation Data')
    ax.set_xlabel('GDP.Per.Capita')
    ax.set_ylabel('Freedom')
    ax.set_zlabel('Happiness')
    ax.set_title('Training and Validation Data Distribution')
    plt.legend()
    plt.show()


def trainedModel(inputs1, inputs2, outputs):
    x = np.column_stack((inputs1, inputs2))
    print(x)
    # model initialisation
    regressor = linear_model.LinearRegression()
    # training the model by using the training inputs and known training outputs
    regressor.fit(x, outputs)
    # save the model parameters
    w0, w1, w2 = regressor.intercept_, regressor.coef_[0], regressor.coef_[1]
    print('the learnt model: f(x) = ', w0, ' + ', w1, ' * x', ' + ', w2, ' * y')
    return w0, w1, w2, regressor


def plotLearnedModel(inputs1, inputs2, outputs, w0, w1, w2, label):
    noOfPoints = 1000
    minGDP, maxGDP = min(inputs1), max(inputs1)
    minFreedom, maxFreedom = min(inputs2), max(inputs2)
    xx, yy = np.meshgrid(np.linspace(minGDP, maxGDP, noOfPoints), np.linspace(minFreedom, maxFreedom, noOfPoints))
    zz = np.array([w0 + w1 * el1 + w2 * el2 for el1, el2 in zip(xx, yy)])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(inputs1, inputs2, outputs, color='green', marker='o', label=label)
    ax.plot_surface(xx, yy, zz, alpha=0.5)

    ax.set_xlabel('GDP capita')
    ax.set_ylabel('Freedom')
    ax.set_zlabel('Happiness')
    string = '3D Linear Regression for ' + label
    plt.title(string)
    plt.legend()
    plt.show()


def predictionError(inputs1, inputs2, outputs, regressor):
    prediction = regressor.predict(np.column_stack((inputs1, inputs2)))
    error = 0.0
    for t1, t2 in zip(prediction, outputs):
        error += (t1 - t2) ** 2
    error = error / len(outputs)
    print('Prediction Error (manual)', error)

    # by using sklearn
    from sklearn.metrics import mean_squared_error

    error = mean_squared_error(prediction, outputs)
    print('Prediction Error (tool)', error)


def main():
    # Load data from the file
    crtDir = os.getcwd()
    filePath = os.path.join(crtDir, 'data', 'v1_world-happiness-report-2017.csv')
    # Plots Histograms for GDP and Freedom after Happiness
    inputsGDP, outputsGDP = loadData1(filePath, 'Economy..GDP.per.Capita.', 'Happiness.Score')
    inputsFreedom, outputsFreedom = loadData1(filePath, 'Freedom', 'Happiness.Score')
    print('in:  ', inputsGDP[:155])
    print('out: ', outputsGDP[:155])
    print('in:  ', inputsFreedom[:155])
    print('out: ', outputsFreedom[:155])

    inputsGDP1 = loadData2(filePath)

    # Plot the Histogram for the feature GDP.per.Capita
    plotDataHistogram(inputsGDP, inputsFreedom, outputsGDP, 'GDP.per.Capita', 'Freedom', 'Happiness')

    # Plot the Histogram for the Happiness score depending on GDP.per.Capita and Freedom
    plotDataHistogram(outputsGDP, 'Happiness score depending on GDP.per.Capita and Freedom')

    # Plot GDP.per.Capita vs Happiness inputs and outputs
    plotSpecificHistogram(inputsGDP, outputsGDP, 'GDP.per.Capita', 'Happiness')

    # Plot Freedom vs Happiness inputs and outputs
    plotSpecificHistogram(inputsFreedom, outputsFreedom, 'Freedom', 'Happiness')

    # Plot 3d GDP.per.Capita vs Freedom vs Happiness
    plot3DHistogram(inputsGDP, inputsFreedom, outputsFreedom)

    # Train Model for GDP.per.Capita and Happiness
    lst = trainAndValidationSample(inputsGDP, outputsGDP)
    trainInputsGDP = lst[0]
    trainOutputs = lst[1]
    validationInputsGDP = lst[2]
    validationOutputs = lst[3]

    # Train Model for Freedom and Happiness
    lst2 = trainAndValidationSample(inputsFreedom, outputsFreedom)
    trainInputsFreedom = lst2[0]
    validationInputsFreedom = lst2[2]

    # Plot 3d the Train and Validation Data of GDP.per.Capita, Freedom and Happiness
    plot3dTrainAndValidation(trainInputsGDP, trainInputsFreedom, trainOutputs, validationInputsGDP, validationInputsFreedom, validationOutputs)

    # Create the linear regressor
    w0, w1, w2, regressor = trainedModel(trainInputsGDP, trainInputsFreedom, trainOutputs)

    # Plot the Learn Model with validation data
    plotLearnedModel(validationInputsGDP, validationInputsFreedom, validationOutputs, w0, w1, w2, 'Validation Data')

    # Prediction Error for the Train Data
    print("Training Data")
    predictionError(trainInputsGDP, trainInputsFreedom, trainOutputs, regressor)

    # Prediction Error for the Validation Data
    print("\nValidation Data")
    predictionError(validationInputsGDP, validationInputsFreedom, validationOutputs, regressor)

    crtDir = os.getcwd()
    filePath = os.path.join(crtDir, 'data', 'v2_world-happiness-report-2017.csv')
    # Plots Histograms for GDP and Freedom after Happiness
    inputsGDP, outputsGDP = loadData1(filePath, 'Economy..GDP.per.Capita.', 'Happiness.Score')
    inputsFreedom, outputsFreedom = loadData1(filePath, 'Freedom', 'Happiness.Score')
    print('in:  ', inputsGDP[:155])
    print('out: ', outputsGDP[:155])
    print('in:  ', inputsFreedom[:155])
    print('out: ', outputsFreedom[:155])

    inputsGDP1 = loadData2(filePath)

    plotDataHistogram(inputsGDP, inputsFreedom, outputsGDP, 'GDP.per.Capita', 'Freedom', 'Happiness')

    # Plot the Histogram for the Happiness score depending on GDP.per.Capita and Freedom
    plotDataHistogram(outputsGDP, 'Happiness score depending on GDP.per.Capita and Freedom')

    # Plot GDP.per.Capita vs Happiness inputs and outputs
    plotSpecificHistogram(inputsGDP, outputsGDP, 'GDP.per.Capita', 'Happiness')

    # Plot Freedom vs Happiness inputs and outputs
    plotSpecificHistogram(inputsFreedom, outputsFreedom, 'Freedom', 'Happiness')

    # Plot 3d GDP.per.Capita vs Freedom vs Happiness
    plot3DHistogram(inputsGDP, inputsFreedom, outputsFreedom)

    # Train Model for GDP.per.Capita and Happiness
    lst = trainAndValidationSample(inputsGDP, outputsGDP)
    trainInputsGDP = lst[0]
    trainOutputs = lst[1]
    validationInputsGDP = lst[2]
    validationOutputs = lst[3]

    # Train Model for Freedom and Happiness
    lst2 = trainAndValidationSample(inputsFreedom, outputsFreedom)
    trainInputsFreedom = lst2[0]
    validationInputsFreedom = lst2[2]

    # Plot 3d the Train and Validation Data of GDP.per.Capita, Freedom and Happiness
    plot3dTrainAndValidation(trainInputsGDP, trainInputsFreedom, trainOutputs, validationInputsGDP,
                             validationInputsFreedom, validationOutputs)

    # Create the linear regressor
    w0, w1, w2, regressor = trainedModel(trainInputsGDP, trainInputsFreedom, trainOutputs)

    # Plot the Learn Model with validation data
    plotLearnedModel(validationInputsGDP, validationInputsFreedom, validationOutputs, w0, w1, w2, 'Validation Data')

    # Prediction Error for the Train Data
    print("Training Data")
    predictionError(trainInputsGDP, trainInputsFreedom, trainOutputs, regressor)

    # Prediction Error for the Validation Data
    print("\nValidation Data")
    predictionError(validationInputsGDP, validationInputsFreedom, validationOutputs, regressor)


main()
