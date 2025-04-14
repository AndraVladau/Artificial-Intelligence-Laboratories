"""
1. Data about a company's employees are known, data saved in the "data/employees.csv" file.

1. a. To establish:
- the number of employees
- number and type of information (properties) held for an employee
- the number of employees for whom complete data is available
- minimum, maximum, average values for each property
- in the case of non-numerical properties, how many possible values does each such property have
- if there are missing values and how this problem can be solved

1.b. To visualize:
- the distribution of the salaries of these employees by salary category
- the distribution of the salaries of these employees by salary category and the team they belong to
- employees who can be considered "outliers"
"""

import csv
import os

import matplotlib.pyplot as plt
from math import log, sqrt

from numpy import mean


# load all the data from a csv file
def loadDataMoreInputs(fileName):
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
    return dataNames, data


# Number of employees
print(len(loadDataMoreInputs("data\employees.csv")[1]))

# Number and type of the properties that an employee has
print(len(loadDataMoreInputs("data\employees.csv")[0]))
print(', '.join(loadDataMoreInputs("data\employees.csv")[0]))


# Number of employees that have complete data
def employeesCompleteData(data):
    noOfEmployees = 0
    for employee in data:
        if '' not in employee:
            noOfEmployees += 1
    return noOfEmployees


print(employeesCompleteData(loadDataMoreInputs("data\employees.csv")[1]))


# extract a particular feature (column)
def extractFeature(all_data, all_names, featureName):
    pos = all_names.index(featureName)
    return [float(data[pos]) for data in all_data]


s = extractFeature(loadDataMoreInputs("data\employees.csv")[1], loadDataMoreInputs("data\employees.csv")[0], 'Salary')
print(max(s))
print(min(s))
print(mean(s))

b = extractFeature(loadDataMoreInputs("data\employees.csv")[1], loadDataMoreInputs("data\employees.csv")[0], 'Bonus %')
print(max(b))
print(min(b))
print(mean(b))


def extractFeatureNonNumber(all_data, all_names, featureName):
    pos = all_names.index(featureName)
    return [data[pos] for data in all_data if data[pos] != '']


n = set(
    extractFeatureNonNumber(loadDataMoreInputs("data\employees.csv")[1], loadDataMoreInputs("data\employees.csv")[0],
                            'First Name'))
g = set(
    extractFeatureNonNumber(loadDataMoreInputs("data\employees.csv")[1], loadDataMoreInputs("data\employees.csv")[0],
                            'Gender'))
sd = set(
    extractFeatureNonNumber(loadDataMoreInputs("data\employees.csv")[1], loadDataMoreInputs("data\employees.csv")[0],
                            'Start Date'))
lt = set(
    extractFeatureNonNumber(loadDataMoreInputs("data\employees.csv")[1], loadDataMoreInputs("data\employees.csv")[0],
                            'Last Login Time'))
sm = set(
    extractFeatureNonNumber(loadDataMoreInputs("data\employees.csv")[1], loadDataMoreInputs("data\employees.csv")[0],
                            'Senior Management'))
t = set(
    extractFeatureNonNumber(loadDataMoreInputs("data\employees.csv")[1], loadDataMoreInputs("data\employees.csv")[0],
                            'Team'))

print(len(n))
print(len(g))
print(len(sd))
print(len(lt))
print(len(sm))
print(len(t))


def plotDataHistogram(x, featureName):
    """
    Histogram of salaries
    :param x: the salaries
    :param featureName: feature for what we do the histogram
    """
    plt.hist(x, 20, rwidth=0.9, color='lightblue', edgecolor='black')
    plt.title('Histogram of ' + featureName)
    plt.show()


plotDataHistogram(s, 'Salary')


def plotDataHistogramSalaryTeam():
    for team in t:
        salariesTeam = []
        for employee in loadDataMoreInputs('data\employees.csv')[1]:
            if employee[7] == team:
                salariesTeam.append(float(employee[4]))
        plotDataHistogram(salariesTeam, team)


# plotDataHistogramSalaryTeam()


def outliers():
    """
    The employees who are outliers
    """
    names, allData = loadDataMoreInputs('data\employees.csv')

    salary = extractFeature(allData, names, 'Salary')

    plt.boxplot(salary)  # From the boxplot it appears that there is no outlier
    plt.title('Boxplot of the Outliers')
    plt.show()


outliers()
