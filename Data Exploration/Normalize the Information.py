import csv
from collections import Counter

import nltk
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from numpy import random
import os


# problem 1

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


def extractFeature(allData, names, featureName):
    pos = names.index(featureName)
    return [float(data[pos]) for data in allData]


def normalizeInformationProblem1():
    names, allData = loadDataMoreInputs('data/employees.csv')
    salaries = extractFeature(allData, names, 'Salary')
    m = sum(salaries) / len(salaries)
    t = (1 / len(salaries) * sum([(s - m) ** 2 for s in salaries])) ** 0.5
    salariesZscore = [(s - m) / t for s in salaries]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.hist(salaries, 20, rwidth=0.9, color='lightblue', edgecolor='black')
    ax1.set_title('Salaries Histogram')
    ax2.hist(salariesZscore, 20, rwidth=0.9, color='lightblue', edgecolor='black')
    ax2.set_title('Z-Score Salaries Histogram')
    plt.show()

    bonuses = extractFeature(allData, names, 'Bonus %')
    m = sum(bonuses) / len(bonuses)
    t = (1 / len(bonuses) * sum([(b - m) ** 2 for b in bonuses])) ** 0.5
    bonusesZscore = [(s - m) / t for s in bonuses]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.hist(bonuses, 20, rwidth=0.9, color='lightblue', edgecolor='black')
    ax1.set_title('Bonuses Histogram')
    ax2.hist(bonusesZscore, 20, rwidth=0.9, color='lightblue', edgecolor='black')
    ax2.set_title('Z-Score Bonuses Histogram')
    plt.show()


normalizeInformationProblem1()


def normalizePixelImages():
    """
    Normalize the pixels in the images
    """

    imagesNormalize = []

    for img in os.listdir('data/images'):
        if img.endswith(('.jpg', '.webp', '.png')):
            cale_imagine = os.path.join('data/images', img)
            picture = Image.open(cale_imagine)
            imageArray = np.asarray(picture, dtype=np.float32)
            imageNormalize = imageArray / 255.0
            imagesNormalize.append(imageNormalize)

            plt.imshow(imageNormalize)
            plt.title('Normalize Image')
            plt.axis('off')
            plt.show()

    return imagesNormalize


print(normalizePixelImages())


# problem 3
def randomPhraseFile():
    """
    Take a random phrase from the text
    :return: the random phrase
    """
    with open('data/texts.txt', 'r', encoding='utf-8') as file:
        text = file.read()
    sentences = nltk.tokenize.sent_tokenize(text)
    randomPhrase = random.choice(sentences)
    return randomPhrase


print(randomPhraseFile())


def normalizeWordFrequency(phrase):
    """
    Normalize the word frequency from a random phrase from a text
    :param phrase: random phrase to be normalized
    """

    words = phrase.split()
    wordCounts = Counter(words)

    totalWords = len(words)

    frequency = {}
    for word, count in wordCounts.items():
        frequency[word] = count / totalWords

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.hist(frequency, 20, rwidth=0.9, color='lightblue', edgecolor='black')
    ax1.set_title('Word frequency Histogram')
    plt.show()

    return frequency


print(normalizeWordFrequency(randomPhraseFile()))

