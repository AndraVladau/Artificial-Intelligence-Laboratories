"""
1. Determine the quality of the text recognition process, both at the character level and at the word level:
   a. by using a distance metric or
   b. by using several distance metrics.
"""

import os
import time

import nltk
from Levenshtein import distance as lev
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from jarowinkler import *
from msrest.authentication import CognitiveServicesCredentials


def takeImage(image):
    subscription_key = os.environ["VISION_KEY"]
    endpoint = os.environ["VISION_ENDPOINT"]
    computerVision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

    img = open(image, "rb")
    read_response = computerVision_client.read_in_stream(
        image=img,
        mode="Printed",
        raw=True
    )

    operation_id = read_response.headers['Operation-Location'].split('/')[-1]
    while True:
        read_result = computerVision_client.get_read_result(operation_id)
        if read_result.status not in ['notStarted', 'running']:
            break
        time.sleep(1)

    result = []
    if read_result.status == OperationStatusCodes.succeeded:
        for text_result in read_result.analyze_result.read_results:
            for line in text_result.lines:
                print(line.text)
                result.append(line.text)

    # print(result)
    return result


def distanceLevenshtein(result, groundTruth):
    distanceForWords = 0
    distanceForCharacters = 0
    for wordResult, wordGroundTruth in zip(result, groundTruth):
        distanceForCharacters += lev(wordResult, wordGroundTruth)
        wordResult = wordResult.split(' ')
        wordGroundTruth = wordGroundTruth.split(' ')
        if len(wordResult) == len(wordGroundTruth):
            for i in range(len(wordResult)):
                if wordResult[i] != wordGroundTruth[i]:
                    distanceForWords += 1
        else:
            distanceForWords += (len(wordResult) - len(wordGroundTruth))
    print(distanceForWords, distanceForCharacters)
    return


def distanceJaroWinkler(result, groundTruth):
    distanceArray = []
    distance = 0
    for r, t in zip(result, groundTruth):
        r = nltk.word_tokenize(r)
        t = nltk.word_tokenize(t)

        for wordResult, wordGroundTruth in zip(r, t):
            jaro = jarowinkler_similarity(wordResult, wordGroundTruth)
            distanceArray.append(1-jaro)

    for i in distanceArray:
        distance += i

    return distance/len(distanceArray)


def distanceHamming(result, groundTruth):
    if len(result) != len(groundTruth):
        return -1
    return sum(i != j for i, j in zip(result, groundTruth))


def main():
    image1 = 'data/test1.png'
    image2 = 'data/test2.jpeg'

    groundTruth1 = ["Google Cloud", "Platform"]
    groundTruth2 = ["Succes în rezolvarea", "tEMELOR la", "LABORAtoarele de", "Inteligență Artificială!"]

    print("Result 1: ")
    result1 = takeImage(image1)

    print("\nResult 2: ")
    result2 = takeImage(image2)

    print("\nDistance Levenshtein: ")
    distanceLevenshtein(result1, groundTruth1)
    distanceLevenshtein(result2, groundTruth2)

    print("\nDistance Jaro-Winkler for words: ")
    print(distanceJaroWinkler(result1, groundTruth1))
    print(distanceJaroWinkler(result2, groundTruth2))

    print("\nDistance Hamming for characters: ")
    print(distanceHamming(result1, groundTruth1))
    print(distanceHamming(result2, groundTruth2))


main()
