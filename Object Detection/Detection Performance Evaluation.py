"""
1. To use an image classification algorithm (inference/testing stage) and to establish the performance of this binary
classification algorithm (images with bicycles vs. images without bicycles).
"""

import os

from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

subscription_key = os.environ["VISION_KEY"]
endpoint = os.environ["VISION_ENDPOINT"]
computerVision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))


def evalClassification(realLabels, computedLabels, labelNames):
    acc = accuracy_score(realLabels, computedLabels)
    precision = precision_score(realLabels, computedLabels, average=None, labels=labelNames)
    recall = recall_score(realLabels, computedLabels, average=None, labels=labelNames)
    return acc, precision, recall


def images(groundTruth):
    directory = 'data'
    bikesArray = []
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)

        bikes = ""

        img = open(f, "rb")
        result = computerVision_client.analyze_image_in_stream(img, visual_features=[VisualFeatureTypes.tags,
                                                                                     VisualFeatureTypes.objects])

        for tag in result.tags:
            if (tag.name == "bike") or (tag.name == "bicycle"):
                bikes = "with bike"
        if bikes == "":
            bikes = "without bike"
        bikesArray.append(bikes)
    print(bikesArray)

    accuracy, precision, recall = evalClassification(groundTruth, bikesArray, ["with bike", "without bike"])

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision[0]:.2f}, {precision[1]:.2f}")
    print(f"Recall: {recall[0]:.2f}, {recall[1]:.2f}")


def main():
    groundTruth = ["with bike", "with bike", "with bike", "with bike", "with bike", "with bike", "with bike",
                   "without bike", "without bike", "with bike", "without bike", "without bike", "without bike",
                   "without bike", "without bike", "without bike", "without bike", "without bike", "without bike",
                   "without bike"]
    images(groundTruth)


main()
