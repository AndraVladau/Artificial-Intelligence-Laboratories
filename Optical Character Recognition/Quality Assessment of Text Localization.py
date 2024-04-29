"""
2. Determine the quality of the correct localization of the text in the image.
3. Determine the possibilities for improving text recognition.
"""

import os
import time

from PIL import Image, ImageDraw, ImageEnhance
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from matplotlib import pyplot as plt
from msrest.authentication import CognitiveServicesCredentials

subscription_key = os.environ["VISION_KEY"]
endpoint = os.environ["VISION_ENDPOINT"]
computerVision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))


def intersectionOverUnion(bboxA, bboxB):
    boxA = [bboxA[0], bboxA[1], bboxA[4], bboxA[5]]
    boxB = [bboxB[0], bboxB[1], bboxB[4], bboxB[5]]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    intersectionArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    bboxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    bboxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = intersectionArea / float(bboxAArea + bboxBArea - intersectionArea)

    return iou


def analyzeImage(image, groundTruth):
    img = open(image, "rb")

    read_response = computerVision_client.read_in_stream(image=img, mode="Printed", raw=True)

    read_operation_location = read_response.headers["Operation-Location"]

    operation_id = read_operation_location.split("/")[-1]

    while True:
        read_result = computerVision_client.get_read_result(operation_id)
        if read_result.status not in ['notStarted', 'running']:
            break
        time.sleep(1)

    intersectionUnion = []

    if read_result.status == OperationStatusCodes.succeeded:
        for text_result in read_result.analyze_result.read_results:
            for idResult, line in enumerate(text_result.lines):
                boxCoordinatesGroundTruth = groundTruth[idResult]
                bbBoxCoordinates = line.bounding_box

                intersectionUnion.append(intersectionOverUnion(bbBoxCoordinates, boxCoordinatesGroundTruth))

                imagePIL = Image.open(image).convert("RGBA")
                draw = ImageDraw.Draw(imagePIL)
                draw.rectangle(((bbBoxCoordinates[0], bbBoxCoordinates[1]), (bbBoxCoordinates[4], bbBoxCoordinates[5])),
                               outline="black", width=2)
                draw.rectangle(((boxCoordinatesGroundTruth[0], boxCoordinatesGroundTruth[1]), (boxCoordinatesGroundTruth[4], boxCoordinatesGroundTruth[5])),
                               outline="red", width=2)

                plt.imshow(imagePIL)
                plt.show(bbox_inches='tight')

    average = 0
    for i in intersectionUnion:
        average += i
    print(intersectionUnion)
    return average / len(intersectionUnion)


def problem2():
    groundTruth1 = [[174.0, 41.5, 417.0, 51.0, 418.0, 106.0, 174.0, 97.0],
                    [234.5, 111.0, 349.0, 110.0, 350.0, 151.0, 235.0, 151.0]]

    groundTruth2 = [[86.0, 300.0, 1335.0, 287.0, 1336.0, 460.0, 86.0, 478.0],
                    [130.0, 570.0, 1045.0, 587.0, 1046.0, 723.0, 140.0, 727.0],
                    [81.0, 915.0, 1007.0, 926.0, 1004.0, 1039.0, 78.0, 1014.0],
                    [108.0, 1129.0, 1450.0, 1151.0, 1446.0, 1370.0, 105.0, 1259.0]]

    image1 = 'data/test1.png'
    image2 = 'data/test2.jpeg'

    print("Localization of the text in the image 1:")
    print("The average difference between the ground truth and the azure localization of image 1: ",
          analyzeImage(image1, groundTruth1))

    print("\nLocalization of the text in the image 2:")
    print("The average difference between the ground truth and the azure localization of image 2: ",
          analyzeImage(image2, groundTruth2))


problem2()


def sharpenImage1(image):
    originalImage = Image.open(image)
    c = ImageEnhance.Contrast(originalImage)
    contrastedImg = c.enhance(3)

    contrastedImg.save("data/contrastedImage1.png")


def sharpenImage2(image):
    originalImage = Image.open(image)
    c = ImageEnhance.Contrast(originalImage)
    contrastedImg = c.enhance(4)

    contrastedImg.save("data/contrastedImage2.jpeg")


def problem3():
    groundTruth1 = [[174.0, 41.5, 417.0, 51.0, 418.0, 106.0, 174.0, 97.0],
                    [234.5, 111.0, 349.0, 110.0, 350.0, 151.0, 235.0, 151.0]]

    groundTruth2 = [[86.0, 300.0, 1335.0, 287.0, 1336.0, 460.0, 86.0, 478.0],
                    [130.0, 570.0, 1045.0, 587.0, 1046.0, 723.0, 140.0, 727.0],
                    [81.0, 915.0, 1007.0, 926.0, 1004.0, 1039.0, 78.0, 1014.0],
                    [108.0, 1129.0, 1450.0, 1151.0, 1446.0, 1370.0, 105.0, 1259.0]]

    image1 = 'data/test1.png'
    image2 = 'data/test2.jpeg'

    image3 = 'data/contrastedImage1.png'
    image4 = 'data/contrastedImage2.jpeg'

    # sharpenImage1(image1)
    # sharpenImage2(image2)

    print("\nLocalization of the text in the contrasted image 1:")
    print("The average difference between the ground truth and the azure localization of enhanced image 1: ",
          analyzeImage(image3, groundTruth1))

    print("\nLocalization of the text in the contrasted image 2:")
    print("The average difference between the ground truth and the azure localization of enhanced image 2: ",
          analyzeImage(image4, groundTruth2))


problem3()
