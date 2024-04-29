"""
2. For images containing bicycles:

a. to automatically locate the bicycles in these images and highlight the frames that frame the bicycles

b. to label (without the help of AI algorithms) these images with borders that frame the bikes as accurately as possible
Which task takes longer (the one from point a or the one from point b)?

c. to determine the performance of the algorithm from point a taking into account the labels made at point b
(at least 2 metrics will be used).
"""

import os

import matplotlib.pyplot as plt
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
from sklearn.metrics import precision_score, average_precision_score

subscription_key = os.environ["VISION_KEY"]
endpoint = os.environ["VISION_ENDPOINT"]
computerVision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))


def intersection_over_union(bboxA, bboxB):
    boxA = [bboxA[0], bboxA[1], bboxA[2], bboxA[3]]
    boxB = [bboxB[0], bboxB[1], bboxB[2], bboxB[3]]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    intersectionArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    bboxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    bboxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = intersectionArea / float(bboxAArea + bboxBArea - intersectionArea)

    return iou


def detection_in_ground_truth(predicted, groundTruth):
    iou = intersection_over_union(predicted, groundTruth)
    if iou >= 0.5:
        return 1
    return 0


def detection_in_ground_truths(predicted, groundTruths):
    ious = []
    for gt in groundTruths:
        iou = intersection_over_union(predicted, gt)
        if iou >= 0.5:
            ious.append(1)
        else:
            ious.append(0)
    return ious


def imagesOneGroundTruth(filename, groundTruth1):
    predicted_bike1_bb = []

    img = open(filename, "rb")
    result = computerVision_client.analyze_image_in_stream(img, visual_features=[VisualFeatureTypes.objects])
    for ob in result.objects:
        if ob.object_property == "bicycle" or ob.object_property == "cycle":
            predicted_bike1_bb = [ob.rectangle.x, ob.rectangle.y, ob.rectangle.x + ob.rectangle.w,
                                  ob.rectangle.y + ob.rectangle.h]

    err = 0
    for v in zip(predicted_bike1_bb, groundTruth1):
        err = err + (v[0] - v[1]) ** 2
    err /= 4

    iou = intersection_over_union(predicted_bike1_bb, groundTruth1)
    detection_iou = detection_in_ground_truth(predicted_bike1_bb, groundTruth1)
    precision = precision_score(predicted_bike1_bb, groundTruth1, average='micro')

    print("Ground Truth Bounding Box: ", groundTruth1)
    print("Predicted Bounding Box: ", predicted_bike1_bb)
    print("Detection Error: ", err)
    print('Intersection Over Union: ', iou)
    print('Detection Intersection over Union: ', detection_iou)
    print(f'Precision: {precision:.2f}')

    im = plt.imread(filename)
    fig = plt.imshow(im)
    fig.axes.add_patch(
        plt.Rectangle(xy=(groundTruth1[0], groundTruth1[1]), width=groundTruth1[2] - groundTruth1[0],
                      height=groundTruth1[3] - groundTruth1[1],
                      fill=False,
                      color="lightgreen", linewidth=2))
    fig.axes.add_patch(
        plt.Rectangle(xy=(predicted_bike1_bb[0], predicted_bike1_bb[1]),
                      width=predicted_bike1_bb[2] - predicted_bike1_bb[0],
                      height=predicted_bike1_bb[3] - predicted_bike1_bb[1], fill=False, color="green",
                      linewidth=2))
    plt.show()
    return [iou, detection_iou, precision]


def imagesTwoGroundTruths(filename, groundTruth1, groundTruth2):
    predicted_bike1_bb = []

    img = open(filename, "rb")
    result = computerVision_client.analyze_image_in_stream(img, visual_features=[VisualFeatureTypes.objects])
    for ob in result.objects:
        if ob.object_property == "bicycle" or ob.object_property == "cycle":
            predicted_bike1_bb = [ob.rectangle.x, ob.rectangle.y, ob.rectangle.x + ob.rectangle.w,
                                  ob.rectangle.y + ob.rectangle.h]

    err = 0
    for v in zip(predicted_bike1_bb, groundTruth1):
        err = err + (v[0] - v[1]) ** 2
    err /= 4

    iou = intersection_over_union(predicted_bike1_bb, groundTruth1)
    detection_iou = detection_in_ground_truths(predicted_bike1_bb, [groundTruth1, groundTruth2])
    precision1 = precision_score(predicted_bike1_bb, groundTruth1, average='micro')
    precision2 = precision_score(predicted_bike1_bb, groundTruth2, average='micro')
    precision = [precision1, precision2]

    print("Ground Truth Bounding Box: ", groundTruth1)
    print("Predicted Bounding Box: ", predicted_bike1_bb)
    print("Detection Error: ", err)
    print('Intersection Over Union: ', iou)
    print('Detection Intersection over Union: ', detection_iou)
    print(f'Precision: {precision[0]:.2f}, {precision[1]:.2f}')

    im = plt.imread(filename)
    fig = plt.imshow(im)
    fig.axes.add_patch(
        plt.Rectangle(xy=(groundTruth1[0], groundTruth1[1]), width=groundTruth1[2] - groundTruth1[0],
                      height=groundTruth1[3] - groundTruth1[1],
                      fill=False,
                      color="lightgreen", linewidth=2))
    fig.axes.add_patch(
        plt.Rectangle(xy=(groundTruth2[0], groundTruth2[1]), width=groundTruth2[2] - groundTruth2[0],
                      height=groundTruth2[3] - groundTruth2[1],
                      fill=False,
                      color="lightgreen", linewidth=2))
    fig.axes.add_patch(
        plt.Rectangle(xy=(predicted_bike1_bb[0], predicted_bike1_bb[1]),
                      width=predicted_bike1_bb[2] - predicted_bike1_bb[0],
                      height=predicted_bike1_bb[3] - predicted_bike1_bb[1], fill=False, color="green",
                      linewidth=2))
    plt.show()
    return [iou, detection_iou, precision]


def main():
    ious = []
    detection_ious = []
    precisions = []

    print("Image bike1.jpg: ")
    iou = imagesOneGroundTruth('data/bike1.jpg', [4.0, 30.0, 409.0, 405.0])
    ious.append(iou[0])
    detection_ious.append(iou[1])
    precisions.append(iou[2])

    print("\nImage bike02.jpg: ")
    iou = imagesOneGroundTruth('data/bike02.jpg', [15.0, 88.0, 380.0, 322.0])
    ious.append(iou[0])
    detection_ious.append(iou[1])
    precisions.append(iou[2])

    print("\nImage bike03.jpg: ")
    iou = imagesTwoGroundTruths('data/bike03.jpg', [155.0, 140.0, 344.0, 407.0], [62.0, 141.0, 197.0, 390.0])
    ious.append(iou[0])
    detection_ious.append(iou[1][0])
    detection_ious.append(iou[1][1])
    precisions.append(iou[2][0])
    precisions.append(iou[2][0])

    print("\nImage bike04.jpg: ")
    iou = imagesOneGroundTruth('data/bike04.jpg', [1.0, 1.0, 415.0, 414.0])
    ious.append(iou[0])
    detection_ious.append(iou[1])
    precisions.append(iou[2])

    print("\nImage bike05.jpg: ")
    iou = imagesOneGroundTruth('data/bike05.jpg', [67.0, 50.0, 355.0, 345.0])
    ious.append(iou[0])
    detection_ious.append(iou[1])
    precisions.append(iou[2])

    print("\nImage bike06.jpg: ")
    iou = imagesTwoGroundTruths('data/bike06.jpg', [155.0, 140.0, 344.0, 407.0], [62.0, 141.0, 197.0, 390.0])
    ious.append(iou[0])
    detection_ious.append(iou[1][0])
    detection_ious.append(iou[1][1])
    precisions.append(iou[2][0])
    precisions.append(iou[2][1])

    print("\nImage bike07.jpg: ")
    iou = imagesOneGroundTruth('data/bike07.jpg', [58.0, 202.0, 300.0, 415.0])
    ious.append(iou[0])
    detection_ious.append(iou[1])
    precisions.append(iou[2])

    print("\nImage bike10.jpg: ")
    iou = imagesOneGroundTruth('data/bike10.jpg', [140.0, 123.0, 375.0, 406.0])
    ious.append(iou[0])
    detection_ious.append(iou[1])
    precisions.append(iou[2])

    IoU = 0
    for i in ious:
        IoU += i
    print('\nAverage IOU: ', IoU / len(ious))

    averageIOU = 0
    for di in detection_ious:
        averageIOU += di
    print('\nAverage Detection IOU: ', averageIOU / len(detection_ious))

    average_precision = average_precision_score(detection_ious, precisions, average='micro')
    print('\nAverage Precision: ', average_precision)


main()
