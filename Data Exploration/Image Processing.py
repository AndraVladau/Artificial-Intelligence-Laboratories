"""
2. Several images are given (saved in the "data/images" folder). Is required:

- to visualize one of the images
- if the images are not the same size, resize them all to 128 x 128 pixels and view the images in a tabular frame.
- to transform the images into gray-levels format and visualize them
- blur an image and display it in "before-after" format
- to identify the edges of an image and to display them in "before-after" format
"""


import matplotlib.pyplot as plt
from skimage import exposure, io, transform
from skimage.feature import hog
import os
from PIL import Image, ImageFilter, ImageOps


def visualiseImageColor():
    """
    Visualise an image with color
    """
    img = io.imread('data/images/YOLO.jpg')

    plt.imshow(img)
    plt.title('Colored images')
    plt.axis('off')

    plt.show()


# visualiseImageColor()


def sizesImage():
    sizes = []
    for image in os.listdir('data/images'):
        img = io.imread('data/images/' + image, as_gray=True)
        sizes.append(img.shape)
    return set(sizes)


def resizeImages():
    sizes = sizesImage()
    if len(sizes) == 1:
        print("All the images have the same size!")
    else:
        resized_images = []
        for image in os.listdir('data/images'):
            img = io.imread('data/images/' + image)
            resizedImg = transform.resize(img, (128, 128))
            resized_images.append(resizedImg)
        fig, axes = plt.subplots(len(resized_images) // 3 + 1, 3, figsize=(15, 15))

        for i, ax in enumerate(axes.flat):
            if i < len(resized_images):
                ax.imshow(resized_images[i], cmap='gray')
                ax.axis('off')
            else:
                ax.axis('off')

        plt.show()
        return


resizeImages()


def visualiseImagesInGray():
    """
    Visualise all images in gray colors
    """

    for image in os.listdir('data/images'):
        img = io.imread('data/images/' + image, as_gray=True)

        hogDescriptor, hogView = hog(img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1),
                                     visualize=True, channel_axis=None)

        print("HOG descriptor values: ", hogDescriptor)
        print("HOG descriptor no of values: ", len(hogDescriptor))
        print("HOG descriptor shape: ", hogDescriptor.shape)
        print("Image shape: ", img.shape)

        fig, (ax1) = plt.subplots(1, 1, figsize=(16, 8), sharex=True, sharey=True)
        ax1.imshow(img, cmap=plt.cm.gray)
        ax1.set_title('Input image')

        plt.show()


# visualiseImagesInGray(i)


def blurImages():
    """
    Blur all the images
    """
    for image in os.listdir('data/images'):
        img = Image.open('data/images/' + image)

        blurred_image = img.filter(ImageFilter.GaussianBlur(radius=3))

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.title('Before')
        plt.imshow(img)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title('After')
        plt.imshow(blurred_image)
        plt.axis('off')

        plt.tight_layout()
        plt.show()


# blurImages()


def findEdgesImages():
    """
    Show all the edges of the images
    """
    for i in os.listdir('data/images'):
        img = Image.open('data/images/' + i)

        image = img.convert('L')
        edges_image = image.filter(ImageFilter.FIND_EDGES)

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 2, 1)
        plt.title('Original Image')
        plt.imshow(img)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title('Edges Detected Image')
        plt.imshow(edges_image)
        plt.axis('off')

        plt.tight_layout()
        plt.show()


findEdgesImages()
