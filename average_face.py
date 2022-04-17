import os
from PIL import Image
import numpy as np
from numpy import asarray
import cv2

DIRECTORY_TRUE = "TRUE/ALL/"
DIRECTORY_FALSE = "FALSE/ALL/"


def get_directory(path):
    directory = []
    for filename in os.listdir(path):
        directory.append(path + filename + "/")
    return directory


def get_files(path):
    files = []
    for filename in os.listdir(path):
        files.append(path + filename)
    return files


def average_image(directory):
    for pair_directory in get_directory(directory):
        for image_directory in get_directory(pair_directory):
            images = []
            for image in get_files(image_directory):
                img = Image.open(image)
                img = asarray(img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)

            average_image = np.mean(([frame for frame in images]), axis=0)
            cv2.imwrite(image_directory + "AVERAGE.jpg", average_image)


average_image(DIRECTORY_TRUE)
average_image(DIRECTORY_FALSE)
