import os
from PIL import Image
from numpy import asarray
import cv2
import matplotlib.pyplot as plt
from skimage.io import imread

from skimage.feature import hog
from skimage import data, exposure

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


def compute_hog_weight_for_video(directory, pca_dict):
    for image in get_files(directory):
        # img = Image.open(image)
        # img = asarray(img)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = img.flatten()
        # weight = hog(img)[0]
        # pca_dict[image] = weight

        img = imread(image, as_gray=True)
        weight = hog(img)
        pca_dict[image] = weight[0]


def do_hog(directories):
    pca_dict = {}

    for main_dir in directories:
        for pair_directory in get_directory(main_dir):
            pairs_directory = []
            for image_directory in get_directory(pair_directory):
                pairs_directory.append(image_directory)

            pair1 = pairs_directory[0]
            pair2 = pairs_directory[1]

            compute_hog_weight_for_video(pair1, pca_dict)
            compute_hog_weight_for_video(pair2, pca_dict)

    return pca_dict


def get_hog_weights():
    directories = [DIRECTORY_TRUE, DIRECTORY_FALSE]

    return do_hog(directories)
