import os
from PIL import Image
import numpy as np
from numpy import asarray
import cv2
from sklearn.decomposition import PCA

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


def get_pca_from_image_directory(directory, pair_img, pair_labels, label):
    for image in get_files(directory):
        img = Image.open(image)
        img = asarray(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pair_img.append(img.flatten())
        pair_labels.append(label)


def get_images(directory):
    images = {}
    for image in get_files(directory):
        img = Image.open(image)
        img = asarray(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images[image] = img

    return images


def get_image(name):
    img = Image.open(name)
    img = asarray(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def get_data_for_pca(directory):
    pair_one_img = []
    pair_one_labels = []
    pair_one_dict_names = {}

    pair_two_img = []
    pair_two_labels = []
    pair_two_dict_names = {}

    label = 0

    for pair_directory in get_directory(directory):
        pairs_directory = []
        for image_directory in get_directory(pair_directory):
            pairs_directory.append(image_directory)

        pair1 = pairs_directory[0]
        pair2 = pairs_directory[1]

        get_pca_from_image_directory(pair1, pair_one_img, pair_one_labels, label)
        get_pca_from_image_directory(pair2, pair_two_img, pair_two_labels, label)
        pair_one_dict_names[label] = pair1
        pair_two_dict_names[label] = pair2

        label += 1

    return (pair_one_img, pair_one_labels, pair_one_dict_names), (pair_two_img, pair_two_labels, pair_two_dict_names)


def pca(data):
    images = data[0]
    labels = data[1]
    dict_names = data[2]
    pca = PCA().fit(images)
    print(pca.explained_variance_ratio_)

    n_components = 50
    eigenfaces = pca.components_[:n_components]

    weights = eigenfaces @ (images - pca.mean_).T

    example_image = get_image("TRUE/ALL/Linda_Dano_2-Linda_Dano_3/Linda_Dano_3/Linda_Dano_3_0.jpg")
    query = example_image.reshape(1, -1)
    query_weight = eigenfaces @ (query - pca.mean_).T
    euclidean_distance = np.linalg.norm(weights - query_weight, axis=0)
    best_match = np.argmin(euclidean_distance)
    print("best_match: ", best_match)
    print("Best match %s with Euclidean distance %f" % (dict_names[labels[best_match]], euclidean_distance[best_match]))


one_true, two_true = get_data_for_pca(DIRECTORY_TRUE)

pca(one_true)
