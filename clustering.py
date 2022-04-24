import random

import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import make_swiss_roll
from sklearn.cluster import KMeans
from PIL import Image
from numpy import asarray
import cv2


def from_string_array_to_array(string_array):
    arr = []
    string_array = string_array[1: len(string_array) - 2]

    split_string_array = string_array.split("_")

    for i in split_string_array:
        num = i.strip()
        arr.append(float(num))

    return arr


def get_pca_data_for_clustering(weights):
    pca = PCA(n_components=3)
    fit = pca.fit(weights)
    new_weights = pca.transform(weights)

    return new_weights


def get_weights_from_csv(file_name):
    image_names = []
    weights = []

    file = open(file_name, "r")
    lines = file.readlines()

    for i in lines:
        line_split = i.split(",")
        name = line_split[0].strip()
        string_array = line_split[1]
        image_names.append(name)
        weights.append(from_string_array_to_array(string_array))

    return image_names, get_pca_data_for_clustering(weights)


def change_name(name):
    name_split = name.split("/")

    return name_split[len(name_split) - 1]


def get_image(image_name):
    img = Image.open(image_name)
    img = asarray(img)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = img.flatten()

    return img


# def get_zeros_and_ones(arr):
#     zeros = []
#     ones = []
#
#     for i in arr:
#         if i == 0:
#             zeros.append(i)
#         if i == 1:
#             ones.append(i)
#
#     print(len(zeros) + len((ones)))
#     print(len(arr))
#
#     return zeros, ones


def get_random_image(images, count):
    choice_images = []

    new_arr = images.copy()
    new_arr = new_arr.tolist()

    for i in range(min(count, len(images))):
        index = random.randint(0, len(new_arr))
        choice_images.append(new_arr[index])
        new_arr.pop(index)

    return choice_images


def show_mages(images):
    choice_images = get_random_image(images, 12)

    figure = plt.figure(figsize=(10, 7))

    for i in range(len(choice_images)):
        image = get_image(choice_images[i])
        figure.add_subplot(4, 4, i + 1)
        plt.imshow(image)
        plt.axis('off')
        plt.title(change_name(choice_images[i]))


def clustering(data):
    names, weights = data

    kmeans = KMeans(n_clusters=2)  # Number of clusters == 3
    kmeans = kmeans.fit(weights)  # Fitting the input data
    labels = kmeans.predict(weights)  # Getting the cluster labels
    centroids = kmeans.cluster_centers_  # Centroid values
    print("Centroids are:", centroids)  # From sci-kit learn

    print("Labels", labels)

    # zeros, ones = get_zeros_and_ones(labels)

    names = np.array(names)
    x = np.array(labels == 0)
    y = np.array(labels == 1)

    cluster1 = names[x]
    cluster2 = names[y]

    show_mages(cluster1)
    show_mages(cluster2)

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111, projection='3d')

    x = np.array(labels == 0)
    y = np.array(labels == 1)
    z = np.array(labels == 2)

    ax.scatter(weights[x][:, 0], weights[x][:, 1], weights[x][:, 2], color='red')
    ax.scatter(weights[y][:, 0], weights[y][:, 1], weights[y][:, 2], color='blue')
    ax.scatter(weights[z][:, 0], weights[z][:, 1], weights[z][:, 2], color='yellow')
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
               marker='x', s=169, linewidths=10,
               color='black', zorder=50)

    plt.show()


data = get_weights_from_csv("images_weights.csv")
clustering(data)
