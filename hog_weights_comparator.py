import os
from PIL import Image
from matplotlib import pyplot as plt
from numpy import asarray
import cv2
from scipy.spatial import distance
import hog as h
import random
from sklearn.decomposition import PCA

DIRECTORY_TRUE = "TRUE/ALL/"
DIRECTORY_FALSE = "FALSE/ALL/"


def get_directory(path):
    directory = []
    for filename in os.listdir(path):
        directory.append(path + filename + "/")
    return directory


def get_files_without_average(path, random_image):
    files = []
    for filename in os.listdir(path):
        if "AVERAGE" not in filename:
            files.append(path + filename)

    if random_image:
        random_file = random.randint(0, len(files) - 1)
        return [files[random_file]]

    return files


def get_images_without_average(directory, random_image):
    img_dict = {}

    image_files = get_files_without_average(directory, random_image)

    for i in range(0, min(len(image_files), len(image_files))):
        image = image_files[i]
        img = Image.open(image)
        img = asarray(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.flatten()
        img_dict[image] = img

    return img_dict


def compute_distances(pair1_videos, pair2_videos, image_weights, weights):
    distances = []

    for file_name_1, img1 in pair1_videos.items():
        for file_name_2, img2 in pair2_videos.items():
            img1_weight = image_weights[file_name_1]
            img2_weight = image_weights[file_name_2]
            distances.append(distance.euclidean(img1_weight, img2_weight))

    return distances


def compare_without_average(directories, image_weights, weights, random_image):
    all_distances = []

    for pair_directory in get_directory(directories):
        pairs_directory = []
        for image_directory in get_directory(pair_directory):
            pairs_directory.append(image_directory)

        pair1 = pairs_directory[0]
        pair2 = pairs_directory[1]
        pair1_videos = get_images_without_average(pair1, random_image)
        pair2_videos = get_images_without_average(pair2, random_image)
        all_distances.extend(compute_distances(pair1_videos, pair2_videos, image_weights, weights))

    return all_distances


# AVERAGE

def get_files_average(path):
    files = []
    for filename in os.listdir(path):
        if "AVERAGE" in filename:
            files.append(path + filename)

    for i in files:
        print(i)
    return files


def get_images_average(directory):
    img_dict = {}

    image_files = get_files_average(directory)

    for i in range(0, min(10, len(image_files))):
        image = image_files[i]
        img = Image.open(image)
        img = asarray(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.flatten()
        img_dict[image] = img

    return img_dict


def compare_average(directories, image_weights, weights):
    all_distances = []

    for pair_directory in get_directory(directories):
        pairs_directory = []
        for image_directory in get_directory(pair_directory):
            pairs_directory.append(image_directory)

        pair1 = pairs_directory[0]
        pair2 = pairs_directory[1]
        pair1_videos = get_images_average(pair1)
        pair2_videos = get_images_average(pair2)
        all_distances.extend(compute_distances(pair1_videos, pair2_videos, image_weights, weights))

    return all_distances


def distance_without_average(random_image):
    image_weights, weights = h.get_hog_weights()

    distances_true = compare_without_average(DIRECTORY_TRUE, image_weights, weights, random_image)
    distances_false = compare_without_average(DIRECTORY_FALSE, image_weights, weights, random_image)

    return distances_true, distances_false


def distance_average():
    image_weights, weights = h.get_hog_weights()

    distances_true = compare_average(DIRECTORY_TRUE, image_weights, weights)
    distances_false = compare_average(DIRECTORY_FALSE, image_weights, weights)

    return distances_true, distances_false


def compute_roc_tpr_fpr(distances_true, trashold):
    tpr_or_fpr = 0

    for dist in distances_true:
        if dist < trashold:
            tpr_or_fpr += 1

    return tpr_or_fpr / len(distances_true)


def generate_trasholds(min, max, step):
    trasholds = []

    min = min + step

    while min < max:
        trasholds.append(min)
        min += step

    return trasholds


def compute_roc_data(distances):
    print("\n\n\n\n\n")
    for u in distances:
        print(u)
    print("\n\n\n\n\n")
    trasholds = []
    tprs = []
    fprs = []

    all = []
    all.extend(distances[0])
    all.extend(distances[1])

    # min distance
    trashold_for_min = min(all)
    trasholds.append(trashold_for_min)
    tprs.append(compute_roc_tpr_fpr(distances[0], trashold_for_min))
    fprs.append(compute_roc_tpr_fpr(distances[1], trashold_for_min))

    a = generate_trasholds(min(all), max(all), 0.0005)
    print("Size", len(a))
    print("trash", a)

    for trashold in a:
        trasholds.append(trashold)
        tprs.append(compute_roc_tpr_fpr(distances[0], trashold))
        fprs.append(compute_roc_tpr_fpr(distances[1], trashold))

    #     max distance
    trashold_for_max = max(all)
    trasholds.append(trashold_for_max)
    tprs.append(compute_roc_tpr_fpr(distances[0], trashold_for_max))
    fprs.append(compute_roc_tpr_fpr(distances[1], trashold_for_max))

    print("len", len(trasholds))
    return trasholds, tprs, fprs


def show_roc(roc_data_all, roc_data_average, roc_data_random):
    plt.figure()
    lw = 2
    # ALL
    plt.plot(
        roc_data_all[2],
        roc_data_all[1],
        color="darkorange",
        lw=lw,
        # label="ROC curve (area = %0.2f)" % roc_auc[2],
    )

    # AVERAGE
    plt.plot(
        roc_data_average[1],
        roc_data_average[2],
        color="navy",
        lw=lw,
        # label="ROC curve (area = %0.2f)" % roc_auc[2],
    )

    # RANDOM
    plt.plot(
        roc_data_random[1],
        roc_data_random[2],
        color="red",
        lw=lw,
        # label="ROC curve (area = %0.2f)" % roc_auc[2],
    )

    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")
    plt.show()


all = compute_roc_data(distance_without_average(False))
average = compute_roc_data(distance_average())
random_image = compute_roc_data(distance_without_average(True))


show_roc(all, average, random_image)
