import math
import os
from PIL import Image
from matplotlib import pyplot as plt
from numpy import asarray
import cv2
from scipy.spatial import distance
import pca as p
import random
import seaborn as sns
import time

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

    # for i in files:
    #     print(i)

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
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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


def compute_average_distances(distances):
    sum = 0

    for i in distances:
        sum += i

    return sum / len(distances)


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
        dist = compute_distances(pair1_videos, pair2_videos, image_weights, weights)
        # all_distances.extend(compute_distances(pair1_videos, pair2_videos, image_weights, weights))
        all_distances.append(compute_average_distances(dist))

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
    image_weights, weights = p.get_pca_weights()

    distances_true = compare_without_average(DIRECTORY_TRUE, image_weights, weights, random_image)
    distances_false = compare_without_average(DIRECTORY_FALSE, image_weights, weights, random_image)

    return distances_true, distances_false


def distance_average():
    image_weights, weights = p.get_pca_weights()

    distances_true = compare_average(DIRECTORY_TRUE, image_weights, weights)
    distances_false = compare_average(DIRECTORY_FALSE, image_weights, weights)

    return distances_true, distances_false


def compute_roc_tpr_fpr(distances_true, trashold, is_avg):
    tpr_or_fpr = 0

    for dist in distances_true:
        if dist < trashold:
            tpr_or_fpr += 1

    if is_avg:
        return tpr_or_fpr / len(distances_true)
    else:
        return tpr_or_fpr


def compute_roc_data(distances):
    trasholds = []
    tprs = []
    fprs = []

    all = []
    all.extend(distances[0])
    all.extend(distances[1])

    best_distance = 0
    tp = 0
    fp = 0

    # min distance
    trashold_for_min = min(all)
    trasholds.append(trashold_for_min)
    tpr = compute_roc_tpr_fpr(distances[0], trashold_for_min, True)
    fpr = compute_roc_tpr_fpr(distances[1], trashold_for_min, True)
    tprs.append(tpr)
    fprs.append(fpr)

    best_distance = math.dist([tpr, fpr], [1, 0])
    tp = compute_roc_tpr_fpr(distances[0], trashold_for_min, False)
    fp = compute_roc_tpr_fpr(distances[1], trashold_for_min, False)

    iteration = 20
    for trashold in range(int(min(all) + iteration), int(max(all)), iteration):
        trasholds.append(trashold)
        tpr = compute_roc_tpr_fpr(distances[0], trashold, True)
        fpr = compute_roc_tpr_fpr(distances[1], trashold, True)
        tprs.append(tpr)
        fprs.append(fpr)
        dst = math.dist([tpr, fpr], [1, 0])
        if dst < best_distance:
            best_distance = dst
            tp = compute_roc_tpr_fpr(distances[0], trashold, False)
            fp = compute_roc_tpr_fpr(distances[1], trashold, False)

    # max distance
    trashold_for_max = max(all)
    trasholds.append(trashold_for_max)
    tpr = compute_roc_tpr_fpr(distances[0], trashold_for_max, True)
    fpr = compute_roc_tpr_fpr(distances[1], trashold_for_max, True)
    tprs.append(tpr)
    fprs.append(fpr)
    dst = math.dist([tpr, fpr], [1, 0])
    if dst < best_distance:
        best_distance = dst
        tp = compute_roc_tpr_fpr(distances[0], trashold_for_max, False)
        fp = compute_roc_tpr_fpr(distances[1], trashold_for_max, False)

    print("len", len(trasholds))

    tn = len(distances[1]) - fp
    fn = len(distances[0]) - tp
    return trasholds, tprs, fprs, (tp, tn, fp, fn)


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
        roc_data_average[2],
        roc_data_average[1],
        color="navy",
        lw=lw,
        # label="ROC curve (area = %0.2f)" % roc_auc[2],
    )

    # RANDOM
    plt.plot(
        roc_data_random[2],
        roc_data_random[1],
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


def confusion_matrix(data):
    a = sns.heatmap([[data[0], data[3]], [data[2], data[1]]], annot=True, cmap='Greens', fmt='g')

    a.set_xlabel('Predicted Values')
    a.set_ylabel('Actual Values')

    a.xaxis.set_ticklabels(['True', 'False'])
    a.yaxis.set_ticklabels(['True', 'False'])

    plt.show()


all = compute_roc_data(distance_without_average(False))
average = compute_roc_data(distance_average())
random_image = compute_roc_data(distance_without_average(True))

start_time_all = time.perf_counter()
show_roc(all, average, random_image)
confusion_matrix(all[3])
confusion_matrix(average[3])
confusion_matrix(random_image[3])
elapsed_time_all = time.perf_counter() - start_time_all
print("Celkový čas: ", elapsed_time_all)
