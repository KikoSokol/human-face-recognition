import os
import csv
import math

import numpy as np

HEADER = ["TP", "FP", "FN", "PRECISION", "RECALL"]


def add_bounding_box(bounding_box, image, color):
    for up in range(int(bounding_box[0][1]), int(bounding_box[1][1]) + 1):
        image[up][int(bounding_box[0][0])] = np.array(color)

    for down in range(int(bounding_box[2][1]), int(bounding_box[3][1]) + 1):
        image[down][int(bounding_box[2][0])] = np.array(color)

    for left in range(int(bounding_box[0][0]), int(bounding_box[2][0]) + 1):
        image[int(bounding_box[0][1])][left] = np.array(color)

    for right in range(int(bounding_box[1][0]), int(bounding_box[3][0]) + 1):
        image[int(bounding_box[1][1])][right] = np.array(color)

    return image


def add_landmarks(landmarks, image, color):
    for i in landmarks:
        image[int(i[1])][int(i[0])] = np.array(color)

    return image


def get_eye_landmarks(landmarks):
    lands = []
    lands.append(landmarks[36])
    lands.append(landmarks[39])
    lands.append(landmarks[42])
    lands.append(landmarks[45])

    return lands


def get_nose_landmark(landmarks):
    return landmarks[29]


def get_center_eye(first, second):
    x_tmp = abs(first[1] - second[1]) / 2
    if second[1] > first[1]:
        x_center = second[1] - x_tmp
    else:
        x_center = first[1] - x_tmp

    y_tmp = abs(first[0] - second[0]) / 2

    if second[0] > first[0]:
        y_center = second[0] - y_tmp
    else:
        y_center = first[0] - y_tmp

    return int(y_center), int(x_center)


def get_center_eye_float(first, second):
    x_tmp = abs(first[1] - second[1]) / 2
    if second[1] > first[1]:
        x_center = second[1] - x_tmp
    else:
        x_center = first[1] - x_tmp

    y_tmp = abs(first[0] - second[0]) / 2

    if second[0] > first[0]:
        y_center = second[0] - y_tmp
    else:
        y_center = first[0] - y_tmp

    return float(y_center), float(x_center)


def create_dot_big_dot(dot):
    pixels = []
    pixels.append(dot)
    pixels.append((dot[0] + 1, dot[1]))
    pixels.append((dot[0] + 2, dot[1]))
    pixels.append((dot[0] - 1, dot[1]))
    pixels.append((dot[0] - 2, dot[1]))

    pixels.append((dot[0], dot[1] + 1))
    pixels.append((dot[0], dot[1] + 2))
    pixels.append((dot[0], dot[1] - 1))
    pixels.append((dot[0], dot[1] - 2))

    return pixels


def create_folder(name):
    if os.path.isdir(name) is True:
        return None
    else:
        os.mkdir(name[0: len(name) - 1])


def get_four_vertices(column, row, width, height):
    x0 = [column, row]
    x1 = [column, row + height]
    x2 = [column + width, row]
    x3 = [column + width, row + height]

    return [x0, x1, x2, x3]


def distance_points(dot1, dot2):
    x = pow(abs(dot1[1] - dot2[1]), 2)
    y = pow(abs(dot1[0] - dot2[0]), 2)

    return math.sqrt(x + y)


def compute_iou(found_face_square, correct_square):
    x_inter1 = max(found_face_square[0], correct_square[0])
    y_inter1 = max(found_face_square[1], correct_square[1])
    x_inter_2 = min(found_face_square[2], correct_square[2])
    y_inter_2 = min(found_face_square[3], correct_square[3])

    width_inter = abs(x_inter_2 - x_inter1)
    height_inter = abs(y_inter_2 - y_inter1)
    area_inter = width_inter * height_inter
    width_box1 = abs(found_face_square[2] - found_face_square[0])
    height_box1 = abs(found_face_square[3] - found_face_square[1])
    width_box2 = abs(correct_square[2] - correct_square[0])
    height_box2 = abs(correct_square[3] - correct_square[1])

    area_box1 = width_box1 * height_box1
    area_box2 = width_box2 * height_box2
    area_union = area_box1 + area_box2 - area_inter

    return area_inter / area_union


def get_cord(square_vertices):
    return square_vertices[0][1], square_vertices[0][0], square_vertices[3][1], square_vertices[3][0]


def compute_squares_iou(vertices_found_face_square, vertices_correct_square):
    return compute_iou(get_cord(vertices_found_face_square), get_cord(vertices_correct_square))


def create_info_data(tp, fp, fn, precision, recall):
    return [tp, fp, fn, precision, recall]


def create_info_file_name(main_directory, category_directory, directory, name, type, suffix):
    dir = main_directory + category_directory
    create_folder(dir)

    directory = dir + directory
    create_folder(directory)

    directory2 = directory + name + "/"
    create_folder(directory2)

    return directory2 + name + "_" + type + suffix


def create_directory_and_get_file_name(main_directory, category_directory, directory, name, type):
    return create_info_file_name(main_directory, category_directory, directory, name, type, ".mp4")


def create_directory_and_get_file_name_img(main_directory, category_directory, directory, name, type):
    return create_info_file_name(main_directory, category_directory, directory, name, type, ".jpg")


def create_info_file(main_directory, category_directory, directory, name, type, data):
    file_name = create_info_file_name(main_directory, category_directory, directory, name, type, ".csv")

    file = open(file_name, "w", newline='')
    write = csv.writer(file)

    write.writerow(HEADER)
    write.writerows(data)
    file.close()


def create_summary_info(main_directory, category_directory, directory, name, type, data):
    file_name = create_info_file_name(main_directory, category_directory, directory, name, type, ".csv")

    file = open(file_name, "w", newline='')
    write = csv.writer(file)

    write.writerow(["nazov", "VIOLA-PRECISION", "VIOLA_PRECISION_AVG", "VIOLA-RECALL", "VIOLA_RECALL_AVG",
                    "CNN-PRECISION", "CNN_PRECISION_AVG", "CNN-RECALL", "CNN_RECALL_AVG",
                    "MSE-LE", "MSE-PE",
                    "WORST-FRAME-LE", "WORST-FRAME-LE-VALUE", "WORST-FRAME-PE", "WORST-FRAME-PE-VALUE"])
    write.writerows(data)
    file.close()


def save_image_weight_to_csv(file_name, image_weights):
    file = open(file_name, "w", newline='')
    write = csv.writer(file)

    for name, weight in image_weights.items():
        if "AVERAGE" not in name:
            write.writerow([name, weight])

    file.close()
