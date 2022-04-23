import math

import cv2
import numpy as np
from PIL import Image

import helper as hp

DIRECTORY_ALL = "ALL/"


def find_faces_viola(img, detected_faces, correct_bounding_box, landmarks):
    tp = 0
    fp = 0
    fn = 0

    for (column, row, width, height) in detected_faces:
        coordinate = hp.get_four_vertices(column, row, width, height)

        iou = hp.compute_squares_iou(coordinate, correct_bounding_box)

        if iou > 0.5:
            tp += 1
            img = hp.add_bounding_box(coordinate, img, [48, 88, 247])
        else:
            fp += 1
            img = hp.add_bounding_box(coordinate, img, [255, 0, 0])

    if tp == 0:
        fn = 1

    if tp == 0 and fp == 0:
        precision = 0.0
    else:
        precision = round(tp / (tp + fp), 2)
    recall = round(tp / (tp + fn), 2)

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img, hp.create_info_data(tp, fp, fn, precision, recall)


def get_center_eyes(landmarks):
    eye_landmarks = hp.get_eye_landmarks(landmarks)

    center_left = hp.get_center_eye(eye_landmarks[0], eye_landmarks[1])
    center_right = hp.get_center_eye(eye_landmarks[2], eye_landmarks[3])

    return (center_left, center_right)


def get_nose(img, landmarks):
    nose = hp.get_nose_landmark(landmarks)
    img[int(nose[1])][int(nose[0])] = [0, 255, 0]

    return img


def get_center_eyes_float(landmarks):
    eye_landmarks = hp.get_eye_landmarks(landmarks)

    # Print computed center of eyes
    center_left = hp.get_center_eye_float(eye_landmarks[0], eye_landmarks[1])
    center_right = hp.get_center_eye_float(eye_landmarks[2], eye_landmarks[3])

    return center_left, center_right


def detect_viola(img):
    grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    detected_faces = face_cascade.detectMultiScale(grayscale_image)
    return detected_faces


def euclidean_distance(a, b):
    x1 = a[0]
    y1 = a[1]
    x2 = b[0]
    y2 = b[1]
    return math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))


def get_rotate(center_eyes):
    left_eye = center_eyes[0]
    right_eye = center_eyes[1]

    left_y = left_eye[1]
    left_x = left_eye[0]

    right_y = right_eye[1]
    right_x = right_eye[0]

    if left_y > right_y:
        point = (right_x, left_y)
        direction = -1
    else:
        point = (left_x, right_y)
        direction = 1

    a = euclidean_distance((left_x, left_y), point)
    b = euclidean_distance((right_x, right_y), (left_x, left_y))
    c = euclidean_distance((right_x, right_y), point)

    cos_a = (b * b + c * c - a * a) / (2 * b * c)

    angle = np.arccos(cos_a)

    angle = (angle * 180) / math.pi

    if direction == -1:
        angle = 90 - angle

    return direction * angle, (a, b, c)


def cut_image(img, nose, image_size, width, height):
    upper_left_point_x = int(nose[0]) - int(width / 2)
    upper_left_point_y = int(nose[1]) - int(height / 2)

    if upper_left_point_y < 0:
        upper_left_point_y = 0

    if upper_left_point_x < 0:
        upper_left_point_x = 0

    down_left_point_y = upper_left_point_y + (height - 1)
    upper_right_point_x = upper_left_point_x + (width - 1)

    if down_left_point_y > (image_size[1] - 1):
        down_left_point_y = image_size[1] - 1

    if upper_right_point_x > (image_size[0] - 1):
        upper_right_point_x = image_size[0] - 1

    img = img[upper_left_point_y:down_left_point_y, upper_left_point_x:upper_right_point_x]

    return img


def cut_face_according_eyes(img, image_size, center_eyes, center_eyes_distance, rotation_matrix):
    left_center_eye = center_eyes[0]
    right_center_eye = center_eyes[1]

    left_center_eye_new_x, left_center_eye_new_y = cv2.transform(np.array([[[left_center_eye[0], left_center_eye[1]]]]),
                                                                 rotation_matrix).squeeze()

    right_center_eye_new_x, right_center_eye_new_y = cv2.transform(
        np.array([[[right_center_eye[0], right_center_eye[1]]]]),
        rotation_matrix).squeeze()

    upper_y = int(left_center_eye_new_y - center_eyes_distance)
    down_y = int(left_center_eye_new_y + (center_eyes_distance * 2))

    left_x = int(left_center_eye_new_x - center_eyes_distance)
    right_x = int(right_center_eye_new_x + center_eyes_distance)

    img = img[max(0, upper_y):min(down_y, int(image_size[1])), max(0, left_x):min(right_x, int(image_size[0]))]

    return img


def to_mp4(main_directory, directory, name, type, video, landmarks, bounding_box):
    final_name = hp.create_directory_and_get_file_name(main_directory, DIRECTORY_ALL, directory, name, type)
    frameSize = (video.shape[1], video.shape[0])

    all_info_data_viola = []

    for i in range(min(20, video.shape[3])):
        img = video[:, :, :, i]

        landmark_image = landmarks[:, :, i]

        center_eyes = get_center_eyes(landmark_image)
        center_eyes_int = center_eyes
        center_eyes = get_center_eyes_float(landmark_image)
        rot, distances_eyes = get_rotate(center_eyes)

        center = (frameSize[0] / 2, frameSize[1] / 2)
        rot_matrix = cv2.getRotationMatrix2D(center, rot, 1)
        img = cv2.warpAffine(img, rot_matrix, (frameSize[0], frameSize[1]))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cut_face_according_eyes(img, frameSize, center_eyes_int, distances_eyes[1], rot_matrix)
        img = cv2.resize(img, (100, 100))
        cv2.imwrite(hp.create_directory_and_get_file_name_img(main_directory, DIRECTORY_ALL, directory, name, str(i)),
                    img)

    # hp.create_info_file(main_directory, DIRECTORY_ALL, directory, name, type + "_viola", all_info_data_viola)
