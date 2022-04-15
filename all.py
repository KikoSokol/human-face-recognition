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


def get_center_eyes(img, landmarks):
    eye_landmarks = hp.get_eye_landmarks(landmarks)

    # Print dot in edges of eyes
    for eye_land in eye_landmarks:
        for dot in hp.create_dot_big_dot(eye_land):
            img[int(dot[1])][int(dot[0])] = [0, 0, 255]

    # Print computed center of eyes
    center_left = hp.get_center_eye(eye_landmarks[0], eye_landmarks[1])
    center_right = hp.get_center_eye(eye_landmarks[2], eye_landmarks[3])

    for i in hp.create_dot_big_dot(center_left):
        img[i[1]][i[0]] = [0, 255, 0]

    for i in hp.create_dot_big_dot(center_right):
        img[i[1]][i[0]] = [0, 255, 0]

    return img, (center_left, center_right)


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

    a = abs(right_y - left_y)
    b = abs(right_x - left_x)
    c = math.sqrt((a*a) + (b*b))

    cos_a = (b * b + c * c - a * a) / (2 * b * c)

    angle = np.arccos(cos_a)

    angle = (angle * 180) / math.pi

    if direction == -1:
        angle = -angle

    return angle


def to_mp4(main_directory, directory, name, type, video, landmarks, bounding_box):
    final_name = hp.create_directory_and_get_file_name(main_directory, DIRECTORY_ALL, directory, name, type)
    frameSize = (video.shape[1], video.shape[0])

    # out = cv2.VideoWriter(final_name, cv2.VideoWriter_fourcc(*'mp4v'), 25, frameSize, True)

    all_info_data_viola = []

    all_viola_precision = []
    all_viola_recall = []

    for i in range(video.shape[3]):
        img = video[:, :, :, i]

        landmark_image = landmarks[:, :, i]
        bounding_box_image = bounding_box[:, :, i]

        faces_viola = detect_viola(img)

        img, center_eyes = get_center_eyes(img, landmark_image)
        center_eyes = get_center_eyes_float(landmark_image)
        print(center_eyes)
        rot = get_rotate(center_eyes)

        new_image = Image.fromarray(img)
        img = np.array(new_image.rotate(rot))

        img, info_data_viola = find_faces_viola(img, faces_viola, bounding_box_image, landmark_image)
        all_info_data_viola.append(info_data_viola)

        all_viola_precision.append(info_data_viola[3])
        all_viola_recall.append(info_data_viola[4])

        img = hp.add_landmarks(landmark_image, img, [0, 0, 255])
        img = hp.add_bounding_box(bounding_box_image, img, [0, 255, 0])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # out.write(img)
        cv2.imwrite(hp.create_directory_and_get_file_name_img(main_directory, DIRECTORY_ALL, directory, name, str(i)), img)

    hp.create_info_file(main_directory, DIRECTORY_ALL, directory, name, type + "_viola", all_info_data_viola)

    # out.release()


