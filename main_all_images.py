import numpy as np

import file_reader_writer as frw
import all_all_images as all
import helper as hp
import time

DIRECTORY_WITH_WIDEO = "videa/"
DIRECTORY_FOR_SAVE_TRUE = "TRUE/"
DIRECTORY_FOR_SAVE_FALSE = "FALSE/"


def get_true_pairs():
    true_pairs = frw.get_pairs_from_csv("pairs/TruePairs.csv")

    # pairs = [true_pairs[10], true_pairs[20], true_pairs[30], true_pairs[40], true_pairs[50], true_pairs[60],
    #          true_pairs[70], true_pairs[80], true_pairs[90], true_pairs[100]]

    return true_pairs


def get_false_pairs():
    false_pairs = frw.get_pairs_from_csv("pairs/FalsePairs.csv")

    # pairs = [false_pairs[5], false_pairs[10], false_pairs[15], false_pairs[20], false_pairs[25], false_pairs[30],
    #          false_pairs[35], false_pairs[40], false_pairs[45], false_pairs[50]]

    return false_pairs


def work_with_true_pairs():
    pairs = get_true_pairs()
    for pair in pairs:
        video_1_name = pair[0]
        video_2_name = pair[1]

        video_1_file = np.load(DIRECTORY_WITH_WIDEO + video_1_name)
        video_2_file = np.load(DIRECTORY_WITH_WIDEO + video_2_name)

        video_1_file_without_suffix = video_1_name.split(".")[0]
        video_2_file_without_suffix = video_2_name.split(".")[0]
        directory_name = video_1_file_without_suffix + "-" + video_2_file_without_suffix + "/"
        video_1_info = all.to_mp4(DIRECTORY_FOR_SAVE_TRUE, directory_name, video_1_file_without_suffix, "ORIGINAL",
                                  video_1_file["colorImages"],
                                  video_1_file["landmarks2D"],
                                  video_1_file["boundingBox"])
        video_2_info = all.to_mp4(DIRECTORY_FOR_SAVE_TRUE, directory_name, video_2_file_without_suffix, "ORIGINAL",
                                  video_2_file["colorImages"],
                                  video_2_file["landmarks2D"],
                                  video_2_file["boundingBox"])

        # summary_info = [video_1_info, video_2_info]
        # hp.create_summary_info(DIRECTORY_FOR_SAVE, "ALL/", directory_name, "SUMMARY", "", summary_info)


def work_with_false_pairs():
    pairs = get_false_pairs()
    for pair in pairs:
        video_1_name = pair[0]
        video_2_name = pair[1]

        video_1_file = np.load(DIRECTORY_WITH_WIDEO + video_1_name)
        video_2_file = np.load(DIRECTORY_WITH_WIDEO + video_2_name)

        video_1_file_without_suffix = video_1_name.split(".")[0]
        video_2_file_without_suffix = video_2_name.split(".")[0]
        directory_name = video_1_file_without_suffix + "-" + video_2_file_without_suffix + "/"
        video_1_info = all.to_mp4(DIRECTORY_FOR_SAVE_FALSE, directory_name, video_1_file_without_suffix, "ORIGINAL",
                                  video_1_file["colorImages"],
                                  video_1_file["landmarks2D"],
                                  video_1_file["boundingBox"])
        video_2_info = all.to_mp4(DIRECTORY_FOR_SAVE_FALSE, directory_name, video_2_file_without_suffix, "ORIGINAL",
                                  video_2_file["colorImages"],
                                  video_2_file["landmarks2D"],
                                  video_2_file["boundingBox"])


start_time_all = time.perf_counter()

work_with_true_pairs()

end_true = time.perf_counter() - start_time_all
print("Čas pre true páry: ", end_true)

start_false = time.perf_counter()

work_with_false_pairs()

end_false = time.perf_counter() - start_false
print("Čas pre false páry: ", end_false)

elapsed_time_all = time.perf_counter() - start_time_all
print("Celkový čas: ", elapsed_time_all)
