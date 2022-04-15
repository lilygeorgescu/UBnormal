import os
import pdb
import cv2 as cv
import numpy as np
import glob
from pathlib import Path

def load_tracks(video_name):
    scene_name = video_name.split("_")[2]
    tracks_path = os.path.join(base_folder_input, f"Scene{scene_name}", video_name + "_annotations",
                               video_name + "_tracks.txt")
    anomaly_per_video = np.loadtxt(tracks_path, delimiter=",")

    if anomaly_per_video.ndim == 1:
        anomaly_per_video = [anomaly_per_video]

    return np.array(anomaly_per_video)


def get_number_of_anomalies(video_names) -> int:
    num_anomalies = 0
    for video_name in video_names:
        tracks = load_tracks(video_name)
        num_anomalies += tracks.shape[0]
    return num_anomalies


def compute_number_of_abnormal_frames(tracks):
    tracks = list(tracks)
    # sort based on the start_idx
    tracks.sort(key=lambda track: track[1])

    num_abnormal_frames = 0
    max_interval = -1
    for track in tracks:
        if track[1] > max_interval:  # no overlap
            num_abnormal_frames += (track[2] - track[1] + 1)
        elif track[1] < max_interval < track[2]:
            num_abnormal_frames += (track[2] - max_interval)
        max_interval = max(max_interval, track[2])

    return num_abnormal_frames


def get_number_of_abnormal_frames_video(video_name):
    tracks = load_tracks(video_name)
    return compute_number_of_abnormal_frames(tracks)


def get_number_of_abnormal_frames(video_names):
    num_abnormal_frames = 0
    for video_name in video_names:
        num_abnormal_frames += get_number_of_abnormal_frames_video(video_name)
    return num_abnormal_frames


def get_number_of_frames_video(video_name):
    scene_name = video_name.split("_")[2]
    video_path = os.path.join(base_folder_input, f'Scene{scene_name}', video_name + ".mp4")
    try:
        assert os.path.exists(video_path), video_name
    except Exception:
        print(video_name)
        pass

    video = cv.VideoCapture(video_path)
    total = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    video.release()
    return total


def get_number_of_normal_frames_for_abnormal_video(video_name):
    return get_number_of_frames_video(video_name) - get_number_of_abnormal_frames_video(video_name)


def get_number_of_abnormal_regions_per_video(video_name):
    tracks = load_tracks(video_name)
    num_abnormal_regions = 0
    for track in tracks:
        num_abnormal_regions += (track[2] - track[1] + 1)

    return num_abnormal_regions


def get_number_of_normal_frames(normal_files, abnormal_files):
    num_normal_frames = 0
    for video_name in normal_files:
        num_normal_frames += get_number_of_frames_video(video_name)

    for video_name in abnormal_files:
        num_normal_frames += get_number_of_normal_frames_for_abnormal_video(video_name)
    return num_normal_frames


def get_number_of_abnormal_regions(video_names):
    return np.sum([get_number_of_abnormal_regions_per_video(video_name) for video_name in video_names])


def get_number_of_unique_objects_in_video(video_name):
    scene_name = video_name.split("_")[2]
    folder_annotation_path = os.path.join(base_folder_input, f'Scene{scene_name}', f"{video_name}_annotations")
    annotation_paths = glob.glob(os.path.join(folder_annotation_path, "*.png"))
    annotation_paths.sort()

    ids = set()

    for frame_idx, annotation_path in enumerate(annotation_paths):
        annotation_map = cv.imread(annotation_path, 0)
        hist = np.bincount(annotation_map.flatten())
        for key, value in enumerate(hist):
            if key != 0 and value != 0:
                ids.add(key)

    return len(ids)

def get_number_of_all_objects_in_video(video_name):
    scene_name = video_name.split("_")[2]
    folder_annotation_path = os.path.join(base_folder_input, f'Scene{scene_name}', f"{video_name}_annotations")
    annotation_paths = glob.glob(os.path.join(folder_annotation_path, "*.png"))
    annotation_paths.sort()

    num = 0

    for frame_idx, annotation_path in enumerate(annotation_paths):
        annotation_map = cv.imread(annotation_path, 0)
        hist = np.bincount(annotation_map.flatten())
        for key, value in enumerate(hist):
            if key != 0 and value != 0:
                num += 1

    return num

def get_number_of_all_objects_in_videos(video_names):
    return np.sum([get_number_of_all_objects_in_video(video_name) for video_name in video_names])

def get_number_of_frames_in_videos(video_names):
    return np.sum([get_number_of_frames_video(video_name) for video_name in video_names])

def get_statics_number_objects_frames(video_names) -> str:
    num_of_objects = get_number_of_all_objects_in_videos(video_names)
    num_of_frames = get_number_of_frames_in_videos(video_names)
    return f"num_of_objects = {num_of_objects}, num_of_frames = {num_of_frames}, avg = {num_of_objects / num_of_frames}"

def print_avg_number_of_objects_per_frame():
    print(f"Training Normal: {get_statics_number_objects_frames(normal_train_video_names)}")
    print(f"Training Abnormal: {get_statics_number_objects_frames(abnormal_train_video_names)}")

    print(f"Validation Normal: {get_statics_number_objects_frames(normal_val_video_names)}")
    print(f"Validation Abnormal: {get_statics_number_objects_frames(abnormal_val_video_names)}")

    print(f"Test Normal: {get_statics_number_objects_frames(normal_test_video_names)}")
    print(f"Test Abnormal: {get_statics_number_objects_frames(abnormal_test_video_names)}")


def compute_number_of_objects_in_videos(video_names):
    return np.sum([get_number_of_unique_objects_in_video(video_name=video_name) for video_name in video_names])


def compute_number_of_frames_per_scene(scene_name):
    full_path = 'E:\\UBnormal\\' + scene_name + "/*.mp4"
    video_paths = glob.glob(full_path)

    return np.sum([get_number_of_frames_video(Path(video_path).parts[-1][:-4]) for video_path in video_paths])


def print_num_of_frames_per_scene():
    for scene_number in range(1, 30):
        print(f"Scene{scene_number} = {compute_number_of_frames_per_scene(f'Scene{scene_number}')}")

def print_number_of_objects_in_videos():
    num_objects_train = compute_number_of_objects_in_videos(normal_train_video_names) \
                        + compute_number_of_objects_in_videos(abnormal_train_video_names)
    print(f"Number of objects in training {num_objects_train}")

    num_objects_val = compute_number_of_objects_in_videos(normal_val_video_names) \
                      + compute_number_of_objects_in_videos(abnormal_val_video_names)
    print(f"Number of objects in validation {num_objects_val}")

    num_objects_test = compute_number_of_objects_in_videos(normal_test_video_names) \
                      + compute_number_of_objects_in_videos(abnormal_test_video_names)

    print(f"Number of objects in test {num_objects_test}")


def print_number_of_abnormal_regions():
    print(f"Number of abnormal regions in training {get_number_of_abnormal_regions(abnormal_train_video_names)}")
    print(f"Number of abnormal regions in validation {get_number_of_abnormal_regions(abnormal_val_video_names)}")
    print(f"Number of abnormal regions in test {get_number_of_abnormal_regions(abnormal_test_video_names)}")


def print_num_of_normal_frames():
    num_normal_frames_train = get_number_of_normal_frames(normal_train_video_names, abnormal_train_video_names)
    print(f"Number of normal frames training {num_normal_frames_train}, minutes = {num_normal_frames_train / (60 * 30)}")

    num_normal_frames_val = get_number_of_normal_frames(normal_val_video_names, abnormal_val_video_names)
    print(f"Number of normal frames in validation {num_normal_frames_val}, minutes = {num_normal_frames_val / (60 * 30)}")

    num_normal_frames_test = get_number_of_normal_frames(normal_test_video_names, abnormal_test_video_names)
    print(f"Number of normal frames in test {num_normal_frames_test}, minutes = {num_normal_frames_test / (60 * 30)}")


def print_num_anomalies():
    print(f"Number of anomalies in training {get_number_of_anomalies(abnormal_train_video_names)}")
    print(f"Number of anomalies in validation {get_number_of_anomalies(abnormal_val_video_names)}")
    print(f"Number of anomalies in test {get_number_of_anomalies(abnormal_test_video_names)}")


def print_num_abnormal_frames():
    num_frames_train = get_number_of_abnormal_frames(abnormal_train_video_names)
    print(f"Number of abnormal frames in training {num_frames_train}, num minutes: {num_frames_train / (30 * 60)}")

    num_frames_val = get_number_of_abnormal_frames(abnormal_val_video_names)
    print(f"Number of abnormal frames in validation {num_frames_val}, num minutes: {num_frames_val / (30 * 60)}")

    num_frames_test = get_number_of_abnormal_frames(abnormal_test_video_names)
    print(f"Number of abnormal frames in test {num_frames_test}, num minutes: {num_frames_test / (30 * 60)}")


if __name__ == '__main__':
    # Path to the data set
    base_folder_input = "E:\\UBnormal"

    abnormal_train_video_names_path = "abnormal_training_video_names.txt"
    abnormal_train_video_names = np.loadtxt(abnormal_train_video_names_path, dtype=str)
    normal_train_video_names_path = "normal_training_video_names.txt"
    normal_train_video_names = np.loadtxt(normal_train_video_names_path, dtype=str)

    abnormal_val_video_names_path = "abnormal_validation_video_names.txt"
    abnormal_val_video_names = np.loadtxt(abnormal_val_video_names_path, dtype=str)
    normal_val_video_names_path = "normal_validation_video_names.txt"
    normal_val_video_names = np.loadtxt(normal_val_video_names_path, dtype=str)

    abnormal_test_video_names_path = "abnormal_test_video_names.txt"
    abnormal_test_video_names = np.loadtxt(abnormal_test_video_names_path, dtype=str)
    normal_test_video_names_path = "normal_test_video_names.txt"
    normal_test_video_names = np.loadtxt(normal_test_video_names_path, dtype=str)

    print_num_of_frames_per_scene()
    print_num_anomalies()

    print_num_abnormal_frames()

    print_num_of_normal_frames()

    print_number_of_abnormal_regions()

    print_number_of_objects_in_videos()

    print_avg_number_of_objects_per_frame()