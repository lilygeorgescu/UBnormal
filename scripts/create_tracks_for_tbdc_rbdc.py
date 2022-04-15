from correct_tracks import load_tracks
import cv2 as cv
import numpy as np
import os
from pathlib import Path
import pdb


class Bbox:
    def __init__(self, x_min: int, y_min: int, x_max: int, y_max: int, is_abnormal: bool = False):
        self.x_min: int = x_min
        self.y_min: int = y_min
        self.x_max: int = x_max
        self.y_max: int = y_max
        self.is_abnormal = is_abnormal


def count_frames(path):
    # grab a pointer to the video file and initialize the total
    # number of frames read
    video = cv.VideoCapture(path)
    total = 0
    total = int(video.get(cv.CAP_PROP_FRAME_COUNT))

    # release the video file pointer
    video.release()
    # return the total number of frames in the video
    return total


def compute_total_num_frames(video_names):
    total_num_frames = 0

    for video_name in video_names:
        scene_name = video_name.split("_")[2]
        base_path = os.path.join(base_input_folder, f'Scene{scene_name}')
        video_path = os.path.join(base_path, f"{video_name}.mp4")
        print(f"Counting {video_path} ...")
        total_num_frames += count_frames(video_path)

    return total_num_frames


def get_bbox(frame) -> Bbox:
    contours, hierarchy = cv.findContours(np.uint8(frame * 255), cv.RETR_TREE,  cv.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None
    max_idx = 0
    max_area = 0
    for cnt_idx in range(len(contours)):
        x, y, w, h = cv.boundingRect(contours[cnt_idx])
        if w * h > max_area:
            max_area = w * h
            max_idx = cnt_idx

    x, y, w, h = cv.boundingRect(contours[max_idx])
    return Bbox(x_min=x, y_min=y, x_max=x + w, y_max=y + h)


def create_tracks(abnormal_video_names):

    for video_name in abnormal_video_names:
        scene_name = video_name.split("_")[2]
        base_path = os.path.join(base_input_folder, f'Scene{scene_name}')
        folder_annotation_path = os.path.join(base_path, f"{video_name}_annotations")
        tracks_path = os.path.join(folder_annotation_path, f"{video_name}_tracks.txt")
        if os.path.exists(os.path.join(folder_annotation_path, f"{video_name}_new_tracks.txt")):
            tracks_path = os.path.join(folder_annotation_path, f"{video_name}_new_tracks.txt")
        print(f"Computing tracks for {os.path.join(base_path, video_name)}...")
        tracks_np = load_tracks(tracks_path)
        tracks = tracks_np.tolist()
        tracks.sort(key=lambda track: track[0])
        results = []
        new_track_id = 0
        for track in tracks:
            track_id = int(track[0])

            start = int(track[1])
            end = int(track[2])

            for frame_idx in range(start, end + 1):

                gt_map_path = os.path.join(folder_annotation_path, '%s_%04d_gt.png' % (video_name, frame_idx))
                gt_map = cv.imread(gt_map_path, 0)
                assert gt_map is not None

                mask = (gt_map == track_id) * 1

                bbox = get_bbox(mask)

                res = np.array([new_track_id, frame_idx, bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max])
                #print(res)
                results.append(res)
            new_track_id += 1
        # print("Track path file for RBDC contains:")
        # print(results)
        rbdc_tbdc_tracks_path = os.path.join(output_folder, f"{video_name}.txt")
        np.savetxt(rbdc_tbdc_tracks_path, results, delimiter=",", fmt="%d")


if __name__ == '__main__':
    # path to the data set
    base_input_folder = "E:\\synthetic_abnormal_events_dataset"
    # path to the output folder
    output_folder = "tracks_val"
    os.makedirs(output_folder, exist_ok=True)
    # paths to the video names
    abnormal_video_names_path = "abnormal_validation_video_names.txt"
    normal_video_names_path = "normal_validation_video_names.txt"

    abnormal_video_names = np.loadtxt(abnormal_video_names_path, dtype=str)
    normal_video_names = np.loadtxt(normal_video_names_path, dtype=str)

    create_tracks(abnormal_video_names)
    print("Computing total number of frames...")
    num_abnormal_frames = compute_total_num_frames(abnormal_video_names)
    num_normal_frames = compute_total_num_frames(normal_video_names)
    total_num_frames = num_abnormal_frames + num_normal_frames
    print(f"There are a total of {total_num_frames} frames")
    # val 29071 frames
    # test 89908 frames