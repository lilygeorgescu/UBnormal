from sklearn.metrics import roc_auc_score
import numpy as np
from scipy.ndimage import gaussian_filter1d


testing_videos_names = []  # list all the testing videos
testing_videos_names.sort()

all_frame_scores = []
all_gt_frame_scores = []
roc_auc_videos = []

for video_name in testing_videos_names:
    video_scores = None # load predicted scores for video

    # try to apply gaussian filter
    # video_scores = gaussian_filter1d(video_scores, 15, mode='constant')  # change the first and the second param

    all_frame_scores = np.append(all_frame_scores, video_scores)
    # read the ground truth scores at frame level
    gt_scores = None # load ground truth for video
    all_gt_frame_scores = np.append(all_gt_frame_scores, gt_scores)

    roc_auc = roc_auc_score(np.concatenate(([0], gt_scores, [1])), np.concatenate(([0], video_scores, [1])))
    roc_auc_videos.append(roc_auc)


roc_auc = roc_auc_score(all_gt_frame_scores, all_frame_scores)
print("Frame-based AUC is %.3f on %s (all data set)." % (roc_auc, 'synthetic'))
print("Avg. (on video) frame-based AUC is %.3f on %s." % (np.array(roc_auc_videos).mean(), 'synthetic'))