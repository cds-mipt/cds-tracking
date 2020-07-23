import os.path as osp
import os
import shutil
import numpy as np


def mkdirs(d):
    # if not osp.exists(d):
    if not osp.isdir(d):
        os.makedirs(d)


data_root = '/home/josh94mur/FairMOT_ws/Datasets/'
seq_root = data_root + 'MOT16/images/train'
label_root = data_root + 'MOT16/labels_with_ids/train'

cls_map = {
    'Pedestrian': 1,
    'Person on vehicle': 2,
    'Car': 3,
    'Bicycle': 4,
    'Motorbike': 5,
    'Non motorized vehicle': 6,
    'Static person': 7,
    'Distractor': 8,
    'Occluder': 9,
    'Occluder on the ground': 10,
    'Occluder full': 11,
    'Reflection': 12
}

if not os.path.isdir(label_root):
    mkdirs(label_root)
else:  #  If it has been generated before: Recursively delete directories and files, regenerate directories
    shutil.rmtree(label_root)
    os.makedirs(label_root)

print("Dir %s made" % label_root)
seqs = [s for s in os.listdir(seq_root)]

tid_curr = 0
tid_last = -1
total_track_id_num = 0
for seq in seqs:  # Each video corresponds to a gt.txt
    print("Process %s, " % seq, end='')

    seq_info_path = osp.join(seq_root, seq, 'seqinfo.ini')
    with open(seq_info_path) as seq_info_h:   # Read *.ini file
        seq_info = seq_info_h.read()
        seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])  # Video width
        seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])  # Video height

    gt_txt = osp.join(seq_root, seq, 'gt', 'gt.txt')  # Read GT file
    gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')  # Load into np format
    idx = np.lexsort(gt.T[:2, :])  # Prioritize sorting by track id (sort the video frames, and then sort the track ID)
    gt = gt[idx, :]

    tr_ids = set(gt[:, 1])
    print("%d track ids in seq %s" % (len(tr_ids), seq))
    total_track_id_num += len(tr_ids) # How to calculate the track id statistics correctly?

    seq_label_root = osp.join(label_root, seq, 'img1')
    mkdirs(seq_label_root)

    # Read each line of GT data (one line is one data)
    for fid, tid, x, y, w, h, mark, cls, vis_ratio in gt:
        # frame_id, track_id, top, left, width, height, mark, class, visibility ratio
        if cls != 3:  # We need Car’s annotation data
            continue

        # if mark == 0:  # Ignore when mark is 0 (not considered in the current frame)
        #     continue

        # if vis_ratio <= 0.2:
        #     continue

        fid = int(fid)
        tid = int(tid)

        # Determine whether it is the same track, record the previous track and the current track
        if not tid == tid_last:  #  has not a higher priority than ==
            tid_curr += 1
            tid_last = tid

        # bbox center point coordinates
        x += w / 2
        y += h / 2

        # Write track id, bbox center point coordinates and width and height in the net label (normalized to 0~1)
        # The 0 in the first column is only one category for multi-target detection and tracking by default (0 is the category)
        label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
            tid_curr,
            x / seq_width,   # center_x
            y / seq_height,  # center_y
            w / seq_width,   # bbox_w
            h / seq_height)  # bbox_h
        # print(label_str.strip())

        label_f_path = osp.join(seq_label_root, '{:06d}.txt'.format(fid))
        with open(label_f_path, 'a') as f:  # Add the label of each frame by appending
            f.write(label_str)

print("Total %d track ids in this dataset" % total_track_id_num)
print('Done')
