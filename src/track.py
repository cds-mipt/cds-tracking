from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import os.path as osp
import cv2
import logging
import argparse
import motmetrics as mm
import numpy as np
import torch

from tracker.multitracker import JDETracker
from tracking_utils import visualization as vis
from tracking_utils.log import logger
from tracking_utils.timer import Timer
from tracking_utils.evaluation import Evaluator
import datasets.dataset.jde as datasets

from tracking_utils.utils import mkdir_if_missing
from opts import opts
#=======================================
from aux_functions import *
from collections import defaultdict


mouse_pts = []

def get_mouse_points(event, x, y, flags, param):
    # Used to mark 4 points on the frame zero of the video that will be warped
    # Used to mark 2 points on the frame zero of the video that are 6 feet away
    global mouseX, mouseY, mouse_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX, mouseY = x, y
        #cv2.circle(image, (x, y), 10, (0, 255, 255), 10)   # TODO fix this bug ()
        if "mouse_pts" not in globals():
            mouse_pts = []
        mouse_pts.append((x, y))
        print("Point detected")
        print(mouse_pts)
#=======================================

def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def eval_seq(opt, dataloader, data_type, result_filename, save_dir=None, show_image=True, frame_rate=30):
    if save_dir:
        mkdir_if_missing(save_dir)
    tracker = JDETracker(opt, frame_rate=frame_rate)
    timer = Timer()
    results = []
    frame_id = 0
    for path, img, img0 in dataloader:
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        # run tracking
        timer.tic()
        blob = torch.from_numpy(img).cuda().unsqueeze(0)
        online_targets = tracker.update(blob, img0)
        online_tlwhs = []
        online_ids = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
        timer.toc()
        # save results

        results.append((frame_id + 1, online_tlwhs, online_ids))
        if show_image or save_dir is not None:
            online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id,
                                          fps=1. / timer.average_time)
        if show_image:
            cv2.imshow('online_im', online_im)
        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
        frame_id += 1
    # save results
    write_results(result_filename, results, data_type)
    return frame_id, timer.average_time, timer.calls, results


def main(opt, data_root='/data/MOT16/train', det_root=None, seqs=('MOT16-05',), exp_name='demo',
         save_images=False, save_videos=False, show_image=True):
    logger.setLevel(logging.INFO)
    result_root = os.path.join(data_root, '..', 'results', exp_name)
    mkdir_if_missing(result_root)
    data_type = 'mot'

    # run tracking
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []
    for seq in seqs:
        output_dir = os.path.join(data_root, '..', 'outputs', exp_name, seq) if save_images or save_videos else None
        logger.info('start seq: {}'.format(seq))
        dataloader = datasets.LoadImages(osp.join(data_root, seq, 'img1'), opt.img_size)
        result_filename = os.path.join(result_root, '{}.txt'.format(seq))
        meta_info = open(os.path.join(data_root, seq, 'seqinfo.ini')).read()
        frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])
        nf, ta, tc, results = eval_seq(opt, dataloader, data_type, result_filename,
                              save_dir=output_dir, show_image=show_image, frame_rate=frame_rate)
        n_frame += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)

        # eval
        logger.info('Evaluate seq: {}'.format(seq))
        evaluator = Evaluator(data_root, seq, data_type)
        accs.append(evaluator.eval_file(result_filename))
        if save_videos:
            output_video_path = osp.join(output_dir, '{}.mp4'.format(seq))
            cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}'.format(output_dir, output_video_path)
            os.system(cmd_str)

            #=======================================
            input_video = output_video_path

            # Get video handle
            cap = cv2.VideoCapture(input_video)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            scale_w = 1.2 / 2
            scale_h = 4 / 2

            SOLID_BACK_COLOR = (41, 41, 41)   #Change Background
            # Setuo video writer
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            output_movie = cv2.VideoWriter("P_detect.avi", fourcc, fps, (width, height))
            bird_movie = cv2.VideoWriter("P_bird.avi", fourcc, fps, (int(width * scale_w), int(height * scale_h))
            )
            # Initialize necessary variables
            frame_num = 0
            total_pedestrians_detected = 0
            total_six_feet_violations = 0
            total_pairs = 0
            abs_six_feet_violations = 0
            pedestrian_per_sec = 0
            sh_index = 1
            sc_index = 1

            cv2.namedWindow('image')
            # cv2.setMouseCallback("image", get_mouse_points)
            num_mouse_points = 0
            first_frame_display = True

            # Process each frame, until end of video
            while cap.isOpened():
                frame_num += 1
                ret, frame = cap.read()

                if not ret:
                    print("end of the video file...")
                    break

                frame_h = frame.shape[0]
                frame_w = frame.shape[1]

                if frame_num == 1:
                    # Ask user to mark parallel points and two points 6 feet apart. Order bl, br, tr, tl, p1, p2
                    image = frame
                    cv2.namedWindow('image')
                    cv2.setMouseCallback('image', get_mouse_points)
                    while True:
                        image = frame
                        cv2.imshow('image', image)
                        cv2.waitKey(1)
                        if len(mouse_pts) == 7:
                            cv2.destroyWindow('image')
                            break
                        first_frame_display = False
                    four_points = mouse_pts

                    # Get perspective
                    M, Minv = get_camera_perspective(frame, four_points[0:4])
                    pts = src = np.float32(np.array([four_points[4:]]))
                    warped_pt = cv2.perspectiveTransform(pts, M)[0]   # Performs the perspective matrix transformation of vectors.

                    d_thresh = np.sqrt(
                        (warped_pt[0][0] - warped_pt[1][0]) ** 2
                        + (warped_pt[0][1] - warped_pt[1][1]) ** 2
                    )

                    corners = src = np.float32(np.array([four_points[0:4]]))
                    wraped_corners = cv2.perspectiveTransform(corners, M)[0]
                    
                    # Calculating the real distance between the polygon points 
                    H_Length = np.sqrt(
                        (wraped_corners[0][0] - wraped_corners[1][0]) ** 2
                        + (wraped_corners[0][1] - wraped_corners[1][1]) ** 2
                    )
                    H_Meter = H_Length * 1.8288 / d_thresh  #don't forget the scale

                    V_Length = np.sqrt(
                        (wraped_corners[0][0] - wraped_corners[2][0]) ** 2
                        + (wraped_corners[0][1] - wraped_corners[2][1]) ** 2
                    )
                    V_Meter = V_Length * 1.8288 / d_thresh

                    bird_image = np.zeros(
                        (int(frame_h * scale_h), int(frame_w * scale_w), 3), np.uint8
                    )
                    ids_previous_frame=[]
                    ids_pre_previous_frame = []
                    Track_ID_List = defaultdict(list)
                    track_image = np.zeros(
                        (int(frame_h * scale_h), int(frame_w * scale_w), 3), np.uint8
                    )
                    #bird_image = track_image

                    bird_image[:] = SOLID_BACK_COLOR
                    track_image[:] = SOLID_BACK_COLOR
                    pedestrian_detect = frame

                print("Processing frame: ", frame_num)

                # draw polygon of ROI
                pts = np.array(
                    [four_points[0], four_points[1], four_points[3], four_points[2]], np.int32
                )
                cv2.polylines(frame, [pts], True, (0, 255, 255), thickness=3)
                # Detect person and bounding boxes using FairMOT 
                (frr, pedestrian_boxes, id_pedestrians) =  results[frame_num]
                num_pedestrians = len(id_pedestrians)
                if len(pedestrian_boxes) > 0:
                    pedestrian_detect = frame 
                    warped_pts, bird_image, track_image, Track_ID_List = plot_points_on_bird_eye_view(
                        frame, track_image, pedestrian_boxes, M, scale_w, scale_h, id_pedestrians , Track_ID_List, ids_previous_frame, ids_pre_previous_frame
                    )
                ids_pre_previous_frame = ids_previous_frame
                ids_previous_frame = id_pedestrians
 
                last_h = 50
                text = "Estimated Area: " + str(round(H_Meter, 2)) + " * " + str(round(V_Meter)) + " M"
                bird_image, last_h = put_text(bird_image, text, text_offset_y=last_h)

                text = "# Pedestrians: " + str(num_pedestrians).zfill(2)
                bird_image, last_h = put_text(bird_image, text, text_offset_y=last_h)

                #cv2.imshow("Cam", pedestrian_detect)
                #filename1 = 'pedestrian_detect'+str(frame_num).zfill(5)+'.jpg'    
                #cv2.imwrite(filename1, pedestrian_detect)
                cv2.waitKey(1)
                output_movie.write(pedestrian_detect)
                filename2 = 'bird_image'+str(frame_num).zfill(5)+'.jpg'
                cv2.imwrite(filename2, bird_image)
                bird_movie.write(bird_image)
            #=======================================

    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))

    # get summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    Evaluator.save_summary(summary, os.path.join(result_root, 'summary_{}.xlsx'.format(exp_name)))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().init()

    if not opt.val_mot16:
        seqs_str = '''KITTI-13
                      KITTI-17
                      ADL-Rundle-6
                      PETS09-S2L1
                      TUD-Campus
                      TUD-Stadtmitte'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/train')
    else:
        seqs_str = '''MOT16-02
                      MOT16-04
                      MOT16-05
                      MOT16-09
                      MOT16-10
                      MOT16-11
                      MOT16-13'''
        data_root = os.path.join(opt.data_dir, 'MOT16/train')
    if opt.test_mot16:
        seqs_str = '''MOT16-01
                      MOT16-03
                      MOT16-06
                      MOT16-07
                      MOT16-08
                      MOT16-12
                      MOT16-14'''
        data_root = os.path.join(opt.data_dir, 'MOT16/test')
    if opt.test_mot15:
        seqs_str = '''ETH-Crossing
                      '''
                    #   Venice-1
                    #   ADL-Rundle-1
                    #   ADL-Rundle-3
                    #   AVG-TownCentre
                    #   ETH-Jelmoli
                    #   ETH-Linthescher
                    #   KITTI-16
                    #   KITTI-19
                    #   PETS09-S2L2
                    #   TUD-Crossing
        data_root = os.path.join(opt.data_dir, 'MOT15/images/test')
    if opt.test_mot17:
        seqs_str = '''MOT17-01-SDP
                      MOT17-03-SDP
                      MOT17-06-SDP
                      MOT17-07-SDP
                      MOT17-08-SDP
                      MOT17-12-SDP
                      MOT17-14-SDP'''
        data_root = os.path.join(opt.data_dir, 'MOT17/images/test')
    if opt.val_mot17:
        seqs_str = '''MOT17-02-SDP
                      MOT17-04-SDP
                      MOT17-05-SDP
                      MOT17-09-SDP
                      MOT17-10-SDP
                      MOT17-11-SDP
                      MOT17-13-SDP'''
        data_root = os.path.join(opt.data_dir, 'MOT17/images/train')
    if opt.val_mot15:
        seqs_str = '''KITTI-13
                      '''
                    #   TUD-Stadtmitte
                    #   KITTI-17
                    #   ETH-Bahnhof
                    #   ETH-Sunnyday
                    #   PETS09-S2L1
                    #   TUD-Campus
                    #   TUD-Stadtmitte
                    #   ADL-Rundle-6
                    #   ADL-Rundle-8
                    #   ETH-Pedcross2
        data_root = os.path.join(opt.data_dir, 'MOT15/images/train')
    if opt.val_mot20:
        seqs_str = '''MOT20-01
                      MOT20-02
                      MOT20-03
                      MOT20-05
                      '''
        data_root = os.path.join(opt.data_dir, 'MOT20/images/train')
    if opt.test_mot20:
        seqs_str = '''MOT20-07  
                      ''' #MOT20-04 MOT20-06 MOT20-08
        data_root = os.path.join(opt.data_dir, 'MOT20/images/test')
    seqs = [seq.strip() for seq in seqs_str.split()]

    main(opt,
         data_root=data_root,
         seqs=seqs,
         exp_name='MOT15_val_all_dla34',
         show_image=False,
         save_images=True,
         save_videos=True)


