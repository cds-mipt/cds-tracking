import cv2
import numpy as np
from scipy.spatial.distance import pdist, squareform
from tracking_utils import visualization as vis

def plot_points_on_bird_eye_view(frame, track_image, pedestrian_boxes, M, scale_w, scale_h, id, Track_ID_List, ids_previous_frame, ids_pre_previous_frame):
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]

    node_radius = 10
    thickness_node = 4
    solid_back_color = (41, 41, 41)
    track_node_radius = 2
    track_thickness_node = 2

    # blank_image = np.zeros(
    #     (int(frame_h * scale_h), int(frame_w * scale_w), 3), np.uint8
    # )
    # blank_image[:] = solid_back_color
    
    # blank_image = track_image

    
    warped_pts = []
    bird_image = track_image.copy()
    for i in range(len(pedestrian_boxes)):

        color_node = vis.get_color(id[i]) # Color_list[id[i]%10]
        #print(id[i])

        mid_point_y = int(              #revrese x,y depending on BB labeling
            pedestrian_boxes[i][1]  + pedestrian_boxes[i][3] / 2
        )
        mid_point_x = int(
            pedestrian_boxes[i][0]  + pedestrian_boxes[i][2] / 2
        )

        pts = np.array([[[mid_point_x, mid_point_y]]], dtype="float32")
        warped_pt = cv2.perspectiveTransform(pts, M)[0][0]
        warped_pt_scaled = [int(warped_pt[0] * scale_w), int(warped_pt[1] * scale_h)]

        warped_pts.append(warped_pt_scaled)

        Track_ID_List[id[i]].append((warped_pt_scaled[0], warped_pt_scaled[1]))


        track_image = cv2.circle(
            track_image,
            (warped_pt_scaled[0], warped_pt_scaled[1]),
            track_node_radius,
            color_node, 
            track_thickness_node,
        )
        
        bird_image = cv2.circle(
            bird_image,
            (warped_pt_scaled[0], warped_pt_scaled[1]),
            node_radius,
            color_node,
            thickness_node,
        )
        # Adding lines between track points
        PTS_Lines = np.array(Track_ID_List[id[i]],np.int32)
        PTS_Lines = PTS_Lines.reshape((-1, 1, 2))

        bird_image = cv2.polylines(bird_image, [PTS_Lines], False, color_node, thickness=2)
        
        # track_image = cv2.rectangle(
        #     track_image,
        #     (warped_pt_scaled[0], warped_pt_scaled[1]), 
        #     (warped_pt_scaled[0]+1, warped_pt_scaled[1]+1), 
        #     color_node,
        # )
        #track_image[warped_pt_scaled[0], warped_pt_scaled[1]]=color_node
    deleted_tracks = [x for x in ids_pre_previous_frame if x not in id and x not in ids_previous_frame]
    for i in deleted_tracks:
        for j in  Track_ID_List[i]:
            track_image = cv2.circle(
                track_image,
                j,
                track_node_radius,
                solid_back_color, 
                track_thickness_node,
            )



    


    return warped_pts, bird_image, track_image, Track_ID_List


def get_camera_perspective(img, src_points):
    IMAGE_H = img.shape[0]
    IMAGE_W = img.shape[1]
    src = np.float32(np.array(src_points))
    dst = np.float32([[0, IMAGE_H], [IMAGE_W, IMAGE_H], [0, 0], [IMAGE_W, 0]])

    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)

    return M, M_inv


def put_text(frame, text, text_offset_y=25):
    font_scale = 0.8
    font = cv2.FONT_HERSHEY_SIMPLEX
    rectangle_bgr = (35, 35, 35)
    (text_width, text_height) = cv2.getTextSize(
        text, font, fontScale=font_scale, thickness=1
    )[0]
    # set the text start position
    text_offset_x = frame.shape[1] - 400
    # make the coords of the box with a small padding of two pixels
    box_coords = (
        (text_offset_x, text_offset_y + 5),
        (text_offset_x + text_width + 2, text_offset_y - text_height - 2),
    )
    frame = cv2.rectangle(
        frame, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED
    )
    frame = cv2.putText(
        frame,
        text,
        (text_offset_x, text_offset_y),
        font,
        fontScale=font_scale,
        color=(255, 255, 255),
        thickness=2,
    )

    return frame, 2 * text_height + text_offset_y
