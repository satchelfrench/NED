# image flowset
# raw optical flow data as tensor (not visualized)
# added dimension to image and saved as tensor
#

import os
import argparse
import numpy as np
import cv2

def calc_optical_flow_dense(prior, current):
    frame1 = cv2.cvtColor(prior, cv2.COLOR_BGR2GRAY)
    frame2 = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)

    hsv = np.zeros_like(prior)
    hsv[..., 1] = 255

    flow = cv2.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return rgb

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, required=True) # proccessed
    parser.add_argument('--labels', type=str, required=True)
    parser.add_argument('--out_path', type=str, default='./neo-echoflow') 

    args = parser.parse_args()

    
    if not os.path.exists(args.data_path):
        raise FileNotFoundError("Data path is invalid. Please try again with an alternate path. ")
    
    if not os.path.exists(args.labels):
        raise FileExistsError("Label or annotation files were not found at provided path.")

    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)
    
    if not os.path.exists(args.labels):
        raise FileExistsError
    
    videos = []
    with open(args.labels, 'r') as labels:
        videos = labels.readlines()
    
    if len(videos) == 0:
        print("ERROR: Unable to read labels file correctly! ")
        raise FileNotFoundError("Wrong error")

    for video in videos:
        parsed_video = video.split()

        video_path = os.path.join(args.data_path, parsed_video[0])
        out_path = os.path.join(args.out_path, parsed_video[0])
        num_frames = int(parsed_video[2])
        
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        
        # Init condition
        frame_name = "img_{0:05d}.jpg".format(int(1))
        frame_path = os.path.join(video_path, frame_name)
        last_frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
        
        for i in range(2, num_frames+1):
            frame_name = "img_{0:05d}.jpg".format(int(i))
            frame_path = os.path.join(video_path, frame_name)
            current_frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)

            flow = calc_optical_flow_dense(last_frame, current_frame)
            last_frame = current_frame

            # Save flow image to output directory..
            flow_name = "flow_{0:05d}.jpg".format(int(i))
            save_path = os.path.join(out_path, flow_name)
            print(f"saving to {save_path}")
            cv2.imwrite(save_path, flow)
            









    