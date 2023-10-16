import os
import argparse
from datility.utils import calc_optical_flow_dense, flow_to_rgb
import numpy as np

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
            flow_rgb = flow_to_rgb(np.shape(current_frame), flow)
            last_frame = current_frame

            # Save flow image to output directory..
            flow_name = "flow_{0:05d}.jpg".format(int(i))
            save_path = os.path.join(out_path, flow_name)
            print(f"saving to {save_path}")
            cv2.imwrite(save_path, flow_rgb)
            









    