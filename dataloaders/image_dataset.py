import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from datility.utils import calc_optical_flow_dense
import torchvision.transforms.functional as TF
import torch
import pickle
import cv2


'''
Needs to randomly select images from the videoset structure and handle resampling..
appent all frames to a list, just needs to generate the image paths from the given video path (os walk or something..)
add to a list and leave it.

also should work for optical flow..

# [ ['path_to_image', 'class_id'] ]

'''

class FrameDataset(Dataset):
    def __init__(self, annotations_file, root_dir, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.data = []

        with open(annotations_file, 'r') as annotations:
            video_samples = annotations.readlines()

        for video_sample in video_samples:
            parsed = video_sample.split()

            video_path = os.path.join(self.root_dir, parsed[0])
            for frame_path in os.listdir(video_path):
                self.data.append(
                    tuple([
                        os.path.join(video_path, frame_path),
                        int(parsed[-1])
                    ])
                )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        image = read_image(image_path)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    

class TensorFrameDataset(Dataset):
    def __init__(self, annotations_file, root_dir, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.data = []

        with open(annotations_file, 'r') as annotations:
            video_samples = annotations.readlines()

            for video_sample in video_samples:
                parsed = video_sample.split()

                video_path = os.path.join(self.root_dir, parsed[0])
                for frame_path in os.listdir(video_path):
                    self.data.append(
                        tuple(
                            os.path.join(video_path, frame_path),
                            int(parsed[-1])
                        )
                    )

    def __len__(self):
        return len(self.data)
    

    '''
    Issue:

    - Cant apply image transforms to tensor must be to image?
    - If applied to image separately that OF, then OF is no longer valid for that frame (depending on transform)
    '''

    def __getitem__(self, idx):
        tensor_path, label = self.data[idx]
        
        # assume its pickled
        tensor = pickle.load(tensor_path)

        if self.transform:
            tensor = self.transform(tensor)
        if self.target_transform:
            label = self.target_transform(label)
        return tensor, label


'''
Experimental:
- Optical Flow Dataset
- Read frames in consecutive order
- Calc Optical Flow, and combine grayscale image channel with optical flow channel, produce as a single sample.
- Cache
'''   
class OpticalFlowDataset(Dataset):
    def __init__(self, annotations_file, root_dir, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.data = []

        with open(annotations_file, 'r') as annotations:
            video_samples = annotations.readlines()

        for video_sample in video_samples:
            parsed = video_sample.split()

            video_path = os.path.join(self.root_dir, parsed[0])
            sorted_frames = sorted(os.listdir(video_path))
            
            for i in range(1, len(sorted_frames)):
                frame_1_path = sorted_frames[i-1]
                frame_2_path = sorted_frames[i]
                self.data.append(
                    tuple([
                        tuple([os.path.join(video_path, frame_1_path),
                              os.path.join(video_path, frame_2_path)]),
                        int(parsed[-1])
                    ])
                )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        images, label = self.data[idx]
        frame_1_path, frame_2_path = images

        frame1 =cv2.imread(frame_1_path, cv2.IMREAD_COLOR)
        frame2 =cv2.imread(frame_2_path, cv2.IMREAD_COLOR)

        # now calculate optical flow
        flow = torch.from_numpy(calc_optical_flow_dense(frame1, frame2))
        frame2 = TF.to_tensor(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))

        output = torch.stack((frame2[0], frame2[1], frame2[2], flow), 0)
 
        if self.transform:
            output = self.transform(output)
        if self.target_transform:
            label = self.target_transform(label)
        return output, label
    