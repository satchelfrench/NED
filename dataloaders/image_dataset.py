import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from neo_echoset.datility.utils import calc_optical_flow_dense
import torchvision.transforms.functional as TF
import torch
import pickle
import shelve
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
    def __init__(self, annotations_file, root_dir, transform=None, target_transform=None, cache_enabled=True):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.data = []
        self.cache_enabled = cache_enabled
        self.cache = shelve.open("of_dataset_cache.shelve")

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
        idx_utf = str(idx)
        # transforms must be applied first
        # augmentations not recommended as they will be cached and returned with 100% chance
        # if using with torch augmentations, turn off cache
        # Best way to handle augmentations for this dataset is technically with offline augmentations 
        # suboptimal for now, recommend using alternate opticalflow calculation and avoid numpy<->tensor
        # consider simply normalizing optical flow maps against image dimensions and into range -1,1 or -255 to 255
        # caould also combine x and y maps into a single feature with e, and
        
        if self.cache_enabled and idx_utf in self.cache:
            output, label = self.cache[idx_utf]
        else:
            images, label = self.data[idx]
            frame_1_path, frame_2_path = images

            frame1 = cv2.imread(frame_1_path, cv2.IMREAD_COLOR)
            frame2 = cv2.imread(frame_2_path, cv2.IMREAD_COLOR)

            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            # perform transform and then convert back to numpy
            # these transforms can be different between frames!
            if self.transform:
                frame1 = self.transform(TF.to_tensor(frame1))
                frame2 = self.transform(TF.to_tensor(frame2))
            
                frame1 = frame1.view(frame1.size(dim=1), frame1.size(dim=2))
                frame2 = frame2.view(frame2.size(dim=1), frame2.size(dim=2))

            
            # now calculate optical flow
            # flow = torch.from_numpy(calc_optical_flow_dense(frame1, frame2))
            flow = torch.from_numpy(cv2.calcOpticalFlowFarneback(frame1.numpy(), frame2.numpy(), None, 0.5, 3, 15, 3, 5, 1.2, 0))

            output = torch.stack((frame2, flow[...,0], flow[...,1]), 0)
            
            if self.cache_enabled:
                self.cache[idx_utf] = (output, label)

        if self.target_transform:
            label = self.target_transform(label)

        return output, label
    