import os
import pandas as pd
from torchvision.io import read_image
import pickle

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
                        tuple(
                            os.path.join(video_path, frame_path),
                            parsed[-1]
                        )
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
                            parsed[-1]
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