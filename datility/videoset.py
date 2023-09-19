import os
import sys
import subprocess
import shutil
import pandas as pd
from typing import List, Tuple, Dict
from datility.dataset import DatasetBuilder
from datility.utils import _check_and_install

class VideoDatasetBuilder(DatasetBuilder):

    @classmethod
    def setup(cls, backend="ffmpeg"):
        print("VDB Setup starting...")
        print("Now installing standard dependencies...")
        # Check and install for all packages

        _check_and_install('pandas')

        print("Installing backend...")
        if backend == "ffmpeg":

            if shutil.which("ffmpeg") is None:
                print("Couldn't find ffmpeg, attempting to install...")
                if not _attempt_ffmpeg_install():
                    raise FileNotFoundError("FFMPEG is not installed, attempts to install failed.")

        elif backend=="opencv":

            if not _check_and_install('cv2'):
                raise ImportError("Unable to find opencv-python or install it.")      
 
            try: 
                import cv2
            except ImportError:
                raise ImportError("Unable to find opencv-python or install it.")
                
        else:
            print("Invalid Backend selection.")
            raise Exception
        
        print("Setup Complete!")

    def __init__(self, srcPath: str, labelPath:str, outPath: str, backend="ffmpeg"):
        super().__init__(srcPath, labelPath, outPath)

        self.video_converter_fn = self._ffmpeg_video_to_imageset if backend=="ffmpeg" else self._video_to_imageset

    def _write_to_annotation(self, out_path, line):
        ann_path = os.path.join(out_path, "annotations.txt")
        ann_file = open(ann_path, "a")
        ann_file.write('\n' + line)
        ann_file.close()

    ''' 
    - Split videos into frames and store in folders named after the label
    - Uses OpenCV as a backend
    '''
    def _video_to_imageset(self, vid_file_path, vid_dir_path, label):

        stream = cv2.VideoCapture(vid_file_path)    
        frames_total = stream.get(cv2.CAP_PROP_FRAME_COUNT)

        while stream.isOpened():
            frame_idx = stream.get(cv2.CAP_PROP_POS_FRAMES)

            if (frame_idx < frames_total):
                success, image = stream.read()

                if not success:
                    print(f"Failed on frame # {frame_idx}")
                    break
                
                frame_name = "img_{0:05d}.jpg".format(int(frame_idx))
                frame_path = os.path.join(vid_dir_path, frame_name)

                if not os.path.exists(frame_path):
                    cv2.imwrite(frame_path, image)
            else:
                break

            stream.set(cv2.CAP_PROP_POS_FRAMES, frame_idx+1)
        
        stream.release()
 
        annotation = os.path.relpath(vid_dir_path, self.out_path) + " " + str(0) + " " + str(int(frame_idx-1)) + " " + str(self._get_class(label))
        self._write_to_annotation(self.out_path, annotation)
 
    ''' 
    - Split videos into frames and store in folders named after the label
    - Uses ffmpeg as a backend
    '''
    def _ffmpeg_video_to_imageset(self, vid_file_path, vid_dir_path, label):

        frame_name = "img_%05d.jpg"
        frame_path = os.path.join(vid_dir_path, frame_name) 

        subprocess.run(["ffmpeg", "-r", str(1), "-i", vid_file_path, "-r", str(1), frame_path, "-threads", str(4), '-loglevel', 'quiet'])

        # count frames in video directory
        num_of_frames = len(list(filter(lambda x: os.path.isfile(x), os.listdir(vid_dir_path))))

        # write to annotation    
        annotation = os.path.relpath(vid_dir_path, self.out_path) + " " + str(0) + " " + str(num_of_frames-1) + " " + str(self._get_class(label))
        self._write_to_annotation(self.out_path, annotation)


    ''' Accepts a path of data, labels and outputs imagefolder datasets (train/test/etc)'''
    def build(self):

        video_file_rows = pd.read_csv(self.label_path)  

        # create annotations txt
        for i in range(len(video_file_rows)):

            # Original files are assumed to be mp4, but may not include extension in raw csv
            if str(video_file_rows.loc[i, "Filename"])[-4:] != '.mp4':
                filepath = os.path.join(self.src_path, (str(video_file_rows.loc[i, "Filename"]) + '.mp4'))
            else:
                filepath = os.path.join(self.src_path, str(video_file_rows.loc[i, "Filename"]))

            label = str(video_file_rows.loc[i, "Classification"])
            filename = self._get_filename(filepath) # get name of video without extension to write make the corresponding folder name
            
            if not os.path.exists(filepath):
                print(f"File at path '{filepath}' does not exist.")  # log
                continue

            if label == "":
                print(f"File at path '{filepath}' is does not contain a label.")  # log
                continue

            # create the class or label folder 
            label_dir_path = os.path.join(self.out_path, label)
            if not os.path.exists(label_dir_path):
                os.mkdir(label_dir_path)

            # create the folder for the video
            video_dir_path = os.path.join(label_dir_path, filename)
            if not os.path.isdir(video_dir_path):
                os.mkdir(video_dir_path) 

            # print(f"Converting {filepath} ...") #log
            self.video_converter_fn(filepath, video_dir_path, label)
            

'''
Input: a list of tuple pairs ("dataset-id", "path-to-annotation")
Output: a string of lines for a combined annotation file
'''
def mergeDatasets(datasets: List[Tuple[str, str]]): 
    merged_ann = list()
    for ds in datasets:
        # open file
        with open(ds[1], "r") as annotations:
            lines = annotations.readlines()
            for line in lines:
                if line != '\n':
                    params = line.split()
                    params[0] = os.path.join(ds[0], params[0])
                    merged_ann.append(" ".join(params) + '\n')

    return merged_ann
        

def _attempt_ffmpeg_install():
            error_msg = ""
            error_msg += "ERROR: Git CLI is not installed!\n" if not shutil.which("git") else ""
            error_msg += "ERROR: Make CLI is not installed!\n" if not shutil.which("make") else ""

            if len(error_msg) > 0:
                error_msg += "Please install missing dependencies and retry!\n"
                print(error_msg)
                return False

            subprocess.run(["git", "clone", "https://git.ffmpeg.org/ffmpeg.git", "ffmpeg"])
            subprocess.run(["./configure"], cwd="./ffmpeg")
            subprocess.run(["make"], cwd="./ffmpeg")
            subprocess.run(["sudo", "make", "install"], cwd="./ffmpeg")
            subprocess.run(["rm", "-rf", "ffmpeg"])

            if shutil.which("ffmpeg") is None:
                error_msg += "ERROR: Failed to install ffmpeg!"
                return False
           
            return True

