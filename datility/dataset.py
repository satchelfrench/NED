from abc import ABC, abstractmethod
import os

class DatasetBuilder(ABC):

    id_count = 0
    classes = dict()

    def __init__(self, srcPath: str, labelPath:str, outPath: str):
        if not os.path.isdir(srcPath):
            raise FileNotFoundError

        if not os.path.exists(labelPath):
            raise NotADirectoryError

        if not os.path.isdir(outPath):
            os.mkdir(outPath)

        self.src_path = srcPath 
        self.label_path = labelPath 
        self.out_path = outPath

    
    @classmethod
    def _get_class(cls, label: str):
        if not cls.classes.get(label):
            cls.classes[label] = cls.id_count
            cls.id_count += 1
        
        return cls.classes[label]

    def _get_class_map(self):
        return self.classes

    def _get_filename(self, path, ext=False):
        f = os.path.split(path)[1]
        if ext:
            return f
    
        return f.split('.')[0]

    @abstractmethod
    def _write_to_annotation(self, out_path, line):
       pass

    ''' Split videos into frames and store in folders named after the label '''
    @abstractmethod
    def _video_to_imageset(self, vid_file_path, vid_dir_path, label):
       pass

    ''' Accepts a path of data, labels and outputs imagefolder datasets (train/test/etc)'''
    @abstractmethod
    def build(self):
        pass

