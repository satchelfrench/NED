from datility.videoset import VideoDatasetBuilder, mergeDatasets
from datility.structures import Case
from sklearn.model_selection import KFold
import os
import argparse
import random
from tqdm import tqdm
import wget
import tarfile
import json
import pickle

def split_prob(arr, prob):
    train, test = [], []

    while arr:
        sample = arr.pop()
        r = random.random()
        if r >= (1-prob):
            train.append(sample)
        else:
            test.append(sample)
    return train, test

def write_annotation_file(train_cases, valid_cases, train_filename, valid_filename):
    train_annotations_path = os.path.join(args.out_path, train_filename)
    with open(train_annotations_path, "w") as train_labels:
        train_merge_list = []
        for case in train_cases:
            train_merge_list.append(
                tuple([case.src.split('/')[-1], os.path.join(case.out, "annotations.txt")])
            )
        train_labels.writelines(mergeDatasets(train_merge_list))

    valid_annotations_path = os.path.join(args.out_path, valid_filename)
    with open(valid_annotations_path, "w") as valid_labels:
        valid_merge_list = []
        for case in valid_cases:
            valid_merge_list.append(
                tuple([case.src.split('/')[-1], os.path.join(case.out, "annotations.txt")])
            )
        valid_labels.writelines(mergeDatasets(valid_merge_list))


# # neonatal echocardiogram dataset (neo-echoset)
if __name__ == '__main__':

    # get cwd
    base_path = os.getcwd()
    default_data_path = os.path.join(base_path, "neo-echoset-raw")
    default_out_path = os.path.join(base_path, "neo-echoset")
    default_download_url = "https://sagemaker-studio-685595588466-uuryx8ysrkm.s3.us-west-1.amazonaws.com/echo_videos.tar.gz"

    parser = argparse.ArgumentParser()

    # Parse args
    parser.add_argument('--data_path', default=str(default_data_path), type=str,
                        help="path to raw data, argument is absolute path.")
    parser.add_argument('--out_path', default=str(default_out_path), type=str,
                        help="path to output dataset, argument is absolute path.") 
    parser.add_argument('--url', default=str(default_download_url), type=str,
                        help="default url for downloading raw data")
    parser.add_argument('--seed', default=42, type=int,
                        help="random seed for split, 42 is default and standard for comparison")
    parser.add_argument('--split', default=0.80, type=float,
                        help="probability of a patient case being added to train set, remainder goes to test. \
                            [0.0 - 1.0], default=0.8")
    parser.add_argument("--label_file_name", default="labels.csv", type=str,
                        help="expected format of case label file (default to labels.csv)")
    parser.add_argument("--kfold", default=1, type=int, help="number of folds for dataset")

    args = parser.parse_args()

    # validate arguments
    if args.split > 1.0 or args.split < 0.0:
        print("Invalid value supplied. \n --split argument is constrained to [0.0 - 1.0], setting to default 0.8..")
        args.split = 0.8

    # check the paths are valid and exist, if the don't then download & unzip from source
    if not os.path.exists(args.out_path):
        print(f"Creating output directory.. at {args.out_path}")
        os.mkdir(args.out_path)

    if not os.path.exists(args.data_path):
        if args.data_path is not default_data_path:
            raise FileNotFoundError(f"The path provided '{args.data_path}' does not exist.")
        else:
            # download and untar
            print("Downloading dataset..")
            raw_file_path = os.path.join(base_path, "neo-echoset-raw.tar.gz")
            wget.download(args.url, raw_file_path)
            
            print("\nDecompressing..")
            with tarfile.open(raw_file_path, 'r') as tar:
                tar.extractall(args.data_path)

    # Format all source cases into named tuples  
    args.data_path = os.path.join(args.data_path, "echos_video") # removable when fixing s3 file
    dirs = filter(lambda x: os.path.isdir(os.path.join(args.data_path, x)), sorted(os.listdir(args.data_path)))

    cases = list()
    for dir in dirs:
        dir_path = os.path.join(args.data_path, dir)
        cases.append(Case(dir_path,
                        os.path.join(dir_path, args.label_file_name),
                        os.path.join(args.out_path, dir)))


    # run check and install backends
    VideoDatasetBuilder.setup()

    print("Building dataset..")
    for case in tqdm(cases):
        ds = VideoDatasetBuilder(case.src, case.labels, case.out)
        ds.build()

    # Save Class Map
    print("Saving class map as json and pickle dump..")
    with open("classes.txt", 'w') as classFile:
        classFile.write(json.dumps(ds._get_class_map()))

    with open("classes.pkl", 'wb') as classPkl:
        classPkl.write(pickle.dumps(ds._get_class_map()))

    # Split Cases into train/val/test
    random.seed(args.seed)
    
    if args.kfold > 1:
        # Do KFold
        fold = KFold(args.kfold)
        data_files = []
        for i, (train_split, valid_split) in enumerate(fold.split(cases)):
            train_cases = [cases[x] for x in train_split]
            valid_cases = [cases[x] for x in valid_split]
            train_file = f"train_annotations_{i}.txt"
            valid_file = f"valid_annotations_{i}.txt"
            data_files.append((train_file, valid_file))
            write_annotation_file(train_cases, valid_cases, train_file, valid_file)

        print("Saving KFolded splits as json and pickle dump..")
        with open("folded_data.txt", 'w') as foldedFile:
            foldedFile.write(json.dumps(data_files))

        with open("folded_data.pkl", 'wb') as foldedPkl:
            foldedPkl.write(pickle.dumps(data_files))

    else:   
        train_cases, valid_cases = split_prob(cases, args.split)
        write_annotation_file(train_cases, valid_cases, "train_annotations.txt", "valid_annotations.txt")
    
    print("Complete!\nExiting..")

