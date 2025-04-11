# Neonatal Echocardiographic Dataset (NED)

Welcome to the repository for the **Neonatal Echocardiographic Dataset (NED)**. This repository provides tools and scripts to facilitate the use of NED, a professionally annotated dataset designed to advance research in neonatal echocardiography.

The code in this repo is meant to accompany our paper ![Temporal Feature Weaving for Neonatal Echocardiographic Viewpoint Video Classification](https://arxiv.org/abs/2501.03967) accepted at ![ISBI 2025](https://biomedicalimaging.org/2025/).

## 🩺 Overview

Echocardiography is a vital tool for cardiac assessment, offering clinicians critical information swiftly and affordably. However, its broader accessibility is often limited by the need for highly skilled technicians.

Our study introduces a novel approach to echocardiographic viewpoint classification by treating it as a **video classification** problem rather than an image classification task. We propose a **CNN-GRU architecture** with a novel **Temporal Feature Weaving (TFW)** method, which leverages both spatial and temporal information to yield a **4.33% increase in accuracy** over baseline image classification while using only four consecutive frames—incurring minimal computational overhead.

In addition, we present the **Neonatal Echocardiogram Dataset (NED)**: a professionally annotated dataset offering sixteen viewpoint classes from echocardiography videos. This release aims to encourage future development and benchmarking in neonatal cardiac AI.

More can be read about our paper below:

📄 **Paper**: [Temporal Feature Weaving for Neonatal Echocardiographic Viewpoint Video Classification](https://arxiv.org/abs/2501.03967)  

A direct download of organized scans can be downloaded below, however we have created a script which auto-handles all dependencies, downloads and processes the raw files into a structured dataset. To use see the **Usage** section.

💾 Download Dataset [Here](https://sagemaker-studio-685595588466-uuryx8ysrkm.s3.us-west-1.amazonaws.com/echo_videos.tar.gz)


<img src="https://github.com/user-attachments/assets/27ffedc7-2f17-4dae-b1d1-08e44098f8ce" width="400"/>


---

## ✨ Features

- **Neonatal Echocardiogram Dataset (NED)**:  
  - 16 viewpoint classes (or 12 in NED-12 subset)

  - 1,049 professionally annotated videos (~1s each)
    
- **Dataloaders, Preprocessing and Tools**:  
  - Load frames as consecutive or evenly spaced sequences
  - Sequence-wide augmentations: rotation, scale, shift, flip, contrast, and more
  - Dataloaders or preprocess scripts for optical flow




## 📖 About our work

Our paper introduces the concept **Temporal Feature Weaving (TFW)**. This approach is based on the assumption that viewpoints are best identified when the dynamic function of the heart is observed, as such we must factor in the structure of anatomy and dynamics of heart function into our prediction. Using TFW we encourage the model to better understand the spatio-temporal information in echocardiograms by grouping spatial features from the same scan across different frames. This is thought to introduce better feature sparsity and call to the models attention the change of high-level spatial features over time. 

In our testing it produces superior results against baseline image classificaiton, majority vote and attention approaches:

<img width="793" alt="Screenshot 2025-04-11 at 1 41 38 PM" src="https://github.com/user-attachments/assets/40496a54-0d3a-4237-a17b-53de7bc17a1d" />

Below are the patient statistics for the dataset:

<img width="797" alt="Screenshot 2025-04-11 at 1 41 26 PM" src="https://github.com/user-attachments/assets/068aee29-1201-4f24-b098-0ea0ac658d0e" />
---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- PyTorch 1.9.0+
- Other dependencies listed in `pyproject.toml`

### Installation & Usage

Run the below commands to clone repository and build the dataset:

```bash

git clone https://github.com/satchelfrench/NED/ ./neo_echoset
cd ./neo_echoset/ && poetry install --no-ansi
poetry run python create_videoset.py --kfold 5 # check script for config options

# optional:
poetry run python create_flowset.py # check script for config
```
**Example of using Dataloder:**

We adopt a functional style of returning the dataset, here `build_dataloders` just returns the dataset and loader.
You may select the number number of segments to divide a clip into, as well as the number of frames from each segment to select. Code for this loader is heavily based on the ![Video-Dataset-Loading-Pytorch](https://github.com/RaivoKoot/Video-Dataset-Loading-Pytorch) library.

Sequence wide augmentations are encouraged as shown below. Simple example of applying a rotation to an entire sequence of frames.

```python

class RandomSeqRotation(RandomRotation):
    def forward(self, images):
        fill = self.fill
        angle = self.get_params(self.degrees)

        transformed = []

        for img in images:
            channels, _, _ = Fn.get_dimensions(img)

            if isinstance(img, Tensor):
                if isinstance(fill, (int, float)):
                    fill = [float(fill)] * channels
                else:
                    fill = [float(f) for f in fill]
            angle = self.get_params(self.degrees)

            transformed.append(Fn.rotate(img, angle, self.interpolation, self.expand, self.center, fill))

        return transformed

```

```python
from neo_echoset.dataloaders.video_dataset import VideoFrameDataset, ImglistToTensor
from torchvision.transforms import transforms
import torch.utils.data as data


# Make this work with multiple dataset types..
def build_dataloaders(dataset_root, annotations_path, preprocess, params, shuffle, drop_last, pin_memory, test_mode=False):
  dataset = VideoFrameDataset(
    root_path=dataset_root,
    annotationfile_path=annotations_path,
    transform=preprocess,
    num_segments=4,
    frames_per_segment=1,
    imagefile_template='img_{:05d}.jpg',
    test_mode=test_mode
  )

  dataloader = data.DataLoader(dataset, batch_size=params.batch_size,
                                 shuffle=shuffle, drop_last=drop_last, num_workers=8,
                                 pin_memory=pin_memory)

  return dataset, dataloader

```

More detailed training / testing instructions are available in **/notebooks**



### 📝 Citation

If you use this repository or dataset, please cite our work:

```yaml
@article{french2024temporal,
  title={Temporal Feature Weaving for Neonatal Echocardiographic Viewpoint Video Classification},
  author={French, Satchel and Zhu, Faith and Jain, Amish and Khan, Naimul},
  journal={arXiv preprint arXiv:2501.03967},
  year={2024}
}
```

### 🤝 Acknowledgments

This work was supported by the Natural Sciences and Engineering Research Council of Canada (NSERC) under Alliance Grant 546302-19 and Discovery Grant 2020-05471.

