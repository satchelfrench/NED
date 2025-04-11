# Neonatal Echocardiographic Dataset (NED)

Welcome to the repository for the **Neonatal Echocardiographic Dataset (NED)**. This repository provides tools and scripts to facilitate the use of NED, a professionally annotated dataset designed to advance research in neonatal echocardiography.

## 🩺 Overview

Echocardiography is a vital tool for cardiac assessment, offering clinicians critical information swiftly and affordably. However, its broader accessibility is often limited by the need for highly skilled technicians.

Our study introduces a novel approach to echocardiographic viewpoint classification by treating it as a **video classification** problem rather than an image classification task. We propose a **CNN-GRU architecture** with a novel **Temporal Feature Weaving (TFW)** method, which leverages both spatial and temporal information to yield a **4.33% increase in accuracy** over baseline image classification while using only four consecutive frames—incurring minimal computational overhead.

In addition, we present the **Neonatal Echocardiogram Dataset (NED)**: a professionally annotated dataset offering sixteen viewpoint classes from echocardiography videos. This release aims to encourage future development and benchmarking in neonatal cardiac AI.

📄 **Paper**: [Temporal Feature Weaving for Neonatal Echocardiographic Viewpoint Video Classification](https://arxiv.org/abs/2501.03967)  
📁 **Repo**: [https://github.com/satchelfrench/NED](https://github.com/satchelfrench/NED)

Download Dataset [Here](https://sagemaker-studio-685595588466-uuryx8ysrkm.s3.us-west-1.amazonaws.com/echo_videos.tar.gz)

---

## ✨ Features

- **Neonatal Echocardiogram Dataset (NED)**:  
  - 16 viewpoint classes  
  - 1,049 professionally annotated videos (~1s each)  
  - Suitable for benchmarking spatial-temporal models

- **Temporal Feature Weaving (TFW)**:  
  - A novel method that integrates spatial and temporal features  
  - Achieves superior performance with minimal additional compute

- **PyTorch Dataloaders**:  
  - Load frames as consecutive or evenly spaced sequences  
  - Sequence-wide augmentations: rotation, scale, shift, flip, contrast, and more

---

## 📁 Repository Structure

NED/ 
├── dataloaders/ # PyTorch dataloaders for NED 
├── datility/ # Utility scripts for data processing 
├── notebooks/ # Training & evaluation notebooks 
├── balance.py # Dataset balancing utility 
├── create_flowset.py # Create flow-based dataset 
├── create_videoset.py # Create video-based dataset

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.9.0+
- Other dependencies listed in `pyproject.toml`

### Installation

```bash
git clone https://github.com/satchelfrench/NED.git
cd NED
pip install -r requirements.txt
```

### Usage

Dataset Preparation: Use scripts to format, balance, and augment the dataset.
Model Training: Train the CNN-GRU + TFW architecture with the provided notebooks.
Evaluation: Evaluate classification accuracy on the test split with provided metrics.

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


For more details, please refer to our paper and the project repository.

