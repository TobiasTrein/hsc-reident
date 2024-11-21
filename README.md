# README for Hello Street Cat - Reidentification Code

## Overview

Welcome to the **Hello Street Cat** reidentification project! This repository focuses on building machine learning and deep learning models for visual reidentification of cats. It uses a Siamese Network Model to identify and match instances of cats from image datasets.

The project includes functionality for:
- Training and evaluating models for cat instance reidentification.
- Data augmentation pipelines for improved generalization.
- Model architectures leveraging pre-trained networks such as EfficientNet, VGG16, and MobileNet variants.

---

## Prerequisites

### Hardware
- A machine with GPU support for training deep learning models (optional but recommended).

### Software
- Python 3.8 or later.
- Install the dependencies listed in `requirements.txt`:
  ```bash
  pip install -r requirements.txt
  ```

---

## Dependencies

The key libraries used in the project include:
- **Core ML/Deep Learning**:
  - TensorFlow
  - Keras
  - Scikit-learn
- **Visualization**:
  - Matplotlib
  - Seaborn
  - Visualkeras
- **Image Processing**:
  - OpenCV
  - PIL
  - `imgaug`
- **Others**:
  - NumPy
  - Pandas
  - tqdm

---

## How to Run

To prepare your dataset, follow these steps:

### 1. Download the Dataset
First, download the dataset from Kaggle by visiting the following link:  
[HeelLostStreetCat Individuals Dataset](https://www.kaggle.com/datasets/tobiastrein/heellostreetcat-individuals).

Once downloaded, unzip the dataset.

Ensure the dataset is organized in the following structure:

```
<dataset_directory>/
├── <cat_folder_1>/
│   ├── top/
│   └── front/
├── <cat_folder_2>/
│   ├── top/
│   └── front/
└── ...
```

- Each `<cat_folder_x>` should represent an individual cat, containing two subfolders: `top` and `front`.
- The `top` and `front` folders will contain images captured from different angles.

This organization will help with processing the images in the project.

### 2. Train the Model

To train the model, you only need to execute the script via Python. The hyperparameters are set in the following lists.You can modify them as you wish:

    photo_types = ['top', 'front', 'all']
    base_models = ['vgg']
    loss_function = ['triplet']
    num_epochs = [100]
    learning_rates = [0.0001]
    augmentation = ['none', 'flip', 'noise', 'rotate']

The program will automatically iterate through every combination of these hyperparameters and train the model accordingly. The results of each combination will be saved in a CSV file for further analysis.

---

## Features

1. **Pre-trained Base Models**:
   - EfficientNet, VGG16, MobileNet, EfficientNet.
   - Transfer learning for faster convergence and improved accuracy.

2. **Data Augmentation**:
   - Horizontal flips, rotations, and noise addition.
   - Customizable augmentation pipelines for flexibility.

3. **Pair/Triplet Generation**:
   - Generate positive and negative image pairs or triplets for training.

4. **Loss Functions**:
   - Contrastive Loss: Focuses on minimizing/maximizing pairwise distances.
   - Triplet Loss: Optimizes relationships among anchor, positive, and negative samples.

5. **Visualization**:
   - Plot model architecture using `visualkeras`.
   - Visualize augmentation effects.

---

## Acknowledgments

This work is part of the research for the article Siamese Networks for Cat Re-Identification: Exploring Neural Models for Cat Instance Recognition. 