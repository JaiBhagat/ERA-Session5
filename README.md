# MNIST Handwritten Digit Recognition with PyTorch

This repository provides a solution for the MNIST handwritten digits classification task using PyTorch.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [File Descriptions](#file-descriptions)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)

## Introduction

This repository contains Python scripts for training and testing a Convolutional Neural Network (CNN) model on the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) for recognizing handwritten digits. 

## Features

- Utilization of PyTorch for efficient computations.
- Preprocessing of the MNIST dataset with data augmentation techniques.
- Visualization of processed images and their labels.
- Training a CNN model on the processed data.
- Evaluation of the trained model on the test data.
- Plotting of accuracy and loss during training and testing.

## Requirements

- Python 3.7+
- PyTorch 1.7+
- torchvision
- Matplotlib
- CUDA (recommended for faster computations)

## Installation

1. Clone the repository:

    ```
    git clone https://github.com/JaiBhagat/ERA-Session5.git
    ```

2. Move to the project directory:

    ```
    cd ERA-Session5
    ```

3. Install necessary packages:

    ```
    pip install -r requirements.txt
    ```

## File Descriptions

- `utils.py`: This script includes all utility functions necessary for the pipeline, such as preprocessing the MNIST dataset, plotting the data, training the model, and testing it.

- `model.py`: This script defines the structure of the CNN model used for the task. It includes convolutional layers and fully connected layers.

- `S5.ipynb`: The main Jupyter notebook that orchestrates the entire process. It imports functions from `utils.py` and the model from `model.py` to train and test the model on the MNIST dataset.

## Usage

- Run the `S5.ipynb` notebook in a Jupyter notebook environment. If CUDA is available on your machine, the code will automatically use it to speed up computations.

- The notebook includes the following steps:

    1. **Data preprocessing**: Application of transformations on the MNIST dataset for data augmentation and normalization.
    2. **Model training**: Training the CNN model using the preprocessed training set.
    3. **Model testing**: Evaluation of the trained model on the test dataset and print of the loss and accuracy.
    4. **Performance visualization**: Plotting of accuracy and loss graphs for both the training and test datasets.

## Results

The final trained model achieves an accuracy of 99.29%% on the MNIST test set.

## Contributing

If you have suggestions for improving this code, please feel free to open a pull request. We appreciate your inputs!

