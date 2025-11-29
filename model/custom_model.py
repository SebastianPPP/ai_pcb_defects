from pathlib import Path

import numpy as np
import cv2
from ultralytics import YOLO
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from torchvision.io import decode_image
import os
from read_labels import read_file_labels, read_image_to_matrix
import torch.nn as nn
import torch.nn.functional as F

# data dirs
TRAIN_IMG_PATH = Path('./data_custom/train/images')
TRAIN_LABEL_PATH = Path('./data_custom/train/labels')

TEST_IMG_PATH = Path('./data_custom/test/images')
TEST_LABEL_PATH = Path('./data_custom/test/labels')

VAL_IMG_PATH = Path('./data_custom/val/images')
VAL_LABEL_PATH = Path('./data_custom/val/labels')
# load data
X_train = np.array([])
y_train = np.array([])

for i in tqdm(range(0, len([path for path in TRAIN_IMG_PATH.iterdir()]))):
    labels = read_file_labels(list(TRAIN_LABEL_PATH.iterdir())[i])
    for label in labels:
        X_train = np.append(X_train, read_image_to_matrix(list(TRAIN_IMG_PATH.iterdir())[i]))
        y_train = np.append(y_train, label)
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)

X_test = np.array([])
y_test = np.array([])
for i in tqdm(range(0, len([path for path in TEST_IMG_PATH.iterdir()]))):
    labels = read_file_labels(list(TEST_LABEL_PATH.iterdir())[i])
    for label in labels:
        X_test = np.append(X_test, read_image_to_matrix(list(TEST_IMG_PATH.iterdir())[i]))
        y_test = np.append(y_test, label)
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)


X_val = np.array([])
y_val = np.array([])
for i in range(0, tqdm(len([path for path in VAL_IMG_PATH.iterdir()]))):
    labels = read_file_labels(list(VAL_LABEL_PATH.iterdir())[i])
    for label in labels:
        X_val = np.append(X_val, read_image_to_matrix(list(VAL_IMG_PATH.iterdir())[i]))
        y_val = np.append(y_val, label)
np.save('X_val.npy', X_val)
np.save('y_val.npy', y_val)

print(len(X_train), len(y_train), len(X_val), len(y_val), len(X_test), len(y_test))
# map data

# preprocessing

# Dataset -> Dataloader

class CustomDataset(Dataset):
    def __init__(self, label_file, img_dir, transform = None, target_transform = None):
        super().__init__()
        self.img_labels = pd.read_csv(label_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[index,0])
        image = decode_image(img_path)
        label = self.img_labels.iloc[index,1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
train_dataloader = DataLoader(X_train, batch_size=64, shuffle=True)
test_dataloader = DataLoader(X_test, batch_size=64, shuffle=True)
val_dataloader = DataLoader(X_val, batch_size=64, shuffle=True)
# model architecture

class CNN_model(nn.module):
    def __init__(self, num_classes=6):
        super(CNN_model, self).__init__()
        self.num_classes = num_classes
        self.conv2d_1 = nn.Conv2d(3, 16, kernel_size = 4, stride = 2)
        self.conv2d_2 = nn.Conv2d(16, 16, kernel_size = 4, stride = 2)
        self.fc1 = nn.Linear(6400, 128)
        self.fc2 = nn.Linear(128, 6)

    def forward(self, x):
        x = self.conv2d_1(x)
        x = F.relu(x)
        x = self.conv2d_2(x)
        x = torch.flatten(x,1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output

# training

