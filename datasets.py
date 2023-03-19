import os
from pathlib import Path

from torch.utils.data import Dataset

import cv2
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir  # the directory containing the image data
        self.transform = transform  # the transformation to apply to images
        self.image_paths = []  # this list will store the paths to images
        self.labels = []  # this list will store the labels

        for label in os.listdir(data_dir):  # label is the name of the folder
            if label == "fields":
                label_idx = 0
            elif label == "roads":
                label_idx = 1
            else:  # if the label is not "fields" or "roads" meaning the test dataset
                continue  # skip, I'll tackle test data in another class

            label_dir = os.path.join(data_dir, label)  # getting the full path for the current label
            for image_name in os.listdir(label_dir):
                image_path = os.path.join(label_dir, image_name)  # getting the full path to the image
                self.image_paths.append(image_path)
                self.labels.append(label_idx)

    def __len__(self):
        return len(self.image_paths)  # return the length of the image paths list

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]  # getting the path to the image at the specified index
        label = self.labels[idx]  # getting the label of the image at the specified index
        image = Image.open(image_path).convert("RGB")  # loading the image and converting it to RGB
        if self.transform:
            image = self.transform(image)
        return image, label  # retufn the transformed image and its label


# The following is another class that reads data from dataframe format
# I use this after applying oversampling to the minority class (field)
class OversampledDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.df.iloc[idx, 0]
        image = Image.open(image_path).convert('RGB')
        label = self.df.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)
        return image, label


#class to load the test dataset
class TestDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_list = os.listdir(self.data_dir)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_list[idx])
        img = Image.open(img_path)
        img = self.transform(img)
        label = 0 if img_path.split("_")[2].split(".")[0] == 'field' else 1
        return img, label
