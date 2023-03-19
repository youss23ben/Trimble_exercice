import os
from pathlib import Path

import cv2
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for label in os.listdir(data_dir):
            if label == "fields":
                label_idx = 0
            elif label == "roads":
                label_idx = 1
            else:
                continue

            label_dir = os.path.join(data_dir, label)
            for image_name in os.listdir(label_dir):
                image_path = os.path.join(label_dir, image_name)
                self.image_paths.append(image_path)
                self.labels.append(label_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        #image = np.array(Image.open(image_path).convert("RGB"))
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            #image = self.transform(image=image)["image"]
            image = self.transform(image)
        return image, label


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
