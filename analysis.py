import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from PIL import Image
import cv2

from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils import to_dataframe, plot_examples


def get_summary(df):
    print(df.head())
    print(df.describe())


def display_images(df):
    # Display images of the dataset with their labels
    random_index = np.random.randint(0, len(df), 8)
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12, 8),
                            subplot_kw={'xticks': [], 'yticks': []})

    for i, ax in enumerate(axes.flat):
        ax.imshow(plt.imread(df.image_path[random_index[i]]))
        ax.set_title('field' if df.label[random_index[i]]==0 else 'road')
    plt.tight_layout()
    plt.show()


def get_image_count_per_label(df):
    fc = df["label"].value_counts()
    plt.figure(figsize=(6,4))
    sns.barplot(x = fc.index, y = fc, palette = "crest")
    plt.title("Number of pictures of each category", fontsize = 15)
    plt.xticks(rotation=90)
    plt.show()


def show_augmentations(image):
    augs = [
            #A.RandomCrop(width=180, height=112, p=0.6),
            A.Rotate(limit=20, p=0.3, border_mode=cv2.BORDER_CONSTANT),
            A.HorizontalFlip(p=0.2),
            A.VerticalFlip(p=0.1),
            A.OneOf(
                [
                    A.Blur(blur_limit=3, p=0.3),
                    A.ColorJitter(p=0.3),
                ],
                p=1.0,
            ),
            A.Resize(width=112, height=112),
        ]
    transform_aug = A.Compose(augs)
    images_list = [image]
    image = np.array(image)
    for i in tqdm(range(24)):
        augmentations = transform_aug(image=image)
        augmented_img = augmentations["image"]
        images_list.append(augmented_img)
    plot_examples(images_list)


if __name__ == "__main__":
    df = to_dataframe()

    #get_image_count_per_label(df)
    #get_summary(df)
    #display_images(df)

    image = Image.open('dataset/fields/2.jpg')
    #show_augmentations(image)