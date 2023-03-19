import pandas as pd
import matplotlib.pyplot as plt

import os

from datasets import ImageDataset


def to_dataframe():
    dataset = ImageDataset('dataset')
    df = pd.DataFrame({
        "image_path": dataset.image_paths,
        "label": dataset.labels
    })
    return df


def plot_examples(images):
    fig = plt.figure(figsize=(15, 15))
    columns = 6
    rows = 5

    for i in range(1, len(images)):
        img = images[i - 1]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()


def save_train_curves(train_losses, val_losses, val_f1_score, model_name, augment):
    """
    Plots the train and validation loss and accuracy over epochs.

    Args:
        train_losses (list): List of training losses over epochs.
        val_losses (list): List of validation losses over epochs.
        val_f1_score (list): List of validation f1 score over epochs.
    """

    # creating directory to save results
    save_dir = 'graphs_and_results/graphs'
    os.makedirs(save_dir, exist_ok=True)

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(15, 5))

    # plot and save training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'bo-', label='Training loss')
    plt.plot(epochs, val_losses, 'ro-', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'loss_curves_{model_name}_augment_{augment}.png'))

    # plot and save f1 score
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_f1_score, label='Validation f1_score')
    plt.title('validation f1_score')
    plt.xlabel('Epochs')
    plt.ylabel('F1_score')
    plt.legend()

    plt.savefig(os.path.join(save_dir, f'f1_curves_{model_name}_augment_{augment}.png'))