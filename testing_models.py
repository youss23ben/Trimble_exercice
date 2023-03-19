import seaborn as sns
import matplotlib.pyplot as plt

import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
import torchvision

from datasets import TestDataset
from models import CNN, Big_CNN, VGG_model
from augmentations import transform


def get_accuracies(trained_model, model_name, augment):

    # creating directory to save results
    save_dir = 'graphs_and_results/test_results'
    os.makedirs(save_dir, exist_ok=True)

    # loading test dataset
    test_dataset = TestDataset("dataset/test_images", transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # create figure for plotting results
    fig = plt.figure(figsize=(16, 16))

    # switching model to evaluation mode
    trained_model.eval()

    # checking if GPU is available
    CUDA = torch.cuda.is_available()

    num_correct = 0
    num_total = 0

    # looping over test data
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_dataloader):
            if CUDA:
                inputs = inputs.cuda()
                labels = labels.cuda()

            # forward pass
            outputs = model(inputs)
            _, predictions = torch.max(outputs.data, 1)

            # calculating accuracy
            num_correct += (predictions == labels).sum().item()
            num_total += labels.size(0)

            # plotting
            ax = plt.subplot(5, 2, i + 1) # create subplot for each image
            ax.axis('off')
            ax.set_title(f'predicted: {predictions.item()}, actual: {labels.item()}')
            img = inputs.squeeze().permute(1, 2, 0).cpu().numpy()
            img = (img * 0.5) + 0.5  # denormalizing the image
            ax.imshow(img)
            if i == 9:
                break

    accuracy = num_correct / num_total
    print(f"Accuracy on test dataset: {accuracy:.2f}")
    plt.savefig(os.path.join(save_dir, f'test_samples_{model_name}_augment_{augment}.png'))
    return accuracy


if __name__ == "__main__":

    #testing and getting accuracy results for the 6 saved models
    accuracies = {}
    for model_name in ["cnn", "big_cnn", "vgg"]:
        for augment in [True, False]:
            if model_name == "cnn":
                model = CNN()
            elif model_name == "big_cnn":
                model = Big_CNN()
            else:
                model = VGG_model()

            model_state_dict = torch.load(f'saved_models/best_model_{model_name}_augment_{augment}.pt')
            model.load_state_dict(model_state_dict)
            accuracy = get_accuracies(trained_model=model, model_name=model_name, augment=augment)
            accuracies[f'{model_name}_augment_{augment}'] = accuracy

    print(accuracies)