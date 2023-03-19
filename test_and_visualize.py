import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models

from datasets import TestDataset


def test_models(model):

    test_dataset = TestDataset("dataset/test_images", transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    fig = plt.figure(figsize=(16, 16))

    # Testing
    model.eval()

    CUDA = torch.cuda.is_available()

    with torch.no_grad():
        num_correct = 0
        num_total = 0
        for i, (inputs, labels) in enumerate(test_dataloader):
            if CUDA:
                inputs = inputs.cuda()
                labels = labels.cuda()

            # forward pass
            outputs = model(inputs)
            _, predictions = torch.max(outputs.data, 1)

            # calculate accuracy
            num_correct += (predictions == labels).sum().item()
            num_total += labels.size(0)

            # plotting
            ax = plt.subplot(5, 2, i + 1)
            ax.axis('off')
            ax.set_title(f'predicted: {predictions.item()}, actual: {labels.item()}')
            img = inputs.squeeze().permute(1, 2, 0).cpu().numpy()
            img = (img * 0.5) + 0.5  # Denormalize the image
            ax.imshow(img)
            if i == 9:
                break

    accuracy = num_correct / num_total
    print(f"Accuracy on test dataset: {accuracy:.2f}")
    plt.show()


if __name__ == "__main__":
    # testing on cnn without augmentation
    model = CNN()
    model_state_dict = torch.load('best_model_cnn_augment_False.pt')
    model.load_state_dict(model_state_dict)
    test_models(model=model)