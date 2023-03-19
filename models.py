import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models


# Define the CNN architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # to keep the image at its same size (same padding) padding = (filter_size -1)/2 = 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        # output_size = (input_size - filter_size + 2 * padding)/stride + 1 =
        # (112 - 3 + 2 * 1)/1 + 1 = 112
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        # output_size = 112/2 = 56

        # same padding: padding = (5 - 1)/2 = 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        # output_size = (56 - 5 + 2 * 2)/1 + 1 = 56
        self.bn2 = nn.BatchNorm2d(32)
        # ouput_size (after maxppoling): 56/2 = 28

        # flatten the 32 feature maps
        self.fc1 = nn.Linear(32 * 28 * 28, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.maxpool(self.relu(self.bn2(self.conv2(x))))
        # flatten the 32 features maps from max pool and feed it to fc2
        x = x.view(-1, 32 * 28 * 28)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# trying a bigger architecture
class Big_CNN(nn.Module):
    def __init__(self):
        super(Big_CNN, self).__init__()
        # to keep the image at its same size (same padding) padding = (filter_size -1)/2 = 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        # output_size = (input_size - filter_size + 2 * padding)/stride + 1 =
        # (112 - 3 + 2 * 1)/1 + 1 = 112
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        # output_size = 112/2 = 56

        # same padding: padding = (5 - 1)/2 = 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        # output_size = (56 - 5 + 2 * 2)/1 + 1 = 56
        self.bn2 = nn.BatchNorm2d(32)
        # ouput_size (after maxppoling): 56/2 = 28

        # same padding: padding = (7 - 1)/2 = 3
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3)
        # output-size = (28 - 7 + 2 * 3)/1 + 1 = 28
        self.bn3 = nn.BatchNorm2d(64)
        # ouput_size (after maxppoling): 28/2 = 14

        # flatten the 14 feature maps
        self.fc1 = nn.Linear(64 * 14 * 14, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.maxpool(self.relu(self.bn2(self.conv2(x))))
        x = self.maxpool(self.relu(self.bn3(self.conv3(x))))
        # flatten the 14 features maps from max pool and feed it to fc2
        x = x.view(-1, 64 * 14 * 14)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# trying transfer learning with VGG18
def VGG_model():
    # Load the pre-trained ResNet18 model
    model = models.resnet18(pretrained=True)

    # Freeze the convolutional layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)

    # Move the model to the device
    CUDA = torch.cuda.is_available()
    if CUDA:
        model = model.cuda()

    return model