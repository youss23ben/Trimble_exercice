import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from torch.utils.data import random_split
import torchvision

import pandas as pd

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.utils import resample

from imblearn.over_sampling import RandomOverSampler

from augmentations import transform, transform_aug
from utils import to_dataframe
from datasets import OversampledDataset
from models import CNN, Big_CNN, VGG_model


def oversample_data():
    df = to_dataframe()
    # data is embalanced, oversampling the field class
    # Oversample the minority class (fields) in the training set
    fields_data = df[df['label'] == 0]
    roads_data = df[df['label'] == 1]
    fields_data_oversampled = resample(fields_data, replace=True, n_samples=len(roads_data))

    # Combine the oversampled fields data with the original roads data
    data_oversampled = pd.concat([roads_data, fields_data_oversampled])

    return data_oversampled


def load_data(batch_size, augment):
    df_oversampled = oversample_data()
    if augment:
        dataset = OversampledDataset(df_oversampled, transform=transform_aug)
    else:
        dataset = OversampledDataset(df_oversampled, transform=transform)

    # Split the data into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, val_dataloader


# training
def train_model(model, batch_size, lr, augment):
    # Set the device
    CUDA = torch.cuda.is_available()
    if CUDA:
        model = model.cuda()

    # Define the optimizer and the loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_loader = load_data(batch_size, augment)[0]

    # Training the model
    correct = 0
    iterations = 0
    iter_loss = 0.0

    # set the model in training mode
    model.train()

    for batch_index, (inputs, labels) in enumerate(train_loader):

        if CUDA:
            inputs = inputs.cuda()
            labels = labels.cuda()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        iter_loss += loss.item()

        # clear the gradient
        optimizer.zero_grad()
        # backpropagation
        loss.backward()
        # update the weights
        optimizer.step()

    return iter_loss


def validate_model(model, batch_size, augment):
    model.eval()

    CUDA = torch.cuda.is_available()

    y_true, y_pred = [], []

    val_loader = load_data(batch_size, augment)[1]
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():

        for data, target in val_loader:

            if CUDA:
                data = data.cuda()
                target = target.cuda()

            output = model(data)
            loss = criterion(output, target)
            iter_loss = loss.item()

            _, pred = torch.max(output, dim=1)
            y_true.extend(target.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    return loss, precision, recall, f1


def training_loop(model, lr, epochs, batch_size, augment, model_name):
    # Define your early stopping criteria
    patience = 5  # number of epochs to wait before stopping
    best_loss = float('inf')
    counter = 0  # counter to keep track of number of epochs since last improvement

    for epoch in range(1, epochs + 1):
        train_loss = train_model(model=model, lr=lr, batch_size=batch_size, augment=augment)

        val_loss, precision, recall, f1 = validate_model(model=model, batch_size=batch_size, augment=augment)

        # update the best validation loss and save the model if it's improved
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), f'saved_models/best_model_{model_name}_augment_{augment}.pt')
            counter = 0
        else:
            counter += 1

        # stop training if the validation loss hasn't improved for 'patience' epochs
        if counter >= patience:
            print(f"Early stopping after {epoch} epochs")
            break

        print(
            f'Epoch {epoch}: train_loss={train_loss}, val_loss={val_loss}, precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}')


if __name__ == "__main__":
    bigger_cnn = Big_CNN()
    training_loop(model=bigger_cnn, batch_size=32, lr=0.001, epochs=23, augment=True, model_name="big_cnn")