import os
import yaml
import argparse

import numpy as np
import pandas as pd

import torch
import torch.nn
from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm

from utils import Model, Dataset


TRAIN_SPLIT = 0.8

parser = argparse.ArgumentParser()

parser.add_argument('ds_path', type=str)
parser.add_argument('device', type=int)
parser.add_argument('seed', type=int)

parser.add_argument('batch_size', type=int)
parser.add_argument('epochs', type=int)


def train(device, model, batch_size, train_loader, test_loader, epochs, loss_fn, optimizer):
    # BEGIN EPOCH
    for epoch in range(epochs):
        print(f'BEGIN EPOCH {epoch + 1} --------------------------------------------------------\n')
        losses = []
        # BEGIN TRAIN
        hidden = model.init_state(device, batch_size)
        for i, (x, y) in enumerate(tqdm(train_loader)):
            x = x.to(device).long()
            y = y.to(device).long()  # Already a torch.LongTensor()

            # print(f'x: {x.size()}')
            # print(f'y: {y.size()}')

            y_pred, _ = model(x, hidden)
            # y_pred = model(x)
            y_pred = torch.transpose(y_pred, 1, 2)
            # y_pred = torch.transpose(y_pred, 1, 1)

            # print(y.size())
            # print(y_pred.size())
            loss = loss_fn(y_pred, y)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Train Loss: {torch.tensor(losses).mean():.3f}')
        # END TRAIN

        losses = []

        # BEGIN VALIDATION
        hidden = model.init_state(device, batch_size)
        print('Validation...')
        for i, (x, y) in enumerate(tqdm(test_loader)):
            x = x.to(device).long()
            y = y.to(device).long()

            y_pred, _ = model(x, hidden)
            y_pred = torch.transpose(y_pred, 1, 2)
            # y_pred = model(x)
            losses.append(loss_fn(y_pred, y).item())

        print(f'Validation Loss: {torch.tensor(losses).mean():.3f}')

        torch.save(model, f'models/end_e_{epoch}.pt')


def main():
    args = parser.parse_args()
    ds_path = args.ds_path
    cuda_idx = args.device

    seed = args.seed
    batch_size = args.batch_size
    epochs = args.epochs

    ds = Dataset(ds_path)
    train_size = round(len(ds) * TRAIN_SPLIT)
    val_size = len(ds) - train_size
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size], generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_ds, batch_size=batch_size, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, drop_last=True)

    print(f'Length of dataset: {len(ds)}')

    # Define our device
    device = f'cuda:{cuda_idx}' if torch.cuda.is_available() else 'cpu'
    print(f'Using device {device}')
    if device == 'cpu':
        print(f'WARNING: Ignoring cuda_idx ({cuda_idx}) because CUDA is not available.')

    # Create our model
    model = Model(ds).to(device)

    # Loss and Optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Train
    train(device, model, batch_size, train_loader, val_loader, epochs, loss_fn, optimizer)

    # Try out the model


if __name__ == '__main__':
    main()
