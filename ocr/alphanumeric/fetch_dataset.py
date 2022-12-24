import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

import os


def get_az(path=None):
    if path is None:
        path = os.curdir

    df = pd.read_csv(path + '/handwritten_az.csv')
    df = df.sample(frac=1).reset_index(drop=True)
    data_label = np.array(df.iloc[:, 0])
    data_features = np.array(df.iloc[:, 1:])

    print(f'Feature shape: {data_features.shape}')
    print(f'Label shape: {data_label.shape}')

    features = data_features.reshape((372450, 1, 28, 28))

    # Add 10 because numbers 0-9
    f = lambda x: x + 10
    data_label = f(data_label)

    # # Show a sample image
    # idx = 32358
    # plt.imshow(features[idx], cmap='gray')
    # plt.title(data_label[idx])
    # plt.show()

    return features, data_label


def get_mnist(path=None):
    if path is None:
        path = os.curdir

    csv_train = pd.read_csv(path + '/train.csv')
    csv_val = pd.read_csv(path + '/test.csv')

    train_label = csv_train['label']
    train_data = csv_train.drop('label', axis=1)

    train_label = train_label.to_numpy()
    train_data = train_data.to_numpy()

    # Reshape
    train_data = train_data.reshape(42000, 1, 28, 28)

    print('Fetched data')

    return train_data, train_label


def main():
    print('This script should not be run directly')
    get_az()
    #
    # x_train, _ = get_mnist()
    #
    # plt.imshow(x_train[33452], cmap='gray')
    # plt.show()


if __name__ == '__main__':
    main()
