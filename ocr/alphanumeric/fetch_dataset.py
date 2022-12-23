import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

import os
import glob


def fit_to_resnet(x, size=(64, 64)):
    img = Image.fromarray(x, )
    img = img.resize(size)
    img = img.convert('RGB')
    img = np.array(img)
    img = img / 255.

    # plt.imshow(img)
    # plt.show()

    return img


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
    
    return train_data, train_label


def main():
    print('This script should not be run directly')
    
    x_train, _ = get_mnist()
    
    plt.imshow(x_train[33452], cmap='gray')
    plt.show()


if __name__ == '__main__':
    main()