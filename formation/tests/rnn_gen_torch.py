import os
import yaml
import argparse

import numpy as np
import pandas as pd

import torch
import torch.nn
from torch.utils.data import TensorDataset, DataLoader

from utils import Model, Dataset


parser = argparse.ArgumentParser()


def main():
    ds = Dataset('alllines.txt')

    print(f'Length of dataset: {len(ds)}')

    print(f'Getitem: {ds.__getitem__(0)}')


if __name__ == '__main__':
    main()
