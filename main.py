import argparse
import yaml
import os
import torch
from torch.utils.data import DataLoader

from utils.helper_path import MNIST_PATH
from utils.helper_functions import create_directory, get_MNIST_dataset

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    create_directory(MNIST_PATH)
    mnist_data = get_MNIST_dataset(MNIST_PATH, download=True, train=True)
    mnist_dataloader = DataLoader(mnist_data, batch_size=64, shuffle=True)

    for batch in mnist_dataloader:
        x, y = batch
        print(torch.min(x[0]), torch.max(x[0]))




