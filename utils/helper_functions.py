import os
import torchvision
from torchvision import transforms


def create_directory(path):
    os.makedirs(path, exist_ok=True)


def get_MNIST_dataset(save_path, download=False, train=True, categorical=False):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255 if categorical else x),  # Scale back to original domain
    ])

    return torchvision.datasets.MNIST(
        root=save_path,
        train=train,
        transform=transform,
        download=download,
    )