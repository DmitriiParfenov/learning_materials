from pathlib import Path
from typing import Tuple

from torch.utils.data import random_split
from torch.utils.data.dataset import Subset
from torchvision import transforms
from torchvision.datasets import MNIST


def load_data(path: str | Path = './', transform=None, train_size: float = 0.85) -> Tuple[Subset, Subset, MNIST]:
    assert 0 < train_size < 1, "train_size must be in the range (0, 1)."
    if not transform:
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor()
        ])
    mnist_dataset = MNIST(path, download=True, train=True, transform=transform)
    train_length = int(train_size * len(mnist_dataset))
    valid_length = len(mnist_dataset) - train_length
    train_dataset, valid_dataset = random_split(dataset=mnist_dataset, lengths=[train_length, valid_length])
    test_dataset = MNIST(path, download=True, train=False, transform=transform)
    return train_dataset, valid_dataset, test_dataset
