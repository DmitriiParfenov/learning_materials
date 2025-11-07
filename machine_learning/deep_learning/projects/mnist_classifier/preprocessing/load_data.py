from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from torchvision import transforms


class DataMnistLoader:
    @staticmethod
    def load_data(name: str) -> Tuple[np.ndarray[int], np.ndarray[int]]:
        x, y = fetch_openml(name=name, return_X_y=True, version=1)
        return x.to_numpy(dtype=int), y.to_numpy(dtype=int)

    @staticmethod
    def normalize_data(x: np.ndarray[int] | torch.Tensor) -> np.ndarray[int] | torch.Tensor:
        return (x / x.max() - 0.5) * 2

    def convert_image_to_tensor(self, path: str, transform: transforms.Compose | None = None) -> torch.Tensor:
        path_to_image = Path(path)
        if not path_to_image.exists() or not path_to_image.is_file():
            raise FileNotFoundError('File not found.')
        if not transform:
            transform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.view(-1))
            ])
        img = Image.open(path_to_image).convert('L')  # Перевели в черно-белое изображение.
        img = ImageOps.invert(img)  # Инвертировали изображение: фон - черный, цифра - белая.
        img_tensor = transform(img)
        return self.normalize_data(img_tensor)

    def get_prepared_data(self, name: str, test_size: float = 0.2, shuffle: bool = True, random_state: int = 1):
        x, y = self.load_data(name)
        x_norm = self.normalize_data(x)
        x_train, x_test, y_train, y_test = train_test_split(
            x_norm, y,
            test_size=test_size,
            shuffle=shuffle,
            random_state=random_state
        )
        x_train, x_test, y_train, y_test = (
            torch.from_numpy(x_train).float(),
            torch.from_numpy(x_test).float(),
            torch.from_numpy(y_train).long(),  # переводим в int
            torch.from_numpy(y_test).long()  # переводим в int
        )
        return x_train, x_test, y_train, y_test
