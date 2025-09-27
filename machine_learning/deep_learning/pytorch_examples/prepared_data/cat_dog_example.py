import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from machine_learning.deep_learning.pytorch_examples.prepared_data.utils import get_file_list, decode_image_to_tensor


class CatDogImagesDataset(Dataset):
    def __init__(self, files: list[str], labels: list[int], transform: Compose) -> None:
        assert len(files) == len(labels), 'The lengths of files and labels must be equal.'
        self.files = files
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        img = Image.open(self.files[idx])
        return self.transform(img), self.labels[idx]


if __name__ == '__main__':
    files_list = get_file_list(r'C:\Users\GTA\Desktop\sklearn\cat_dogs', '*.jpg')
    labels = [1 if 'dog' in os.path.basename(path) else 0 for path in files_list]
    transform = decode_image_to_tensor(80, 120)

    data = CatDogImagesDataset(files_list, labels, transform)

    for i in data:
        print(i)

