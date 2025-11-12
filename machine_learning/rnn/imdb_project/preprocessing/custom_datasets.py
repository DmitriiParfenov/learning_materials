from typing import Tuple

from datasets.arrow_dataset import Dataset as HuggingDataset
from datasets.dataset_dict import DatasetDict as HuggingDatasetDict
from torch.utils.data import Dataset, random_split
from torch.utils.data.dataset import Subset


class IMDBDataset(Dataset):
    def __init__(self, data: HuggingDataset) -> None:
        self.samples = [review['text'] for review in data]
        self.labels = [sentiment['label'] for sentiment in data]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[str, int]:
        return self.samples[index], self.labels[index]


def get_datasets(data: HuggingDatasetDict) -> Tuple[Subset, Subset, IMDBDataset]:
    train_dataset = IMDBDataset(data['train'])
    test_dataset = IMDBDataset(data['test'])
    train, valid = random_split(train_dataset, [20000, 5000])
    return train, valid, test_dataset
