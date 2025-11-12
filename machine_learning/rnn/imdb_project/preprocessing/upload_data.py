import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from machine_learning.rnn.imdb_project.preprocessing import get_datasets, IMDBCustomCollate
from machine_learning.rnn.preprocessing import SimpleTextVectorizer


def load_data(batch_size: int = 32, device: str = 'cpu'):
    torch.manual_seed(1)
    # Загрузили сырые данные.
    imdb = load_dataset('imdb')
    train_dataset, valid_dataset, test_dataset = get_datasets(imdb)
    # Получения словаря с токенами.
    vectorizer = SimpleTextVectorizer()
    vectorizer.fit(train_dataset)
    # Получаем загрузчики данных.
    collate_fn = IMDBCustomCollate(vectorizer.get_vocabulary(), vectorizer.get_tokenizer, device)
    train_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    valid_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    return train_dataloader, valid_dataloader, test_dataloader
