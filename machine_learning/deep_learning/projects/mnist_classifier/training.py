import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from machine_learning.deep_learning.projects.mnist_classifier.models import MnistClassifier
from machine_learning.deep_learning.projects.mnist_classifier.preprocessing import DataMnistLoader


def train(learning_rate: float = 0.001, epochs: int = 20, batch_size: int = 64):
    # Подготовка данных.
    data_service = DataMnistLoader()
    x_train, x_test, y_train, y_test = data_service.get_prepared_data('mnist_784')
    dataset = TensorDataset(x_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Подготовка модели.
    model = MnistClassifier(x_train.shape[1], (32, 16), 10)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = [0] * epochs
    accuracies = [0] * epochs

    # Обучение.
    for epoch in range(epochs):
        for x_mini, y_mini in dataloader:
            y_pred = model(x_mini)
            loss = loss_fn(y_pred, y_mini)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses[epoch] += loss.item() * y_mini.size(0)
            is_correct = (torch.argmax(y_pred, dim=1) == y_mini).float()
            accuracies[epoch] += is_correct.sum().item()
        losses[epoch] /= len(dataloader.dataset)
        accuracies[epoch] /= len(dataloader.dataset)

    # Тестирование.
    with torch.no_grad():
        y_pred = model(x_test)
        probas = torch.softmax(y_pred, dim=1)
        is_correct = (torch.argmax(probas, dim=1) == y_test).float()
        accuracy = is_correct.sum().item() / y_test.size(0)
        print(f'Точность при тестировании: {accuracy:.4f}')


if __name__ == '__main__':
    train()
