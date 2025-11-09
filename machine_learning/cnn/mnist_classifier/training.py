from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from machine_learning.cnn.mnist_classifier.models import MnistCnnClassifier
from machine_learning.cnn.mnist_classifier.preprocessing import load_data


def train(learning_rate: float = 0.001, epochs: int = 10, batch_size: int = 64):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Получаем данные.
    path = Path().cwd() / 'preprocessing'
    train_dataset, valid_dataset, test_dataset = load_data(path)
    train_dataloader, valid_dataloader = (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
        DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    )
    # Объявляем модель.
    model = MnistCnnClassifier().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_losses = [0] * epochs
    train_accuracies = [0] * epochs
    valid_losses = [0] * epochs
    valid_accuracies = [0] * epochs
    # Обучение.
    for epoch in range(epochs):
        model.train()
        for x_mini, y_mini in train_dataloader:
            x_mini, y_mini = x_mini.to(device), y_mini.to(device)
            y_pred = model(x_mini)  # логиты
            loss = loss_fn(y_pred, y_mini)  # softmax + NLLLoss
            is_correct = (torch.argmax(y_pred, dim=1) == y_mini).float()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_losses[epoch] += loss.item() * y_mini.size(0)
            train_accuracies[epoch] += is_correct.sum().item()
        train_losses[epoch] /= len(train_dataloader.dataset)
        train_accuracies[epoch] /= len(train_dataloader.dataset)

        # Проверка модели на валидационной выборке.
        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in valid_dataloader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                y_pred = model(x_batch)
                loss = loss_fn(y_pred, y_batch)
                valid_losses[epoch] += loss.item() * y_batch.size(0)
                is_correct = (torch.argmax(y_pred, dim=1) == y_batch).float()
                valid_accuracies[epoch] += is_correct.sum().item()
        valid_losses[epoch] /= len(valid_dataloader.dataset)
        valid_accuracies[epoch] /= len(valid_dataloader.dataset)

    # Визуализация.
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(range(epochs), train_losses, linestyle='-', marker='o', label='Потери при обучении.')
    ax.plot(range(epochs), valid_losses, linestyle='--', marker='<', label='Потери при валидации.')
    ax.legend(fontsize=15)
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(range(epochs), train_accuracies, linestyle='-', marker='o', label='Точность при обучении.')
    ax.plot(range(epochs), valid_accuracies, linestyle='--', marker='<', label='Точность при валидации.')
    ax.legend(fontsize=15)
    ax.set_xlabel('Эпoxa', size=15)
    ax.set_ylabel('Точность', size=15)
    plt.show()

    # Проверка на отложенной выборке.
    # Получаем тензор [batch_size, 1, 28, 28] через добавление новой оси в 1 положение. Работаем с объектом MNIST.
    x_test = (test_dataset.data.unsqueeze(1) / 255.0).to(device)
    y_test = test_dataset.targets.to(device)
    y_pred = model(x_test)
    is_correct = (torch.argmax(y_pred, dim=1) == y_test).float()
    print(f'Точность при тестировании: {is_correct.sum().item() / len(test_dataset):.4f}')
    torch.save(model, 'cnn_mnist_classifier.pt')


if __name__ == '__main__':
    train()
