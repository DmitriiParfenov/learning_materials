import matplotlib.pyplot as plt
import torch.cuda
import torch.nn as nn

from machine_learning.rnn.imdb_project.models import RNNModel
from machine_learning.rnn.imdb_project.preprocessing.upload_data import load_data


def train(batch_size: int = 512, epochs: int = 10, learning_rate: float = 0.002):
    torch.manual_seed(1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Получаем данные для обучения.
    train_dataloader, valid_dataloader, test_dataloader, vocabulary = load_data(batch_size, device)

    # Получаем модель.
    model = RNNModel(len(vocabulary), 20, 64, 64).to(device)
    loss_fn = nn.BCELoss()  # ожидает на вход вероятности
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_losses = [0] * epochs
    train_accuracies = [0] * epochs
    valid_losses = [0] * epochs
    valid_accuracies = [0] * epochs

    # Обучение.
    for epoch in range(epochs):
        model.train()
        for x_batch, y_batch, length_size in train_dataloader:
            y_pred = model(x_batch, length_size)[:, 0]
            loss = loss_fn(y_pred, y_batch.float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_losses[epoch] += loss.item() * y_batch.size(0)
            train_accuracies[epoch] += ((y_pred >= 0.5).float() == y_batch).float().sum().item()
        train_losses[epoch] /= len(train_dataloader.dataset)
        train_accuracies[epoch] /= len(train_dataloader.dataset)

        model.eval()
        with torch.no_grad():
            for x_batch, y_batch, length_size in valid_dataloader:
                y_pred = model(x_batch, length_size)[:, 0]
                loss = loss_fn(y_pred, y_batch.float())
                valid_losses[epoch] += loss.item() * y_batch.size(0)
                valid_accuracies[epoch] += ((y_pred >= 0.5).float() == y_batch).float().sum().item()
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

    torch.save(model, 'rnn_classifier.pt')


if __name__ == '__main__':
    train()
