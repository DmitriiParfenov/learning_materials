import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from machine_learning.deep_learning.projects.car_fuel_prediction.models import RegressionModel
from machine_learning.deep_learning.projects.car_fuel_prediction.preprocessing import DataCarLoader


def train(learning_rate: float = 0.001, epochs: int = 200, batch_size: int = 8):
    # Получаем данные для нейронной сети.
    torch.manual_seed(1)
    data_service = DataCarLoader()
    x_train, x_test, y_train, y_test = data_service.get_prepared_data(
        name='autoMpg',
        numeric_columns=['displacement', 'horsepower', 'weight', 'acceleration'],
        categorical_columns=['cylinders', 'origin'],
    )
    dataset = TensorDataset(x_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Объявляем модель, функцию ошибки и оптимизатор весов.
    model = RegressionModel(n_features=x_train.shape[1], n_hidden=(8, 4))
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    losses = [0] * epochs

    # Обучение.
    torch.manual_seed(1)
    for epoch in range(epochs):
        for x_mini, y_mini in dataloader:
            y_pred = model(x_mini)[:, 0]
            loss = loss_fn(y_pred, y_mini)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses[epoch] += loss.item() * y_mini.size(0)
        losses[epoch] /= len(dataloader.dataset)

    # Тестирование.
    with torch.no_grad():
        y_pred = model(x_test)[:, 0]
        loss = loss_fn(y_pred, y_test)
        print(f'MSE при тестировании: {loss.item():.4f}')
        print(f'МAE при тестировании: {nn.L1Loss()(y_pred, y_test).item():.4f}')


if __name__ == '__main__':
    train()
