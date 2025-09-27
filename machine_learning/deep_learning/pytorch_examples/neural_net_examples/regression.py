import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Формируем x и y.
x = np.arange(10, dtype='float32').reshape((10, 1))
y = np.array([1.0, 1.3, 3.1, 2.0, 5.0, 6.3, 6.7, 7.4, 8.0, 9.0])
# Стандартизируем x и преобразуем x и y в тензоры
x_std = torch.from_numpy((x - x.mean()) / x.std())
y_std = torch.from_numpy(y).float()
# Объединяем x и y.
dataset = TensorDataset(x_std, y_std)
# Создаем загрузчик данных, который будет формировать пакеты размером 1.
dataloader = DataLoader(dataset, shuffle=True, batch_size=1)
loss_fn = nn.MSELoss(reduction='mean')  # объявляем функцию потерь.
model = nn.Linear(in_features=1, out_features=1)  # объявляем модель для линейной регрессии.
# Обновление весов и смещения через SGD.
updater = torch.optim.SGD(
    model.parameters(),  # содержит тензоры с весами и смещением для обновления
    lr=0.001
)
for epoch in range(200):
    for x_batch, y_batch in dataloader:
        # Предсказываем значение отклика.
        y_pred = model(x_batch)[:, 0]
        # Вычисляем потери.
        loss = loss_fn(input=y_pred, target=y_batch)
        # Вычисляем градиенты.
        # В model.weight.grad и model.bias.grad сохраняются вычисленные градиенты.
        # Градиенты у модели накапливаются!
        loss.backward()
        # Обновляем параметры, используя градиенты.
        # В model.parameters() есть веса и смещения и они обновляются через градиенты в .grad.
        updater.step()
        # Обнуляем градиенты.
        # Необходимо обнулять градиенты, так как они накапливаются.
        updater.zero_grad()
    if epoch % 10 == 0:
        print(f' Эпохи {epoch} Потери {loss.item():.4f} ')

print('Окончательные параметры: ', model.weight.item(), model.bias.item())
