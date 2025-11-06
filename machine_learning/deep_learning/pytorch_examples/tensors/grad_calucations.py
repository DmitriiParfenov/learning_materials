import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# Инициализируем вес и смещение.
w_1 = torch.empty((1, 1), requires_grad=True)
b = torch.empty((1, 1), requires_grad=True)
nn.init.xavier_normal_(w_1)
nn.init.xavier_normal_(b)
# Входные данные.
x = torch.tensor([1.0, 2.0, 3.0, 5.0, 7.0, 9.0, 11.0])
y = torch.tensor([35.6, 37.8, 39.8, 40.2, 42.2, 45.5, 50.3])
# Гиперпараметры.
epochs = 1000
learning_rate = 0.001
# Обучение.
losses = []
for epoch in range(epochs):
    # Регрессия - z = x * w + b.
    z = torch.mul(x, w_1) + b
    # Определяем ошибку - SUM(y_true - y_pred)^2
    loss_fn = (y - z).pow(2).sum()
    # Для каждого листа в графе вычисляем градиенты.
    loss_fn.backward()
    with torch.no_grad():  # отключили отслеживание градиентов Pytorch
        w_1 -= learning_rate * w_1.grad
        b -= learning_rate * b.grad
    # Обнуляем градиенты в тензорах w_1 и b
    w_1.grad = None
    b.grad = None
    losses.append(loss_fn.item())

plt.plot(losses)
plt.title('Ошибка при обучении')
plt.xlabel('Эпохи')
plt.ylabel('Ошибка')
plt.show()
