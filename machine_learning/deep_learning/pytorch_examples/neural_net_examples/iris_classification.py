import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


def get_test_data(test_size: float, random_state: int = 1):
    data = load_iris(as_frame=True)
    df = data.frame
    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return train_test_split(x, y, test_size=test_size, random_state=random_state)


class Model(nn.Module):
    """
    Двухслойный перцептрон: 1 скрытый слой и 1 выходной слой. Используется Linear - полносвязный слой, который может
    быть представлен в виде функции: f(w * x + b), где f - это функция активации.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.activation = nn.Sigmoid()
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Метод для прямого распространения сигнала, который возвращает логиты (логарифм шансов) принадлежности к классам.
        """
        x = self.activation(self.input_layer(x))
        x = self.output_layer(x)
        return x


if __name__ == '__main__':
    # Подготовка данных.
    x_train, x_test, y_train, y_test = get_test_data(1 / 3, 1)
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    x_train_norm = (x_train - mean) / std
    x_train = torch.from_numpy(x_train_norm).float()
    y_train = torch.from_numpy(y_train)
    train_dataset = TensorDataset(x_train, y_train)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=2)

    # Определяем модель и ее гиперпараметры.
    learning_rage = 0.001  # скорость обучения.
    epochs = 100  # количество эпох.
    model = Model(x_train_norm.shape[1], 16, 3)  # двухслойный перцептрон.
    loss_fn = nn.CrossEntropyLoss()  # функция ошибки - softmax + neg loss likelihood.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rage)  # для обновления весов.
    losses = [0] * epochs  # храним потери
    accuracies = [0] * epochs

    for epoch in range(epochs):
        for x_batch, y_batch in train_dataloader:
            pred = model(x_batch)  # логиты в виде тензора [кол-во образцов в y_batch * кол-во классов]
            loss = loss_fn(pred, y_batch)  # тензор со средней ошибкой в батче (softmax + neg loss likelihood)
            loss.backward()  # рассчитали градиенты.
            optimizer.step()  # обновили веса и смещение у модели.
            optimizer.zero_grad()  # обнулили градиенты.
            losses[epoch] += loss.item() * y_batch.size(0)  # значение ошибки * кол-во образцов в батче - суммарная ошибка
            is_correct = (torch.argmax(pred, dim=1) == y_batch).float()  # проверяем корректность предсказания
            accuracies[epoch] += is_correct.sum()  # суммируем верные предсказания для расчета accuracy
        losses[epoch] /= len(train_dataloader.dataset)
        accuracies[epoch] /= len(train_dataloader.dataset)

    # Проверка на тестовой выборке.
    x_test_norm = (x_test - mean) / std
    x_test_norm = torch.from_numpy(x_test_norm).float()
    y_test = torch.from_numpy(y_test)
    y_pred = model(x_test_norm)
    accuracy = ((torch.argmax(y_pred, dim=1) == y_test).float()).mean()
    print(f'Точность на тестовой выборке - {accuracy:.2f} %')

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(losses, lw=3)
    ax.set_title('Training loss', size=15)
    ax.set_xlabel('Epoch', size=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(accuracies, lw=3)
    ax.set_title('Training accuracy', size=15)
    ax.set_xlabel('Epoch', size=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.show()
