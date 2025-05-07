import numpy as np


class AdalineSGD:

    def __init__(self,
                 rate: float = 0.01,
                 epoch: int = 10,
                 shuffle: bool = True,
                 random_state: int | float | None = None) -> None:
        """
        Метод инициализирует следующие атрибуты:
            1) rate - скорость обучения (от 0.0 до 1.0) или шаг градиентного спуска.
            2) epoch - количество итераций градиентного спуска.
            4) shuffle - если True, то данные будут перемешиваться в каждой эпохе, чтобы избежать циклических
            зависимостей.
            3) random_state - случайное число для генерации случайных весов. Если None, то задается случайное состояние
            генератора случайных чисел.
        """
        self.rate = rate
        self.epoch = epoch
        self.shuffle = shuffle
        self.random_state = random_state

        self._weights_init = False  # флаг, указывающий инициализированы ли веса или нет.
        self._losses = []  # для хранения значения среднеквадратичной функции потерь каждой эпохи.
        self._rgen = np.random.RandomState(self.random_state)  # для генерации случайных чисел.

    def fit(self, x, y):
        """
        Обучение нейронной сети по данным, где:
            1) x - это массив с признаками у образцов: np.array([[2, 3], [1, 1], [4, 5], [3, 2]]).
            2) y - массив с откликами образцов: np.array([1, 0, 1, 0]).
        """
        # Инициализируем веса и смещение вокруг количества признаков у образцов.
        self._initialize_weights(x.shape[1])

        # Обучение. В рамках одной эпохи определяем веса для каждого образца отдельно. В self._losses записываем среднюю
        # ошибку функции потерь среди всех образцов в рамках 1 эпохи.
        for _ in range(self.epoch):
            if self.shuffle:  # Перемешиваем данные признаков (x) и откликов (y).
                x, y = self._shuffle(x, y)
            losses = []  # храним ошибку MSE ДЛЯ КАЖДОГО ОБРАЗЦА ОТДЕЛЬНО.

            # Определяем веса для каждого образца отдельно.
            for xi, target in zip(x, y):
                losses.append(self._update_weights(xi, target))

            # Определяем среднее значение ошибок MSE в рамках одной эпохи и кладем ее в self._losses.
            avg_loss = np.mean(losses)
            self._losses.append(avg_loss)
        # Возвращаем self, чтобы у экземпляра класса могли вызывать методы в виде цепочки.
        return self

    def partial_fit(self, x: np.array, y: np.array) -> None:
        """
        Метод обновляет веса без необходимости повторной инициализации весов. Необходим для реализации обучения
        модели <на лету>.
        """
        if not self._weights_init:
            self._initialize_weights(x.shape[1])

        # Обновляем веса. Метод ravel преобразует в одномерный массив: [[1, 2], [5, 7]] -> [1, 2, 5, 7]
        if y.ravel().shape[0] > 1:
            for xi, target in zip(x, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(x, y)

    def _shuffle(self, x: np.array, y: np.array) -> tuple[np.array, np.array]:
        """Метод перемешивает данные признаков и откликов в каждой эпохе, чтобы избежать циклических зависимостей."""

        # rgen.permutation(n) -> перемешивает индексы от 0 до n-1 случайным образом. Порядок для x и y будет одинаковым.
        # rgen,permutation(3) -> [1 2 0].
        # Xдо = [[1, 2], [4, 3], [5, 2]]; Yдо = [2, 5, 6]
        # Xпосле = [[4, 3], [5, 2], [1, 2]]; Yпосле = [5, 6, 2]
        r = self._rgen.permutation(len(y))
        return x[r], y[r]

    def _initialize_weights(self, m: int) -> None:
        """Метод инициализирует веса и смещение значениями из нормального распределения вокруг m."""
        self._weights = self._rgen.normal(loc=0.0, scale=0.01, size=m)
        self._bias = 0.0
        self._weights_init = True

    def _update_weights(self, xi, target):
        """
        Метод обновляет веса в соответствии с алгоритмом Adaline:
            - расчет значения z для текущего образца (self.net_input(xi))
            - получение ypred через линейную функцию активации (self.activation(self.net_input(xi))).
            - обновление весов для текущего образца.
            - обновление смещения для текущего образца.
            - определение MSE.
        """

        output = self.activation(self.net_input(xi))
        error = (target - output)
        self._weights += self.rate * 2.0 * xi * error
        self._bias += self.rate * 2.0 * error
        loss = error ** 2
        return loss

    def net_input(self, x):
        """Определяем фактический выход z для всех образцов, путем произведения матрицы x на вектор w + bias."""
        return np.matmul(x, self._weights) + self._bias

    def activation(self, x):
        """Линейная функция активации для получения ypred."""
        return x

    def predict(self, x):
        """Возвращаем значение пороговой функции: если z >= 0.5, то функция вернет 1, иначе - 0."""
        return np.where(self.activation(self.net_input(x)) >= 0.5, 1, 0)


def main():
    sample = np.array([[1, 0, 0], [1, 1, 0], [1, 0, 1], [0, 1, 0]])
    print(f'Правильный вывод - [0 1 1 0] для образца: \n{sample}')

    print(f'____ПЕРВАЯ ИТЕРАЦИЯ_____')
    # 1 итерация обучения модели
    x = np.array([[1, 0, 1], [0, 0, 0]])
    y = np.array([1, 0])
    net = AdalineSGD(rate=0.15, epoch=50)
    net.fit(x, y)
    # Выводим значение весов после 1 итерации и результаты предсказания.
    print(f'Веса после первой итерации:\t{net._weights}')
    print(f'Результаты предсказания после первой итерации:\t{net.predict(sample)}\n')
    print(f'____ВТОРАЯ ИТЕРАЦИЯ_____')
    # 2 итерация обучения модели
    x1 = np.array([[1, 0, 0],[1, 1, 0],[1, 1, 1],[0, 1, 1],[0, 0, 1],[0, 1, 0]])
    y2 = np.array([0, 1, 1, 1, 0, 0])
    net.partial_fit(x1, y2)
    print(f'Веса после первой итерации:\t{net._weights}')
    print(f'Результаты предсказания после первой итерации:\t{net.predict(sample)}')


if __name__ == '__main__':
    main()
