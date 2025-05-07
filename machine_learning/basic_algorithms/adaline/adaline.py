import numpy as np


class Adaline:
    def __init__(self, rate: float, epochs: int = 50, random_state: int = 1) -> None:
        """
        Метод инициализирует следующие атрибуты:
            1) rate - скорость обучения (от 0.0 до 1.0) или шаг градиентного спуска.
            2) epoch - количество итераций градиентного спуска.
            3) random_state - случайное число для генерации случайных весов.
        """
        self.rate = rate
        self.epochs = epochs
        self.random_state = random_state
        self._weights = np.array(0)  # веса для каждого признака у образца (в виде np.array).
        self._bias = 0.0  # смещение.
        self.losses = []  # сюда складываем результаты MSE.

    def fit(self, x: np.array, y: np.array):
        """
        Обучение нейронной сети по данным, где:
            1) x - это массив с признаками у образцов: np.array([[2, 3], [1, 1], [4, 5], [3, 2]]).
            2) y - массив с откликами образцов: np.array([1, 0, 1, 0]).
        """
        # Инициализируем веса и смещение.
        rgen = np.random.RandomState(self.random_state)
        self._weights = rgen.normal(loc=0.0, scale=0.01, size=x.shape[1])
        self._bias = 0.0

        # Обучение.
        for _ in range(self.epochs):
            # Суммируем все признаки и веса.
            z = self.net_input(x)  # z = x * wT + bias
            # Получаем ypred через линейную функцию активации.
            linear_activation_func = self.activation(z)  # G(z) = z -> в нашем случае.
            error = (y - linear_activation_func)  # (ytrue - ypred) в виде матрицы для всех образцов.
            # Обновляем веса по формуле: w + rate * 2 (СУММА(ytrue - ypred) * xi) / n
            self._weights += self.rate * 2 * x.T.dot(error) / x.shape[0]
            # Обновляем смещения по формуле: w + rate * 2 (СУММА(ytrue - ypred)) / n
            self._bias += self.rate * 2 * error.mean()
            # Расчет MSE - mean square error.
            loss = (error ** 2).mean()
            self.losses.append(loss)
        return self

    def net_input(self, x: np.array) -> np.array:
        """Определяем фактический выход z для всех образцов, путем произведения матрицы x на вектор w + bias."""
        return np.matmul(x, self._weights) + self._bias

    def activation(self, x: np.array) -> np.array:
        """Линейная функция активации для получения ypred."""
        return x

    def predict(self, x):
        """Возвращаем значение пороговой функции: если z >= 0.5, то функция вернет 1, иначе - 0."""
        return np.where(self.activation(self.net_input(x)) >= 0.5, 1, 0)


def main():
    x = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    y = np.array([0, 1, 1, 1])
    net = Adaline(rate=0.1)
    net.fit(x, y)
    print(net.predict(x))


if __name__ == '__main__':
    main()
