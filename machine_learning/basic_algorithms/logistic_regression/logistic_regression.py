import numpy as np


class LogisticRegression:

    def __init__(self, eta: float = 0.1, n_iter: int = 50, random_state: int = 1) -> None:
        """
        Метод инициализирует следующие атрибуты:
            1) eta - скорость обучения (от 0.0 до 1.0) или шаг градиентного спуска.
            2) n_iter - количество итераций градиентного спуска.
            3) random_state - случайное число для генерации случайных весов.
        """
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

        self._weights = np.array(0)
        self._bias = 0.0
        self.losses = []

    def fit(self, x: np.array, y: np.array):
        """
        Обучение нейронной сети по данным, где:
            1) x - это массив с признаками у образцов: np.array([[2, 3], [1, 1], [4, 5], [3, 2]]).
            2) y - массив с откликами образцов: np.array([1, 0, 1, 0]).
        """
        rgen = np.random.RandomState(seed=self.random_state)
        self._weights = rgen.normal(loc=0.0, scale=0.01, size=x.shape[1])

        for _ in range(self.n_iter):
            z = self.net_input(x)  # Фактический выход.
            y_pred = self.activation(z)  # Вероятность предсказания через сигмоиду.
            errors = (y - y_pred)  # Ошибка предсказания.
            # Обновляем веса по формуле: w + rate * (СУММА(y - ypred) * xi) / n
            self._weights += self.eta * x.T.dot(errors) / x.shape[0]
            # Обновляем смещение по формуле: b + rate * (СУММА(y - ypred)) / n
            self._bias += self.eta * errors.mean()
            # Считаем потери для всех образцов.
            loss = (-y.dot(np.log(y_pred)) - ((1 - y).dot(np.log(1 - y_pred)))) / x.shape[0]
            self.losses.append(loss)

        return self

    def net_input(self, x: np.array):
        """Определяем фактический выход z для всех образцов, путем произведения матрицы x на вектор w + bias."""
        return np.matmul(x, self._weights) + self._bias

    def activation(self, z):
        """Определение вероятности через функцию активации (сигмоида)."""
        sigmoid = 1 / (1 + np.exp(-np.clip(z, -250, 250)))
        return sigmoid

    def predict(self, x: np.array):
        """Возвращаем значение пороговой функции: если z >= 0.5, то функция вернет 1, иначе - 0."""
        return np.where(self.activation(self.net_input(x)) >= 0.5, 1, 0)


if __name__ == '__main__':
    from sklearn.datasets import load_iris

    iris = load_iris()
    x = iris.data[:100, [1, 2]]
    y = iris.target[:100]
    log_reg = LogisticRegression(0.001, 100, 1)
    log_reg.fit(x, y)
    x_test = np.array([2.5, 1.5])
    print(log_reg.predict(x_test))


