"""
Алгоритм работы:
1) При создании экземпляра класса Perceptron инициализируем:
    - rate - скорость обучения (от 0.0 до 1.0).
    - epoch - максимальное количество проходов по данным.
    - random_state - случайное число для генерации случайных весов.
2) Создаем X = np.array([[x1, x2, ...], [x1, x2, ...], ...], ) - образцы с признаками.
3) Создаем Y = np.array([y1, y2, ...]) - массив с откликами. X и Y должны быть равны.
4) Обучение модели:
    - instance.fit(X, Y)
    - При вызове метода вначале инициализируются weight и bias случайными малыми значениями, исходя из нормального
распределения, где loc определяет центральное значение выборки, scale - точность знаков у весов и смещения после
запятой и size - количество весов в зависимости от размера X (сколько признаков, столько и весов).
    - Вызов метода predict: определение значения решающей функции G(z), где z - это фактический выход (результат работы
метода net_input - скалярное произведение векторов w и x). Если z >= 0, то G(z) = 1, иначе - 0.
    - Определяем дельту: rate * (Ytrue - Ypred)
    - Переопределяем значение весов: weight += rate * (Ytrue - Ypred) * xi
    - Переопределяем значение смещения: bias += rate * (Ytrue - Ypred)
"""

from pathlib import Path
import pandas as pd
import numpy as np

# Объявляем переменные.
BASE_DIR = Path().resolve().parent.parent
PATH_TO_FILE = BASE_DIR / 'data' / 'iris_data.csv'
# Создаем dataframe из файла: X1, X2, X3, X4, Y.
TEST_DATAFRAME = pd.read_csv(PATH_TO_FILE, delimiter='\t', encoding='UTF-8')


class Perceptron:
    def __init__(self, rate: float, epoch: int = 50, random_state: int = 1) -> None:
        """
        Метод инициализирует следующие атрибуты:
            1) rate - скорость обучения (от 0.0 до 1.0).
            2) epoch - максимальное количество проходов по данным.
            3) random_state - случайное число для генерации случайных весов.
        """
        self.rate = rate
        self.epoch = epoch
        self.random_state = random_state
        self._weight = None  # веса для каждого признака у образца (в виде np.array).
        self._bias = None  # смещение.
        self._errors = []  # сюда складываем результаты обучения нейронной сети.

    def fit(self, x: np.array, y: np.array):
        """
        Обучение нейронной сети по данным, где:
            1) x - это массив с признаками у образцов: np.array([[2, 3], [1, 1], [4, 5], [3, 2]]).
            2) y - массив с откликами образцов: np.array([1, 0, 1, 0]).
        """

        # Устанавливаем случайные веса и смещение рандомными значениями.
        # rgen - генератор случайных чисел.
        rgen = np.random.RandomState(self.random_state)
        # Устанавливаем веса случайными значениями из нормального распределения в виде [w1  w2 ...]
        #   loc - центральное значение нормального распределения (mean = median = mode).
        #   scale - количество знаков после запятой у случайного веса.
        #   size - количество случайных чисел в зависимости от количества признаков у образцов в x.
        self._weight = rgen.normal(loc=0.0, scale=0.01, size=x.shape[1])
        self._bias = 0.0

        # Обучение.
        for _ in range(self.epoch):
            errors = 0
            for xi, target in zip(x, y):  # Для каждого образца с факторами в x и откликом y.
                update = self.rate * (target - self.predict(xi))  # n * (Ytrue - Ypred)
                # Обновление весов: deltaWi = Wi + (n * (Ytrue - Ypred) * Xi). Идет обновление элементов, а не append,
                # так как self._weight - это np.array, а не list.
                self._weight += update * xi
                self._bias += update  # Обновление смещения: deltaBi = Bi + (n * (Ytrue - Ypred))
                errors += int(update != 0.0)
            self._errors.append(errors)
        return self

    def net_input(self, x: np.array) -> int:
        """Определяем фактический выход z для образца, как скалярного произведение векторов w и x."""
        return np.matmul(x, self._weight) + self._bias

    def predict(self, x: np.array, show_equation: bool = False) -> np.array:
        """Возвращаем значение решающей функции: если z >= 0, то функция вернет 1, иначе - 0."""
        if show_equation:
            print(self._weight, '\t\t', self._bias)
        return np.where(self.net_input(x) >= 0.0, 1, 0)


def main():
    a = Perceptron(0.1, 20, 1)
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 1])
    a.fit(x, y)
    print(a.predict(np.array([[0, 0], [0, 1]])))


if __name__ == '__main__':
    main()
