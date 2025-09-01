import numpy as np


def get_dummies(y: np.array, labels: int) -> np.array:
    """
    Метод для получения dummy-переменных через one-hot encoding.
    Например,
        y = [1 2 5] и labels = 10 =>
            [
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            ]
    Args:
        y: массив с метками
        labels: количество ожидаемых уникальных меток
    Returns:
        массив с dummy-переменными
    """
    dummies = np.zeros((y.shape[0], labels))  # строки * столбцы
    for idx, num in enumerate(y):
        dummies[idx, num] = 1
    return dummies
