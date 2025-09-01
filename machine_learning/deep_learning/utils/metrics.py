import numpy as np
from machine_learning.deep_learning.utils import get_dummies


def get_mse_loss(y_true: np.ndarray, probas: np.ndarray, labels: int = 10) -> np.floating:
    """
    Возвращает ошибку предсказания по метрике MSE.

    Здесь y_true - это истинные метки классов (например, цифры от 0 до 9). Их необходимо кодировать, например:
        y = [1, 2, 3] в y_dummies = [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]] при labels=4
    Это необходимо, потому что probas - это массив с вероятностями принадлежности экземпляра к числам от 0 до 9
    (например).

    Args:
        y_true: массив с истинными метками
        probas: вероятности принадлежности к меткам, спрогнозированные моделью
        labels: количество классов
    Returns:
        массив с ошибками
    """
    encoded_targets = get_dummies(y_true, labels)
    return np.mean((encoded_targets - probas) ** 2)


def get_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> np.floating:
    """
    Метод для расчета accuracy (точность прогнозирования).
    Выражение np.mean(y_true == y_pred) - это mean([True, False, False]) = 0.33
    """
    return np.mean(y_true == y_pred)
