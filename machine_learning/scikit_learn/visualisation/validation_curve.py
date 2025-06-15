import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import validation_curve


def visualize_validation_curve(
        estimator: BaseEstimator,
        x: np.array,
        y: np.array,
        param_name: str,
        param_range: np.array,
        cv: int = 5,
        n_jobs: int = 1,
) -> None:
    """
    Функция для построения графика "Кривая валидации". Этот график для подбора гиперпараметров, а также для проверки
    переобучения, так как является функцией точности модели от варьируемого гиперпараметра

    Args:
        estimator: классификатор
        x: выборка
        y: метки
        cv: значение k для k-fold cross-validation
        param_name: название параметра для оптимизации
        param_range: диапазон значений параметра для оптимизации
        n_jobs: количество процессоров
    Returns:
        None
    """
    train_scores, test_scores = validation_curve(
        estimator=estimator,
        X=x,
        y=y,
        param_name=param_name,
        param_range=param_range,
        cv=cv,
        n_jobs=n_jobs,
    )

    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(
        param_range,
        train_mean,
        color='blue',
        marker='o',
        label='Точность при обучении (std)'
    )
    plt.fill_between(
        param_range,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.15,
        color='blue'
    )
    plt.plot(
        param_range,
        test_mean,
        color='green',
        marker='o',
        label='Точность при валидации (std)'
    )
    plt.fill_between(
        param_range,
        test_mean - test_std,
        test_mean + test_std,
        alpha=0.15,
        color='green'
    )
    plt.grid()
    plt.xscale('log')
    plt.ylabel('Точность')
    plt.xlabel('Параметр С')
    plt.legend(loc='lower right')
    plt.ylim([0.8, 1])
    plt.show()
