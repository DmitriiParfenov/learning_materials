import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import learning_curve


def visualize_learning_curve(
        estimator: BaseEstimator,
        x: np.array,
        y: np.array,
        train_sizes: np.linspace = np.linspace(0.1, 1.0, 5),
        cv: int = 5,
        n_jobs: int = 1,
        shuffle: bool = False,
        random_state: int = 1,
) -> None:
    """
    Функция для построения графика "Кривая обучения". Этот график нужен, чтобы понять, есть ли у модели недообучение
    или переобучение, нужны ли дополнительные образцы.

    Args:
        estimator: классификатор
        x: выборка
        y: метки
        train_sizes: например np.linspace(0.1, 1.0, 10) - обучаем модель на 10 подвыборках различного размера:
         10%, 20%, ..., 100% от общего количества обучающих данных.
        cv: Для каждой подвыборки применяется k-fold cross-validation (здесь cv=10), то есть данные разбиваются на 10
         фолдов: модель обучается на 9 фолдах и валидируется на 1, поочерёдно меняя фолды
        n_jobs: количество процессоров
        shuffle: перемешивать ли данных или нет
        random_state: случайное число для генерации случайных чисел
    Returns:
        None
    """
    res_train_sizes, train_scores, test_scores = learning_curve(
        estimator=estimator,
        X=x,
        y=y,
        train_sizes=train_sizes,
        cv=cv,
        n_jobs=n_jobs,
        shuffle=shuffle,
        random_state=random_state
    )

    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(
        res_train_sizes,
        train_mean,
        color='blue',
        marker='o',
        label='Точность при обучении (std)'
    )
    plt.fill_between(
        res_train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.15,
        color='blue'
    )
    plt.plot(
        res_train_sizes,
        test_mean,
        color='green',
        marker='o',
        label='Точность при валидации (std)'
    )
    plt.fill_between(
        res_train_sizes,
        test_mean - test_std,
        test_mean + test_std,
        alpha=0.15,
        color='green'
    )
    plt.grid()
    plt.ylabel('Точность')
    plt.xlabel('Количество обучающих примеров')
    plt.legend(loc='lower right')
    plt.ylim([0.5, 1.03])
    plt.show()
