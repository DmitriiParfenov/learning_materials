from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def visualize_residual_plot(
        y_train_pred: np.array,
        y_test_pred: np.array,
        y_train_true: np.array,
        y_test_true: np.array,
        metrics: dict[str, dict[str, int | float]]
) -> tuple[Figure, Any]:
    figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3), sharey=True)
    figure.suptitle('Остатки против Предсказанные значения')
    # Плот для обучающей выборки.
    train_metrics = [f'{name}={value:.2f}' for (name, value) in metrics.get('train', dict()).items()]
    ax1.scatter(
        x=y_train_pred,
        y=y_train_pred - y_train_true,
        c='steelblue',
        marker='s',
        edgecolors='white',
        label='Обучающие данные',
    )
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax1.legend()
    ax1.set_xlabel(f'Предсказанные значения\n{"; ".join(train_metrics)}')
    ax1.set_ylabel('Остатки')

    # Плот для тестовой выборки.
    test_metrics = [f'{name}={value:.2f}' for (name, value) in metrics.get('test', dict()).items()]
    ax2.scatter(
        x=y_test_pred,
        y=y_test_pred - y_test_true,
        c='limegreen',
        marker='o',
        edgecolors='white',
        label='Тестовые данные'
    )
    ax2.legend()
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax2.set_xlabel(f'Предсказанные значения\n{"; ".join(test_metrics)}')
    ax2.set_ylabel('Остатки')

    return figure, (ax1, ax2)
