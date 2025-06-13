from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def visualize_discriminants(eig_vals: Sequence[int | float]) -> None:
    """
    Метод визуализирует коэффициенты дискриминируемости (разделимости).
    """
    total = sum(eig_vals)
    # Получили доли коэффициентов дискриминируемости для каждой дискриминанты.
    ratios = [round(d / total, 4) for d in sorted(eig_vals, reverse=True)]
    # Вычисляем накопительную дискриминируемость: [1, 3, 5, 7] -> [1, 4, 9, 16]
    cum_vals = np.cumsum(ratios)
    plt.bar(range(1, len(eig_vals) + 1), ratios, align='center', label='Индивидуальная дискриминируемость')
    plt.step(range(1, len(eig_vals) + 1), cum_vals, where='mid', label='Накоnительная дискриминируемость')
    plt.ylabel('Коэффициент дискриминируемости')
    plt.xlabel('Линейные дискриминанты')
    plt.legend()
    plt.tight_layout()
    plt.show()


def visualize_new_lda_frame(X: np.array, Y: np.array) -> None:
    """
    Метод для визуализации разделения классов в новой системе координат, где ось x - это LD1, а ось Y - это LD2.
    """
    colors = ['r', 'b', 'g']
    markers = ['o', 's', '^']

    for tag, marker, color in zip(np.unique(Y), markers, colors):
        plt.scatter(
            x=X[Y == tag, 0],  # LD1
            y=X[Y == tag, 1],  # LD2
            c=color,
            marker=marker,
            label=f'Class {tag}',
        )
    plt.ylabel('LD2')
    plt.xlabel('LD1')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.show()
