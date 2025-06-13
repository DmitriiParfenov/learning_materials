from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def visualize_dispersions(eig_vals: Sequence[int | float]) -> None:
    """
    Метод визуализирует доли частных дисперсий и накопительную дисперсию.
    """
    total = sum(eig_vals)
    # Получили долю объясненной дисперсии каждой компоненты
    ratios = [round(d / total, 4) for d in sorted(eig_vals, reverse=True)]
    # Вычисляем накопительную дисперсию: [1, 3, 5, 7] -> [1, 4, 9, 16]
    cum_vals = np.cumsum(ratios)
    plt.bar(range(1, len(eig_vals) + 1), ratios, align='center', label='Отдельные объясненные дисперсии')
    plt.step(range(1, len(eig_vals) + 1), cum_vals, where='mid', label='Совокупная объясненная дисперсия')
    plt.ylabel('Доля объясненной дисперсии')
    plt.xlabel('Индекс главной компоненты')
    plt.legend()
    plt.tight_layout()
    plt.show()


def visualize_new_pca_frame(X: np.array, Y: np.array) -> None:
    """
    Метод для визуализации системы координат, где ось x - это PC1, а ось Y - это PC2.
    РСА - это метод без учителя, который не использует никакой информации о метке класса (здесь просто для наглядности).
    """
    colors = ['r', 'b', 'g']
    markers = ['o', 's', '^']

    for tag, marker, color in zip(np.unique(Y), markers, colors):
        plt.scatter(
            x=X[Y == tag, 0],  # PC1
            y=X[Y == tag, 1],  # PC2
            c=color,
            marker=marker,
            label=f'Class {tag}',
        )
    plt.ylabel('PC2')
    plt.xlabel('PC1')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.show()


def visualize_features_impacts_for_pca(
        df: pd.DataFrame,
        idx_features: list[int],
        eig_vals: np.array,
        eig_vectors: np.array,
        pc_number: int
) -> None:
    """Для визуализации коэффициентов корреляции между признаками и главной компонентной pc_number."""

    loadings = np.sqrt(eig_vals) * eig_vectors
    plt.bar(df.columns[idx_features], height=loadings[:, pc_number])
    plt.title(f'Loadings for РС{pc_number + 1}')
    plt.ylim([-1, 1])
    plt.ylabel(f'Нагрузка на для PC{pc_number + 1}')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
