import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mlxtend.plotting import (
    scatterplotmatrix,
    heatmap
)


def get_feature_info(df: pd.DataFrame, feature: str) -> None:
    """
    Метод показывает распределение указанного признака, а также предоставляет информацию об описательных статистиках
    признака.
    Args:
        df: dataframe с данными
        feature: название признака
    Returns:
        None
    """
    if feature not in df.columns:
        raise ValueError(f'Feature {feature} not found')

    # Фигура будет состоять из двух фреймов: график и описательные статистики.
    figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={'width_ratios': [3, 1]})

    ax1.hist(df[feature])
    ax1.set_xlabel(f'{feature.capitalize()}')
    ax1.set_ylabel('Count')
    ax1.grid()

    stats_df = df[feature].describe()
    stats_text = f"""
Count: {stats_df['count']:.2f}
Mean: {stats_df['mean']:.2f}
StdDev: {stats_df['std']:.2f}
Minimum: {stats_df['min']:.2f}
1st Quartile: {stats_df['25%']:.2f}
Median: {stats_df['50%']:.2f}
3rd Quartile: {stats_df['75%']:.2f}
Maximum: {stats_df['max']:.2f}
        """
    ax2.axis('off')
    ax2.text(0.1, 0.5, stats_text, fontsize=12, family='monospace', verticalalignment='center', linespacing=2.5)
    plt.suptitle(f'Распределение {feature.capitalize()}')
    plt.show()


if __name__ == '__main__':
    from machine_learning.scikit_learn.prepared_data import get_ames_housing_data
    from machine_learning.scikit_learn.regression.utils import prepared_ames_housing_data

    features = ['Overall Qual', 'Overall Cond', 'Gr Liv Area', 'Central Air', 'Total Bsmt SF', 'SalePrice']
    df = prepared_ames_housing_data(
        df=get_ames_housing_data(r'C:\Users\GTA\Desktop\sklearn\AmesHousing.csv'),
        columns=features
    )
    df = df.dropna(axis=0)

    user_input = input("""
    Визуализация:
    1. Распределение конкретного признака
    2. Матрица диаграмм рассеяния признаков
    3. Матрица корреляций.
    """)
    if user_input == '1':
        available_features = '\n'.join([f'{idx}. {feature}' for idx, feature in enumerate(features, start=1)])
        feature_select = input(f'Выберете признак:\n{available_features} \n')
        get_feature_info(df, features[int(feature_select) - 1])
    elif user_input == '2':
        scatterplotmatrix(df.values, figsize=(12, 10), names=df.columns, alpha=0.5)
        plt.tight_layout()
        plt.show()
    elif user_input == '3':
        corr_matrix = np.corrcoef(df.values.T)
        heatmap(corr_matrix, row_names=df.columns, column_names=df.columns, cmap='RdYlGn')
        plt.tight_layout()
        plt.show()
