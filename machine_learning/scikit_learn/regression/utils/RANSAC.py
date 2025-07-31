from machine_learning.scikit_learn.prepared_data import get_ames_housing_data
from machine_learning.scikit_learn.regression.utils import prepared_ames_housing_data
from sklearn.linear_model import RANSACRegressor, LinearRegression
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Получаем данные для модели из файла в виде df.
    features = ['Overall Qual', 'Overall Cond', 'Gr Liv Area', 'Central Air', 'Total Bsmt SF', 'SalePrice']
    df = prepared_ames_housing_data(
        df=get_ames_housing_data(r'C:\Users\GTA\Desktop\sklearn\AmesHousing.csv'),
        columns=features
    )
    df = df.dropna(axis=0)

    # Для обучения используем 1 признак исключительно для примера.
    available_features = '\n'.join([f'{idx}. {feature}' for idx, feature in enumerate(features, start=1)])
    feature_select = input(f'Выберете признак:\n{available_features} \n')
    feature_name = features[int(feature_select) - 1]

    # Обучаем модель на инлаерах (95% точек от исходного набора).
    ransac = RANSACRegressor(
        estimator=LinearRegression(),
        min_samples=0.95,  # Кол-во инлаеров - 95% от исходного набора
        max_trials=100,  # Кол-во итераций
        residual_threshold=None,  # Порог считать по MAD
        random_state=123
    )
    x = df[[feature_name]].values
    y = df['SalePrice'].values[:, np.newaxis]

    ransac.fit(x, y)
    # Получаем точки, которые прошли порог. Эти точки по методу RANSAC НЕ считаются выбросами.
    inliers = ransac.inlier_mask_  # маска типа [True, False, ...]
    # Получаем точки, которые НЕ прошли порог. Эти точки по методу RANSAC считаются выбросами.
    outliers = np.logical_not(inliers)

    plt.scatter(x=x[inliers], y=y[inliers], c='steelblue', edgecolor='white', marker='o', label='Инлаеры')
    plt.scatter(x=x[outliers], y=y[outliers], c='limegreen', edgecolor='white', marker='s', label='Аутлаеры')
    plt.ylabel('SalePrice')
    plt.xlabel(f'{feature_name.capitalize()}')
    plt.title(f'График зависимости SalePrice от {feature_name.capitalize()}')
    plt.legend()
    plt.tight_layout()
    plt.show()


