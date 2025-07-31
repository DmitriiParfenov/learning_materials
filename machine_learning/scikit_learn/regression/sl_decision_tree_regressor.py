import pandas as pd

from machine_learning.scikit_learn.prepared_data import get_ames_housing_data
from machine_learning.scikit_learn.regression.utils import prepared_ames_housing_data
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score


def get_data() -> pd.DataFrame:
    df = get_ames_housing_data(r'C:\Users\GTA\Desktop\sklearn\AmesHousing.csv')
    columns = ['Overall Qual', 'Overall Cond', 'Gr Liv Area', 'Central Air', 'Total Bsmt SF', 'SalePrice']
    df = prepared_ames_housing_data(df, columns)
    df = df.dropna(axis=0)
    df.columns = [x.replace(' ', '') for x in df.columns]
    return df

def main(df: pd.DataFrame) -> None:
    target = 'SalePrice'
    features = df.columns[df.columns != target]

    # Для обучения используем 1 признак исключительно для примера.
    available_features = '\n'.join([f'{idx}. {feature}' for idx, feature in enumerate(features, start=1)])
    feature_select = input(f'Выберете признак:\n{available_features} \n')
    feature= features[int(feature_select) - 1]

    # Объявляем переменные.
    x = df[[feature]].values
    y = df[target].values

    # Обучаем модель.
    tree_model = DecisionTreeRegressor(max_depth=3)
    tree_model.fit(x, y)
    y_pred = tree_model.predict(x)

    # Расчет MSE и R2.
    mse = mean_absolute_error(y_true=y, y_pred=y_pred)
    r2 = r2_score(y_true=y, y_pred=y_pred)

    # Визуализируем результаты.
    plt.scatter(x, y, c='steelblue', edgecolor='white', marker='o')
    sorted_idx = x.flatten().argsort()  # Массив с индексами отсортированных значений Х
    plt.plot(x[sorted_idx], y_pred[sorted_idx], color='black', lw=2)  # Линия тренда.
    plt.xlabel(f'{feature}\nMSE={mse:.2f}, R2={r2:.2f}')
    plt.ylabel(target)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    dataframe = get_data()
    main(dataframe)
