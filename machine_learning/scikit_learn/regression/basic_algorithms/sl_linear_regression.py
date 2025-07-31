import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

from machine_learning.scikit_learn.prepared_data import get_ames_housing_data
from machine_learning.scikit_learn.regression.utils import prepared_ames_housing_data

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

    lr = LinearRegression()
    x = df[[feature_name]].values
    y = df['SalePrice'].values[:, np.newaxis]
    lr.fit(x, y)
    weights = [f'{w:.2f} * X{idx}' for idx, w in enumerate(lr.coef_.flatten().tolist(), start=1)]
    equation = f'{lr.intercept_.tolist()[0]:.2f} + {" + ".join(weights)}'

    # Результаты.
    user_input = input("""
    Визуализация:
    1. График линейной регрессии.
    2. Получить предсказание.
        """)

    if user_input == '1':
        plt.scatter(x=x, y=y, c='steelblue', edgecolor='white', s=70)
        plt.plot(
            x,
            lr.predict(x),
            color='black',
            lw=2,
            label=equation
        )
        plt.ylabel('SalePrice (standardized)')
        plt.xlabel(f'{feature_name.capitalize()} (standardized)')
        plt.title(f'График зависимости SalePrice от {feature_name.capitalize()}')
        plt.legend()
        plt.tight_layout()
        plt.show()
    elif user_input == '2':
        user_data = input('Введите значение признака:\n')
        try:
            user_data = float(user_data)
            y_pred = lr.predict(np.array([[user_data]]))
            print(f'Цена продажи: {y_pred.flatten()[0]:.2f} $')
        except ValueError:
            print('Ввели некорректное значение.')
