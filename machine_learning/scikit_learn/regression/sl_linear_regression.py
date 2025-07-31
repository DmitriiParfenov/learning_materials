import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.formula.api import ols

from machine_learning.scikit_learn.prepared_data import get_ames_housing_data
from machine_learning.scikit_learn.regression.utils import prepared_ames_housing_data
from machine_learning.scikit_learn.regression.visualisation import (
    visualize_residual_plot,
    anova_results,
    get_statistics_result,
    get_accuracy_scores,
)


def main():
    # Получаем данные для модели из файла в виде df.
    features = ['Overall Qual', 'Overall Cond', 'Gr Liv Area', 'Central Air', 'Total Bsmt SF', 'SalePrice']
    df = prepared_ames_housing_data(
        df=get_ames_housing_data(r'C:\Users\GTA\Desktop\sklearn\AmesHousing.csv'),
        columns=features
    )
    df = df.dropna(axis=0)
    df.columns = [x.replace(' ', '') for x in df.columns]

    # Линейный регрессионный анализ.
    target = 'SalePrice'
    features = df.columns[df.columns != target]
    x = df[features].values
    y = df[target].values
    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        test_size=0.3,
        random_state=123
    )
    lr = LinearRegression()
    lr.fit(x_train, y_train)

    # Расчеты.
    y_train_pred = lr.predict(x_train)
    y_test_pred = lr.predict(x_test)
    mae_train = mean_absolute_error(y_true=y_train, y_pred=y_train_pred)
    mae_test = mean_absolute_error(y_true=y_test, y_pred=y_test_pred)

    # Объявляем модель для расчета статистик (чтобы самому не считать, в реале такое не использовать).
    model = ols(f'{target} ~ {" + ".join(features)}', data=df).fit()

    # Результаты.
    user_input = input("""
Визуализация:
1. График остатков.
2. Anova.
3. Коэффициенты регрессии.
4. Усложнение модели.
""")
    if user_input == '1':
        figure, _ = visualize_residual_plot(
            y_train_pred,
            y_test_pred,
            y_train,
            y_test,
            {'test': {'MAE': mae_train}, 'train': {'MAE': mae_test}}
        )
        plt.tight_layout()
        plt.show()
    elif user_input == '2':
        lr_anova = anova_results(model)
        lr_accuracy = get_accuracy_scores(model)
        print(lr_anova)
        print(lr_accuracy)
    elif user_input == '3':
        coef_data = get_statistics_result(model)
        print(coef_data)
    elif user_input == '4':
        # Для примера берем общую квадратичную модель.
        pn_model = PolynomialFeatures(degree=2, include_bias=False)
        quad_features_values = pn_model.fit_transform(x)
        quad_features_names = pn_model.get_feature_names_out(features)

        # Объявляем модель для расчета статистик (чтобы самому не считать, в реале такое не использовать).
        interactions = []
        quadratic = []
        for name in quad_features_names:
            if ' ' in name:
                interactions.append(name.replace(' ', ':'))
            elif '^2' in name:
                quadratic.append(f'I({name.replace("^2", "**2")})')
        formula = 'y ~ ' + ' + '.join(features) + ' + ' + ' + '.join(interactions) + ' + ' + ' + '.join(quadratic)
        quad_df = pd.DataFrame(quad_features_values, columns=quad_features_names)
        quad_df[target] = df[target].values
        quad_model = ols(formula, data=quad_df).fit()

        # Линейная регрессия (полином)
        x_quad = quad_df[quad_features_names].values
        x_quad_train, x_quad_test, y_train, y_test = train_test_split(
            x_quad, y,
            test_size=0.3,
            random_state=123
        )
        quad_lr = LinearRegression()
        quad_lr.fit(x_quad_train, y_train)

        # Расчеты.
        y_train_pred = quad_lr.predict(x_quad_train)
        y_test_pred = quad_lr.predict(x_quad_test)
        mae_train = mean_absolute_error(y_true=y_train, y_pred=y_train_pred)
        mae_test = mean_absolute_error(y_true=y_test, y_pred=y_test_pred)

        # Результаты.
        user_input = input("""
        Визуализация:
        1. График остатков.
        2. Anova.
        3. Коэффициенты регрессии.
        """)
        if user_input == '1':
            figure, _ = visualize_residual_plot(
                y_train_pred,
                y_test_pred,
                y_train,
                y_test,
                {'test': {'MAE': mae_train}, 'train': {'MAE': mae_test}}
            )
            plt.tight_layout()
            plt.show()
        elif user_input == '2':
            lr_anova = anova_results(quad_model)
            lr_accuracy = get_accuracy_scores(quad_model)
            print(lr_anova)
            print(lr_accuracy)
        elif user_input == '3':
            coef_data = get_statistics_result(quad_model)
            print(coef_data)


if __name__ == '__main__':
    main()
