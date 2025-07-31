import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from machine_learning.scikit_learn.prepared_data import get_ames_housing_data
from machine_learning.scikit_learn.regression.utils import prepared_ames_housing_data
from machine_learning.scikit_learn.regression.visualisation import visualize_residual_plot


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
    # Объявляем переменные.
    x = df[features].values
    y = df[target].values
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, random_state=1, test_size=0.3
    )

    # Обучаем модель.
    forest_model = RandomForestRegressor(
        criterion="squared_error",
        n_estimators=1000,
        random_state=1
    )
    forest_model.fit(x_train, y_train)
    y_train_pred = forest_model.predict(x_train)
    y_test_pred = forest_model.predict(x_test)

    # Расчет MSE и R2.
    mse_train = mean_absolute_error(y_true=y_train, y_pred=y_train_pred)
    r2_train = r2_score(y_true=y_train, y_pred=y_train_pred)
    mse_test = mean_absolute_error(y_true=y_test, y_pred=y_test_pred)
    r2_test = r2_score(y_true=y_test, y_pred=y_test_pred)

    visualize_residual_plot(
        y_train_pred,
        y_test_pred,
        y_train,
        y_test,
        {'train': {'MSE': mse_train, 'R2': r2_train}, 'test': {'MSE': mse_test, 'R2': r2_test}}
    )
    plt.show()


if __name__ == '__main__':
    dataframe = get_data()
    main(dataframe)
