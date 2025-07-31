import numpy as np


class LinearRegressionGD:
    def __init__(self, rate: float, epochs: int = 50, random_state: int = 1) -> None:
        self.rate = rate
        self.epochs = epochs
        self.random_state = random_state
        self._weights = np.array(0)
        self._bias = 0.0
        self.losses = []

    def fit(self, x, y):
        rgen = np.random.RandomState(self.random_state)
        self._weights = rgen.normal(loc=0.0, scale=0.1, size=x.shape[1])

        for _ in range(self.epochs):
            z = self.net_input(x)
            errors = (y - z)
            self._weights += self.rate * 2 * x.T.dot(errors) / x.shape[0]
            self._bias += self.rate * 2 * errors.mean()
            loss = (errors ** 2).mean()
            self.losses.append(loss)

        return self

    def net_input(self, x):
        return np.matmul(x, self._weights) + self._bias

    def predict(self, x):
        return self.net_input(x)


if __name__ == '__main__':
    from machine_learning.scikit_learn.prepared_data import get_ames_housing_data
    from machine_learning.scikit_learn.regression.utils import prepared_ames_housing_data
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt

    # Получаем данные для модели из файла в виде df.
    features = ['Overall Qual', 'Overall Cond', 'Gr Liv Area', 'Central Air', 'Total Bsmt SF', 'SalePrice']
    df = prepared_ames_housing_data(
        df=get_ames_housing_data(r'C:\Users\GTA\Desktop\sklearn\AmesHousing.csv'),
        columns=features
    )
    df = df.dropna(axis=0)

    # Объявляем модель.
    lr = LinearRegressionGD(rate=0.1)

    # Получаем данные от пользователя.
    available_features = '\n'.join([f'{idx}. {feature}' for idx, feature in enumerate(features, start=1)])
    feature_select = input(f'Выберете признак:\n{available_features} \n')
    feature_name = features[int(feature_select) - 1]

    # Стандартизируем данные.
    y_sc = StandardScaler()
    x_sc = StandardScaler()
    x_std = x_sc.fit_transform(df[[feature_name]].values)
    # Преобразуем в 2D массив через np.newaxis и обратно через .flatten().
    y_std = y_sc.fit_transform(df['SalePrice'].values[:, np.newaxis]).flatten()

    # Обучаем модель.
    lr.fit(x_std, y_std)

    # Результаты.
    user_input = input("""
Визуализация:
1. Функция потерь в зависимости от количества эпох.
2. График линейной регрессии.
3. Получить предсказание.
    """)

    if user_input == '1':
        plt.plot(lr.losses)
        plt.ylabel('MSE')
        plt.xlabel('Количество эпох')
        plt.title('Функция потерь в зависимости от количества эпох')
        plt.tight_layout()
        plt.show()
    elif user_input == '2':
        plt.scatter(x=y_std,y=x_std, c='steelblue', edgecolor='white', s=70)
        plt.plot(x_std, lr.predict(x_std), color='black', lw=2, label=f'y = {lr._weights[0]:.2f} * x + {lr._bias:.2f}')
        plt.ylabel('SalePrice (standardized)')
        plt.xlabel(f'{feature_name.capitalize()} (standardized)')
        plt.title(f'График зависимости SalePrice от {feature_name.capitalize()}')
        plt.legend()
        plt.tight_layout()
        plt.show()
    elif user_input == '3':
        user_data = input('Введите значение признака:\n')
        try:
            user_data = float(user_data)
            user_data_std = x_sc.transform(np.array([[user_data]]))
            predicted_std = lr.predict(user_data_std)
            result = y_sc.inverse_transform(predicted_std.reshape(-1, 1))
            print(f'Цена продажи: {result.flatten()[0]:.2f} $')
        except ValueError:
            print('Ввели некорректное значение.')



