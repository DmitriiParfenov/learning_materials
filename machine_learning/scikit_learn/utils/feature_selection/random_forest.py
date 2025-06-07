from machine_learning.scikit_learn.prepared_data.wine_data import get_wine_data
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt


def main():
    df_wine, x_train, x_train_std, x_test, x_test_std, y_train, y_test = get_wine_data()

    # Обучаем модель и смотрим значимость признаков.
    forest_model = RandomForestClassifier(n_estimators=500, random_state=1)
    forest_model.fit(x_train, y_train)
    features_importance = forest_model.feature_importances_  # нормализованы => их сумма равна 1.
    features_title = df_wine.columns[1:]

    # Сохраняем признаки и их значимость в df.
    df_features = pd.DataFrame({'feature': features_title, 'importance': features_importance})
    df_features.sort_values(by=['importance'], ascending=False, inplace=True)

    # Рисуем график.
    plt.bar(x=df_features['feature'], height=df_features['importance'])
    plt.title('Значимость признаков')
    plt.xlim([-1, len(df_features['importance'].values)])
    plt.ylim([0, 0.2])
    plt.xlabel('Название признаков')
    plt.ylabel('Значимость, н.е.')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()

