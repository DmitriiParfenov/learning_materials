import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from machine_learning.scikit_learn.prepared_data import get_cancer_data
from machine_learning.scikit_learn.visualisation import (
    visualize_learning_curve,
    visualize_validation_curve
)


def main():
    df, x_train, x_test, y_train, y_test = get_cancer_data()

    # Пайплайн: стандартизация -> уменьшение признакового пространства -> обучение модели + классификация.
    lg_pipeline = make_pipeline(
        StandardScaler(),
        PCA(n_components=2),
        LogisticRegression(random_state=1, max_iter=10000)
    )
    lg_pipeline.fit(x_train, y_train)

    y_pred = lg_pipeline.predict(x_test)
    score = accuracy_score(y_true=y_test, y_pred=y_pred)
    print(f'Точность классификации: {score:.3f}')

    user_input = input('Визуализация\n1. Кривая обучения\n2. Кривая валидации\n ')
    if user_input == '1':
        visualize_learning_curve(
            estimator=lg_pipeline,
            x=x_train,
            y=y_train,
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=10
        )
    elif user_input == '2':
        visualize_validation_curve(
            estimator=lg_pipeline,
            x=x_train,
            y=y_train,
            param_name='logisticregression__C',
            param_range=[0.001, 0.01, 0.1, 1, 10, 100],
            cv=10
        )


if __name__ == '__main__':
    main()
