from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from machine_learning.scikit_learn.prepared_data import get_cancer_data


def svm_grid_search_tuning_example():
    """
    Пример функции для оптимизации гиперпараметров SVM классификатора через полный перебор всех комбинаций с
    использованием GridSearch.
    """
    # Данные для примера.
    df, x_train, x_test, y_train, y_test = get_cancer_data()

    # Пайплайн: стандартизация значений -> классификация через SVM.
    svm_pipline = make_pipeline(
        StandardScaler(),
        SVC(random_state=1)
    )

    # Выбор параметров для поиска наилучшей комбинации гиперпараметров.
    param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    param_grid = [
        {'svc__C': param_range, 'svc__kernel': ['linear']},  # для линейного SVM
        {'svc__C': param_range, 'svc__gamma': param_range, 'svc__kernel': ['rbf', 'sigmoid']}  # для нелинейного SVM
    ]

    # Создает поиск оптимальных параметров через GridSearch.
    gs = GridSearchCV(
        estimator=svm_pipline,
        param_grid=param_grid,
        cv=10,
        scoring='accuracy',
        refit=True,  # не нужно переобучать модель с наилучшей комбинацией гиперпараметров
        n_jobs=-1
    )
    gs.fit(x_train, y_train)

    # Наилучшие гиперпараметры.
    best_score = gs.best_score_
    best_params = gs.best_params_
    best_estimator = gs.best_estimator_

    print(f'Лучшие гиперпараметры: {best_params}.')
    print(f'Лучшая оценка: {best_score:.3f}')
    print(f'Точность при тестировании: {best_estimator.score(x_test, y_test):.3f}')


if __name__ == '__main__':
    svm_grid_search_tuning_example()
