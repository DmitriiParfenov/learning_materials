from typing import Any

import numpy as np
import scipy
from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from machine_learning.scikit_learn.prepared_data import get_cancer_data


def get_scores_by_nested_cross_val_and_randomized_search(
        estimator: BaseEstimator,
        x: np.array,
        y: np.array,
        param_distributions: list[dict[str, Any]],
        n_iter: int = 10,
        cv: int = 10,
        n_jobs: int = 1,
        scoring: str = 'accuracy'
) -> list[float]:
    """
    Метод оценивает классификатор estimator через nested cross-validation. Для этого обучающий набор x разделяется на
    k-folds. Внутри каждого цикла определяется модель с наилучшей комбинацией гиперпараметров, найденной при помощи
    RandomizedSearchCV. Оценка этой модели является результирующей в рамках одного цикла. В результате метод возвращает
    список оценок моделей с наилучшей комбинацией гиперпараметров.

    Args:
        estimator: классификатор
        x: обучающий набор
        y: метки классов
        param_distributions: список гиперпараметров для оптимизации
        n_iter: количество итераций поиска оптимальных гиперпараметров
        cv: количество folds
        n_jobs: количество процессоров
        scoring: способ оценивания модели
    Returns:
        list[float]: список оценок моделей
    """
    rs_tuning = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
    )

    return cross_val_score(
        estimator=rs_tuning,
        X=x,
        y=y,
        cv=cv,
        n_jobs=n_jobs,
        scoring=scoring
    )


def select_algorithm_example():
    """
    Метод для сравнения двух алгоритмов между собой - SVM и DecisionTreeClassifier.
    """
    # Тестовые данные.
    df, x_train, x_test, y_train, y_test = get_cancer_data()

    # Классификатор на основе SVM.
    svm_model = make_pipeline(
        StandardScaler(),
        SVC(random_state=1)
    )
    svm_param_range = scipy.stats.loguniform(0.0001, 1000.0)
    svm_param_grid = [
        {'svc__C': svm_param_range, 'svc__kernel': ['linear']},
        {'svc__C': svm_param_range, 'svc__gamma': svm_param_range, 'svc__kernel': ['rbf', 'sigmoid']}
    ]

    # Классификатор на основе DecisionTree.
    tree_model = DecisionTreeClassifier(random_state=1)
    tree_param_grid = [
        {'max_depth': [1, 2, 3, 4, 5, None], 'criterion': ['gini', 'entropy']}
    ]

    svm_scores = get_scores_by_nested_cross_val_and_randomized_search(
        estimator=svm_model,
        x=x_train,
        y=y_train,
        param_distributions=svm_param_grid,
        n_iter=10,
        cv=10,
        n_jobs=-1,
        scoring='accuracy',
    )
    tree_scores = get_scores_by_nested_cross_val_and_randomized_search(
        estimator=tree_model,
        x=x_train,
        y=y_train,
        param_distributions=tree_param_grid,
        n_iter=10,
        cv=10,
        n_jobs=-1,
        scoring='accuracy',
    )

    print(f'Точность SVM: {np.mean(svm_scores):.2f} +/- {np.std(svm_scores):.2f}')
    print(f'Точность Tree: {np.mean(tree_scores):.2f} +/- {np.std(tree_scores):.2f}')


if __name__ == '__main__':
    select_algorithm_example()
