import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import _name_estimators, make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


class MajorityVoteClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
            self,
            classifiers: list[BaseEstimator],
            vote: str = 'classlabel',
            weights: list[float] | None = None
    ):
        self.classifiers = classifiers
        self.vote = vote
        self.weights = weights
        self.named_estimators = {
            name: estimators for name, estimators in _name_estimators(classifiers)
        }

        self._label_encoder = LabelEncoder()  # Для кодирования меток из ['a', 'b', 'c'] в [0, 1, 2]
        self._fitted_classifiers = []  # Храним обученные классификаторы с преобразованными метками
        self.classes_ = None  # Нужно, чтобы наш класс считался классификатором через ClassifierMixin

    def fit(self, x, y):
        """
        Метод обучает каждый классификатор из self.classifiers. Если в y находятся метки, отличные от 0, 1 и тп, то
        метод преобразует эти метки в 0, 1 и тп. Все обученные классификаторы будут клонированы и сохранены в переменной
        self._transformed_classifiers.
        """
        if self.vote not in ['classlabel', 'probability']:
            raise ValueError("Vote must be 'probability' or 'classlabel'")

        if self.weights and len(self.weights) != len(self.classifiers):
            raise ValueError(f'Number of classifiers and weights must be equal')

        self._label_encoder.fit(y)  # Кодируем метки в 0, 1, 2 ...
        self.classes_ = self._label_encoder.classes_  # Сохраняем оригинальные метки в правильном порядке
        for estimator in self.classifiers:
            fitted_estimator = clone(estimator).fit(x, self._label_encoder.transform(y))
            self._fitted_classifiers.append(fitted_estimator)

        return self

    def predict(self, x):
        """
        Метод для предсказания метки для образца x.
        """
        if self.vote == 'probability':
            # Мажоритарное голосование через прогнозируемые вероятности классов.
            majority_vote = np.argmax(self.predict_proba(x), axis=1)
        else:
            # Мажоритарное голосование через предсказанные метки классов.
            predictions = [estimator.predict(x) for estimator in self._fitted_classifiers]
            predictions = np.vstack(predictions)
            majority_vote = np.apply_along_axis(
                lambda labels: np.argmax(np.bincount(labels, weights=self.weights)),
                axis=0,
                arr=predictions
            )
        majority_vote = self._label_encoder.inverse_transform(majority_vote)
        return majority_vote

    def predict_proba(self, x):
        """
        Метод для выполнения усреднения предсказанных вероятностей классов от всех базовых классификаторов ансамбля,
        то есть для каждого образца рассчитывается средние вероятности принадлежности к классам среди всех
        классификаторов, которые есть в ансамбле.
        """
        predicted_probas = np.asarray([clf.predict_proba(x) for clf in self._fitted_classifiers])
        avg_proba = np.average(predicted_probas, axis=0, weights=self.weights)
        return avg_proba

    def get_params(self, deep=True):
        """
        Метод нужен для того, чтобы получить доступ ко всем параметрам классификаторов, из которых состоит ансамбль.
        """
        if not deep:
            return super().get_params(deep=False)
        else:
            out = self.named_estimators.copy()
            for name, step in self.named_estimators.items():
                for key, value in step.get_params(deep=True).items():
                    out[f'{name}__{key}'] = value
            return out


if __name__ == '__main__':
    iris = load_iris()
    iris_target = iris.target[50:]
    iris_data = iris.data[50:, [1, 3]]

    x_train, x_test, y_train, y_test = train_test_split(
        iris_data, iris_target,
        stratify=iris_target,
        random_state=1,
        test_size=0.5
    )

    lg_model = make_pipeline(StandardScaler(), LogisticRegression(random_state=1, C=0.001))
    tree_model = DecisionTreeClassifier(random_state=0, max_depth=4, criterion='entropy')
    knn_model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=2, p=2, metric='minkowski'))
    ensemble = MajorityVoteClassifier(
        classifiers=[lg_model, tree_model, knn_model],
        vote='probability'
    )

    estimator_titles = ['Логистическая регрессия', 'Дерево решений', 'kNN-ближайшие соседи', 'Ансамбль']

    ensemble.fit(x_train, y_train)

    for classifier, label in zip((lg_model, tree_model, knn_model, ensemble), estimator_titles):
        score = cross_val_score(
            estimator=classifier,
            X=x_train,
            y=y_train,
            scoring='roc_auc',
            cv=10,
            n_jobs=-1,
        )
        print(f'ROC AUC: {np.mean(score):.2f} +/- {np.std(score):.2f} [{label}]')

    # Оптимизация гиперпараметров ансамбля через GridSearch CV.
    param_range = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    param_grid = {
        'pipeline-1__logisticregression__C': param_range,
        'decisiontreeclassifier__max_depth': [1, 2, 3, 4],
        'decisiontreeclassifier__criterion': ['gini', 'entropy'],
    }
    gs = GridSearchCV(
        estimator=ensemble,
        param_grid=param_grid,
        cv=10,
        scoring='roc_auc',
        n_jobs=-1
    )
    gs.fit(x_train, y_train)
    print(gs.best_params_)
    # Просматриваем оценку каждой комбинации параметров
    # for r, _ in enumerate(gs.cv_results_['mean_test_score']):
    #     mean_score = gs.cv_results_['mean_test_score'][r]
    #     std_dev = gs.cv_results_['std_test_score'][r]
    #     params = gs.cv_results_['params'][r]
    #     print(f'{mean_score: .3f} +/- {std_dev: .2f} {params} ')
