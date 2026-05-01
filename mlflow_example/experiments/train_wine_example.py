from datetime import datetime
from itertools import product
from typing import Type, Any

import mlflow
import numpy as np
from mlflow.models import infer_signature
from sklearn.base import BaseEstimator
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATASET = load_wine()
X_train, X_test, Y_train, Y_test = train_test_split(DATASET.data, DATASET.target, test_size=0.3, random_state=42)
PARAMS = {
    "C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
    "solver": ["saga", "lbfgs"]
}


def train_model(
        model: Type[BaseEstimator],
        model_name: str,
        parameters: dict[str, Any],
        x_train: np.ndarray,
        x_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        preprocessing: bool = False
) -> None:
    if preprocessing:
        model = make_pipeline(StandardScaler(), PCA(n_components=min(10, x_train.shape[1])), model(**parameters))
    # Логируем параметры обучения модели.
    mlflow.log_params(params=parameters)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    # Логируем метрики.
    acc = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", acc)
    # Определяем "подпись" модели.
    model_signature = infer_signature(x_test, y_pred)
    # Логируем модель.
    mlflow.sklearn.log_model(model, name=model_name, signature=model_signature)


if __name__ == '__main__':
    mlflow.set_tracking_uri("http://localhost:55050")
    name = "WineClassificationExperiment"
    experiment = mlflow.set_experiment(name)

    run_name = f"{name} {datetime.now(tz=None).strftime('%Y-%m-%d %H:%M:%S')}"
    for number, (C, solver) in enumerate(product(*PARAMS.values()), start=1):
        with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=f"[{number}] {run_name}"):
            train_model(
                LogisticRegression,
                "logistic_regression",
                {"C": C, "solver": solver},
                X_train,
                X_test,
                Y_train,
                Y_test,
                True,
            )
