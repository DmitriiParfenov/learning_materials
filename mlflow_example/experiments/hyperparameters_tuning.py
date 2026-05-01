import math
from datetime import datetime
from typing import Sequence

import mlflow
import optuna
import xgboost as xgb
from mlflow.models import infer_signature
from optuna import Trial
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from mlflow_example.utils import visualize_residual_plot

X, Y = load_diabetes(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)
dx_train = xgb.DMatrix(x_train, label=y_train)
dx_test = xgb.DMatrix(x_test, label=y_test)


def objective(trial: Trial) -> float | Sequence[float]:
    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
        # Определяем гиперпараметры.
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
            "lambda": trial.suggest_float("lambda", 1e-3, 10, log=True),
            "alpha": trial.suggest_float("alpha", 1e-3, 10, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        }
        # Параметры только для деревьев.
        if params["booster"] in ["gbtree", "dart"]:
            params["max_depth"] = trial.suggest_int("max_depth", 3, 10)
            params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
            params["gamma"] = trial.suggest_float("gamma", 1e-3, 1.0, log=True)

        # Обучаем модель.
        model = xgb.train(params, dx_train)
        y_pred = model.predict(dx_test)
        mse = mean_squared_error(y_test, y_pred)

        # Логируем в MLFlow.
        mlflow.log_params(params)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", math.sqrt(mse))

        return mse


if __name__ == '__main__':
    mlflow.set_tracking_uri("http://localhost:55050")
    name = "Diabetes XGBoost hyperparameters tuning"
    experiment = mlflow.set_experiment(name)
    run_name = f"{name} {datetime.now(tz=None).strftime('%Y-%m-%d %H:%M:%S')}"

    with mlflow.start_run(run_name=run_name, experiment_id=experiment.experiment_id):
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=20)

        # Обучаем модель.
        xgb_model = xgb.train(study.best_params, dx_train)
        y_pred = xgb_model.predict(dx_test)
        residual_plot = visualize_residual_plot(y_pred, y_test)

        # Логируем в MLFlow
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_mse", study.best_value)
        mlflow.log_figure(figure=residual_plot, artifact_file="residual.png")
        model_signature = infer_signature(x_test, y_pred)
        mlflow.xgboost.log_model(xgb_model=xgb_model, name="XGBoost", signature=model_signature)
