import argparse
from datetime import datetime

import mlflow
import xgboost as xgb
from mlflow.models import infer_signature
from sklearn.metrics import mean_squared_error

from mlflow_example.utils import visualize_residual_plot, get_dataset

def configurate_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tuning_experiment_name", default="Diabetes XGBoost hyperparameters tuning", type=str)
    return parser


def get_best_params_for_xgboost_model(experiment_name: str) -> dict[str, str]:
    """
    Получает лучшие гиперпараметры XGBoost-модели из эксперимента MLflow.
    Функция обращается к MLflow Tracking Server, находит эксперимент по имени и выбирает run с наилучшим значением
    целевой метрики. Из этого run извлекаются параметры модели и значение метрики.
    Args:
        experiment_name (str): Имя эксперимента MLflow.
    Returns:
        dict: список параметров для запуска модели
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        return {}
    last_run_id = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=1,
        filter_string="status = 'FINISHED'",
        order_by=["metrics.best_mse ASC"],
        output_format="list"  # по дефолту pandas
    )
    if not last_run_id:
        return {}
    return last_run_id[0].data.params


if __name__ == '__main__':
    # Получаем название эксперимента по оптимизации гиперпараметров.
    parser = configurate_parser()
    tuning_experiment_name = parser.parse_args().tuning_experiment_name
    mlflow.set_tracking_uri("http://localhost:55050")
    # Создаем эксперимент по обучению модели.
    name = "Diabetes XGBoost training"
    experiment = mlflow.set_experiment(name)
    run_name = f"{name} {datetime.now(tz=None).strftime('%Y-%m-%d %H:%M:%S')}"
    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=run_name):
        x_train, x_test, y_train, y_test, dx_train, dx_test = get_dataset()
        params = get_best_params_for_xgboost_model(tuning_experiment_name)
        mlflow.log_params(params)

        # Обучаем модель.
        model = xgb.train(params=params, dtrain=dx_train)
        y_pred = model.predict(dx_test)
        residual_plot = visualize_residual_plot(y_pred, y_test)
        mse = mean_squared_error(y_test, y_pred)
        mlflow.log_metric("mse", mse)
        mlflow.log_figure(figure=residual_plot, artifact_file="residual.png")

        # Регистрируем модель в model registry.
        model_signature = infer_signature(x_test, y_pred)
        mlflow.xgboost.log_model(
            xgb_model=model,
            name="XGBoost",
            signature=model_signature,
            registered_model_name="diabetes-xgboost"
        )
