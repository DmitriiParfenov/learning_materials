import xgboost as xgb
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split


def get_dataset():
    """
    Демонстрационный dataset. Лучше использовать отдельную сущность для подготовки dataset, который можно использовать
    в текущем скрипте путем выгрузки из какого-либо хранилища.
    """
    X, Y = load_diabetes(return_X_y=True)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
    dX_train, dX_test = xgb.DMatrix(X_train, label=Y_train), xgb.DMatrix(X_test, label=Y_test)
    return X_train, X_test, Y_train, Y_test, dX_train, dX_test
