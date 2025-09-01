import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from machine_learning.deep_learning.basic_algorithms.NeuralNetMLP import NeuralNetMLP
from machine_learning.deep_learning.data import get_prepared_mnist_data
from machine_learning.deep_learning.utils import NeuralNetHelper


def main():
    """Пример реализации нейронной сети для распознавания цифр от 0 до 9."""
    nn_model = NeuralNetMLP(
        n_features=28 * 28,  # изображения 28*28 пикселей (уже в виде векторов размером 1*784).
        n_hidden=100,
        n_classes=10  # будем распознавать цифры от 0 до 9.
    )
    helper = NeuralNetHelper()

    # Подготовили данные: x_valid и y_valid - это отложенная выборка для финального тестирования.
    x, y = get_prepared_mnist_data()
    x_temp, x_valid, y_temp, y_valid = train_test_split(x, y, test_size=10000, random_state=123, stratify=y)
    x_train, x_test, y_train, y_test = train_test_split(x_temp, y_temp, test_size=5000, random_state=123,
                                                        stratify=y_temp)

    # Обучаем нейронную суть
    epochs = 50
    rate = 0.1
    losses, train_acc, test_acc = helper.train(nn_model, x_train, y_train, x_test, y_test, rate, epochs)
    user_input = input('Визуализация:\n1. Потери при обучении.\n2. Точность при обучении.\n')
    if user_input == '1':
        plt.plot(range(epochs), losses)
        plt.ylabel('Среднеквадратичная ошибка (MSE)')
        plt.xlabel('Эпохи')
        plt.show()
    elif user_input == '2':
        plt.plot(range(epochs), train_acc, label='Обучающая выборка')
        plt.plot(range(epochs), test_acc, label='Тестовая выборка')
        plt.legend(loc='lower right')
        plt.ylabel('Точность')
        plt.xlabel('Эпохи')
        plt.show()

    # Проверяем точность на отложенной выборке.
    valid_mse, valid_acc = helper.calc_mse_and_acc(nn_model, x_valid, y_valid)
    print(f'Точность на отложенной выборке: {valid_acc * 100:.2f}%')

    # image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
    # image = cv2.resize(image, (28, 28))
    # vector = image.flatten()
    # vector = (vector / vector.max() - 0.5) * 2
    # _, probas = nn_model.forward(-vector)
    # for idx, i in enumerate(probas):
    #     print(f'Цифра {idx} с вероятностью {i * 100:.2f} ')


if __name__ == '__main__':
    main()
