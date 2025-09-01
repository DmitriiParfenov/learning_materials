import matplotlib.pyplot as plt

from machine_learning.deep_learning.data import get_prepared_mnist_data


def visualize_numbers():
    x, y = get_prepared_mnist_data()
    fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
    ax = ax.ravel()
    for num in range(10):
        number = x[y == num][0].reshape(28, 28)
        ax[num].imshow(number, cmap='Greys')

    plt.show()


if __name__ == '__main__':
    visualize_numbers()
