from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


def main():
    # Получаем рандомные значение x с 2мя признаками и y (принадлежность к кластеру)
    x, y = make_blobs(n_samples=150, n_features=2, centers=3, cluster_std=0.5, shuffle=True, random_state=1)

    # Алгоритм k-means.
    k_mean = KMeans(
        n_clusters=3,
        n_init=10,  # 10 независимых анализов с разными центроидами, из которых выбирается наилучший по SSE
        init='random',  # Способ инициализации центроидов
        max_iter=100,
        tol=1e-04,  # Допуск сходимости (точность)
        random_state=0
    )
    y_pred = k_mean.fit_predict(x)
    centroids_coords = k_mean.cluster_centers_

    # Первый кластер, где x - это признак 1, а y - признак 2.
    plt.scatter(x=x[y_pred == 0, 0], y=x[y_pred == 0, 1], c='green', marker='o', label='Кластер 1', edgecolors='black')
    # Второй кластер.
    plt.scatter(x=x[y_pred == 1, 0], y=x[y_pred == 1, 1], c='blue', marker='s', label='Кластер 2', edgecolors='black')
    # Третий кластер.
    plt.scatter(x=x[y_pred == 2, 0], y=x[y_pred == 2, 1], c='orange', marker='v', label='Кластер 3', edgecolors='black')

    # Показываем центроиды.
    plt.scatter(x=centroids_coords[:, 0], y=centroids_coords[:, 1], s=250, c='red', marker='*', label='Центроиды')

    plt.xlabel('Признак 1')
    plt.ylabel('Признак 2')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    plt.show()


if __name__ == '__main__':
    main()
