from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np


def main():
    x, y = make_blobs(n_samples=1000, n_features=2, shuffle=True, centers=3, random_state=20, cluster_std=1.0)

    km = KMeans(
        n_clusters=3,
        n_init=10,
        init='k-means++',
        max_iter=300,
        tol=1e-4,
        random_state=20,
    )
    y_pred = km.fit_predict(x)
    unique_clusters = np.unique(y_pred)
    # Получаем силуэтный коэффициент для каждого образца в X.
    silhouette_coefficients = silhouette_samples(X=x, labels=y_pred, metric="euclidean")

    y_axis_lower, y_axis_upper = 0, 0
    y_ticks = []
    for idx, cluster in enumerate(unique_clusters):
        s_cfs = silhouette_coefficients[y_pred == cluster]
        s_cfs.sort()
        y_axis_upper += len(s_cfs)
        plt.barh(range(y_axis_lower, y_axis_upper), s_cfs, height=1)
        y_ticks.append((y_axis_lower + y_axis_upper) / 2)
        y_axis_lower += len(s_cfs)

    silhouette_avg = np.mean(silhouette_coefficients)
    plt.axvline(
        silhouette_avg,
        color="red",
        linestyle="--"
    )
    plt.yticks(ticks=y_ticks, labels=unique_clusters + 1)
    plt.ylabel('Кластер')
    plt.xlabel('Силуэтный коэффициент')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()