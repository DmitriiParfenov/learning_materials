from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def main():
    x, y = make_blobs(n_features=2, n_samples=1000, random_state=1, shuffle=True, centers=3, cluster_std=0.5)

    # Оптимизируем количество кластеров k.
    distortions = []  # для хранения SSE (искажения).
    for k_number in range(1, 11):
        km = KMeans(
            n_clusters=k_number,
            n_init=10,
            init='k-means++',
            random_state=1,
            max_iter=300
        )
        km.fit(x)
        distortions.append(km.inertia_)

    # Строим график SSE от количества кластеров.
    plt.plot(
        range(1, len(distortions)+1),
        distortions,
        marker='o',
        c='black',
    )
    plt.xlabel('Количество кластеров')
    plt.ylabel('Искажение (SSE)')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
