import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

def main():
    x, y = make_moons(n_samples=1000, shuffle=True, noise=0.1, random_state=1)
    dbscan = DBSCAN(
        eps=0.1,
        min_samples=5,
        metric="euclidean",
    )
    y_pred = dbscan.fit_predict(x)

    plt.scatter(x[y_pred==0, 0], x[y_pred==0, 1], c='green', label='Кластер 1', edgecolors='black', marker='o', s=40)
    plt.scatter(x[y_pred==1, 0], x[y_pred==1, 1], c='blue', label='Кластер 2', edgecolors='black', marker='s', s=40)
    plt.scatter(x[y_pred==-1, 0], x[y_pred==-1, 1], c='red', label='Шум', edgecolors='black', marker='.', s=40)
    plt.xlabel('Признак 1')
    plt.ylabel('Признак 2')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()