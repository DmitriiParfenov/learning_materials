import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform
from machine_learning.scikit_learn.cluster_analysis.hierarchical_clustering.get_example_data import get_example_data

def main():
    df = get_example_data(5, 3)

    # Получаем матрицу расстояний всех точек (расчет через евклидово расстояние).
    matrix_dist = squareform(pdist(X=df, metric='euclidean'))
    df_dist = pd.DataFrame(data=matrix_dist, index=df.index, columns=df.index)

    # Иерархическая агломеративная кластеризация по алгоритму "Полная связь".
    clusters = linkage(y=df.values, method='complete', metric='euclidean')
    df_clusters = pd.DataFrame(
        data=clusters,
        columns=['row label 1', ' row label 2', 'distance', 'no. of items in clust.'],
        index=[f'cluster {(i + 1)}' for i in range(clusters.shape[0])]
    )

    # Получение дендрограммы.
    dendrogram(clusters, labels=df.index, color_threshold=np.inf)
    plt.tight_layout()
    plt.ylabel('Евклидово расстояние')
    plt.show()


if __name__ == '__main__':
    main()
