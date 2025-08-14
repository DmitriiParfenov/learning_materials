from sklearn.cluster import AgglomerativeClustering

from machine_learning.scikit_learn.cluster_analysis.hierarchical_clustering.get_example_data import get_example_data


def main():
    data = get_example_data(5, 3)

    ac = AgglomerativeClustering(n_clusters=2, metric='euclidean', linkage='complete')
    clusters = ac.fit_predict(data)
    print(clusters)


if __name__ == '__main__':
    main()
