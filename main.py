import numpy as np
from kmean import KMeanClustering
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


def main():
    data_points = make_blobs(n_samples=100, n_features=2, centers=5)
    data_points = data_points[0]
    start = 1
    end = 5
    plt.figure(figsize=(10*(end+1-start), 10))
    variances = []
    for k in range(start, end+1):
        plt.subplot(1, end+1-start, k)
        km = KMeanClustering(k=k)
        colors = km.fit(data_points, max_iterations=500)

        plt.scatter(data_points[:, 0], data_points[:, 1], c=colors)
        plt.scatter(km.centroids[:, 0], km.centroids[:, 1], c=range(k), marker="*", s=200)
        variance = 0
        for color in range(k):
            indices = np.argwhere(colors == color)
            if len(indices) == 0:
                continue
            variance += np.var(data_points[indices])
        variances.append(variance)
    plt.savefig("scatter_cluster.png")
    plt.figure()
    plt.plot([k for k in range(start, end+1)], variances)
    plt.savefig("deciding_k.png")


if __name__ == "__main__":
    main()