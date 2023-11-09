import numpy as np


class KMeanClustering:
    def __init__(self, k=3):
        self.centroids = None
        self.k = k

    @staticmethod
    def l2norm(data_point, centroids):
        return np.sqrt(np.sum((data_point - centroids) ** 2, axis=1))

    def fit(self, x, max_iterations=300):
        self.centroids = np.random.uniform(np.amin(x, axis=0), np.amax(x, axis=0), size=(self.k, x.shape[1]))
        colors = list()
        for _ in range(max_iterations):
            colors = []
            for data_point in x:
                dist = KMeanClustering.l2norm(data_point, self.centroids)
                colors.append(np.argmin(dist))

            colors = np.array(colors)

            cluster_indices = []
            for i in range(self.k):
                cluster_indices.append(np.argwhere(colors == i))

            new_centers = []

            for idx, indices in enumerate(cluster_indices):
                if len(indices) == 0:
                    new_centers.append(self.centroids[idx])
                else:
                    new_centers.append(np.mean(x[indices], axis=0)[0])

            new_centers = np.array(new_centers)
            if np.max(self.centroids - new_centers) < 0.0001:
                return colors
            self.centroids = new_centers

        return colors

