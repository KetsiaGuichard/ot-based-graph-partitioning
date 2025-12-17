"""
Initialization strategies for clustering, based on k-means usual init.
Provides random and k-means++ initialization methods.
"""

from random import sample, seed
import numpy as np
from sklearn.cluster import kmeans_plusplus
from scipy.spatial.distance import cdist
from src.clustering.utils import centroids_to_ot


class ClusteringInit:
    """Initialization

    Args:
        n (int): number of individuals
        k (int): number of class
    """

    def __init__(self, n: int, k: int, distance_matrix: np.ndarray, set_seed=None):
        self.n = n
        self.k = k
        self.distance_matrix = distance_matrix
        self.set_seed = set_seed
        if set_seed is not None:
            seed(set_seed)
            np.random.seed(set_seed)

    def random_init(self):
        """Pick k points at random in range(0,n)

        Returns:
            list(int): list of k integers
            np.ndarray: transportation plan from these centers
        """
        centers = sample(range(0, self.n), self.k)
        return centers, centroids_to_ot(self.distance_matrix, centers)

    def kmeanspp_init(self):
        """K-means ++ implementation

        Returns:
            list(int): list of k integers
            np.ndarray: transportation plan from these centers
        """
        centers = [np.random.choice(self.n)]
        for _ in range(1, self.k):
            dist_to_centers = self.distance_matrix[centers] ** 2
            min_dist = (
                dist_to_centers.min(axis=0)
                if dist_to_centers.ndim > 1
                else dist_to_centers
            )
            total = min_dist.sum()
            prob = min_dist / total
            next_center = np.random.choice(self.n, p=prob)
            while next_center in centers:
                next_center = np.random.choice(self.n, p=prob)
            centers.append(next_center)
        return centers, centroids_to_ot(self.distance_matrix, centers)

    def embedded_kmeanspp_init(self, d=None):
        """Apply K-means ++ directly on distance matrix

        Args:
            d (np.ndarray, optional): precomputed pairwise distance matrix. Defaults to None.

        Returns:
            list(int): list of k centers
            np.ndarray: transportation plan from these centers
        """
        _, centers = kmeans_plusplus(self.distance_matrix, self.k, n_local_trials=1)
        if d is None:
            d = cdist(self.distance_matrix, self.distance_matrix)
        return centers, centroids_to_ot(d, centers)
