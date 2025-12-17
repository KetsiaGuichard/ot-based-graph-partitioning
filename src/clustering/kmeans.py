"""
k-means clustering on attributed graphs.
Cluster nodes using a distance matrix that combines 
topological and attribute information.
"""

from abc import ABC, abstractmethod
import warnings
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from src.clustering.utils import (
    compute_cellules,
    compute_centroids,
    compute_kmeans_criteria,
)
from src.distances import combine_alpha


class GraphKMeans(ABC):
    """
    Abstract base class for partitioning attributed graphs with Kmeans methods.

    Args:
        k (int): Number of clusters/classes.
        centroids_init (list[int]): List of k integers for initial centroids.
        max_iter (int, optional): Maximum number of iterations. Default is 100.
    """

    def __init__(self, k: int, centroids_init, max_iter: int = 100):
        """
        Initialize kmeans methods.

        Args:
            k (int): Number of clusters.
            centroids_init (list[int]): Initial centroid indices.
            max_iter (int, optional): Maximum iterations. Default is 100.
        """
        self.k = k
        self.centroids_init = centroids_init
        self.max_iter = max_iter
        self.centroids = None
        self.labels = None
        self.iterations = None
        self.criteria = None

    def get_results(self):
        """
        Return clustering results: centroids, labels, iterations, and criteria.

        Returns:
            dict: Results with centroids, labels, iterations, and kmeans criteria.
        """
        return {
            "centroids": self.centroids,
            "labels": self.labels,
            "iterations": self.iterations,
            "criteria": self.criteria,
        }

    def same_assignment(self, labels, new_labels):
        """
        Check if two label assignments are the same.
        """
        return np.array_equal(labels, new_labels)

    @abstractmethod
    def partitioning(self):
        """
        Abstract method for partitioning the graph. Must be implemented in subclasses.
        """

    @abstractmethod
    def update_alpha(self):
        """
        Abstract method for updating alpha (level depends of the method used)
        """

    @abstractmethod
    def update_centroids(self, distance_matrix, labels):
        """
        Abstract method for updating centroids
        """


class FrechetKMeans(GraphKMeans):
    """
    K-means implementation for attributed graph clustering.

    Args:
        distance_matrix (np.ndarray): Matrix of distances (topological and attributes combined).
        k (int): Number of clusters/classes.
        centroids_init (list[int]): List of k integers for initial centroids.
        max_iter (int, optional): Maximum number of iterations. Default is 100.
    """

    def __init__(self, k: int, centroids_init: list, max_iter: int = 100):
        """
        Initialize the clustering and run the k-medoids algorithm.

        Args:
            distance_matrix (np.ndarray): Distance matrix.
            k (int): Number of clusters.
            centroids_init (list[int]): Initial centroid indices.
            max_iter (int, optional): Maximum iterations. Default is 100.
        """
        super().__init__(k, centroids_init, max_iter)

    def update_alpha(
        self,
        structural_matrix,
        attributes_matrix,
        centroids,
        labels,
        alpha_type="global",
    ):
        n = structural_matrix.shape[0]
        alphas = np.zeros(n)
        if alpha_type == "local":
            for i in range(n):
                centroid_i = centroids[labels[i]]
                alphas[i] = int(
                    structural_matrix[i, centroid_i] <= attributes_matrix[i, centroid_i]
                )
            return alphas
        numerator = 0
        denominator = 0
        for i in set(labels):
            centroid = [int(centroids[i])]
            idx = np.where(labels == i)[0]
            structural_distances = structural_matrix[np.ix_(idx, centroid)]
            attributes_distances = attributes_matrix[np.ix_(idx, centroid)]
            diff_distances = structural_distances - attributes_distances
            numerator_class = (structural_distances * attributes_distances).sum()
            denominator_class = diff_distances.sum()
            if alpha_type == "class":
                alpha_class = (
                    -numerator_class / denominator_class
                    if denominator_class != 0
                    else 0.5
                )
                alphas[idx] = np.clip(alpha_class, 0, 1)
            else:
                numerator += numerator_class
                denominator += denominator_class
        if alpha_type == "class":
            return alphas
        alpha = -numerator / denominator if denominator != 0 else 0.5
        return np.clip(alpha, 0, 1)

    def update_centroids(self, distance_matrix_squared, labels):
        new_centroids = np.array([])
        for i in range(0, self.k):
            idx = np.where(labels == i)[0]
            if len(idx) > 0:
                new_centroid_idx = compute_centroids(distance_matrix_squared, idx)
                new_centroids = np.append(new_centroids, idx[new_centroid_idx])
            else:
                new_centroids = np.append(new_centroids, idx)
        return new_centroids.astype(int)

    def partitioning_simple(self, structural_matrix):
        """
        Perform k-medoids clustering on the distance matrix.

        Args:
            structural_matrix (np.ndarray): Distance matrix.

        Returns:
            tuple: (centroids, labels, iterations, kmeans criteria)
        """
        iter_ct = 0
        centroids = self.centroids_init
        old_centroids = [l + 1 for l in self.centroids_init]

        data_carre = structural_matrix**2

        while (iter_ct < self.max_iter) & (set(centroids) != set(old_centroids)):
            labels = compute_cellules(data_carre, centroids)
            new_centroids = self.update_centroids(data_carre, labels)
            old_centroids = centroids
            centroids = new_centroids.astype(int)
            iter_ct = iter_ct + 1

        kmeans_criteria = compute_kmeans_criteria(structural_matrix, labels, centroids)

        return {
            "centroids": centroids,
            "labels": labels,
            "iter_ct": iter_ct,
            "kmeans_criteria": kmeans_criteria,
        }

    def partitioning(
        self, structural_matrix, attributes_matrix, alpha, alpha_type="global"
    ):
        default_alpha = [0.5 if isinstance(alpha, bool) and alpha is True else alpha]
        distance_matrix = combine_alpha(
            structural_matrix, attributes_matrix, default_alpha
        )

        if isinstance(alpha, float):
            results = self.partitioning_simple(distance_matrix)
            return {
                "centroids": results["centroids"],
                "labels": results["labels"],
                "iter_ct": results["iter_ct"],
                "kmeans_criteria": results["kmeans_criteria"],
                "alpha": alpha,
            }

        if alpha is True:
            centroids = self.centroids_init
            labels = compute_cellules(distance_matrix, centroids)
            old_labels = np.array([])
            iter_ct = 0
            while (iter_ct < self.max_iter) & (
                not self.same_assignment(old_labels, labels)
            ):
                old_labels = labels
                centroids = self.update_centroids(distance_matrix, labels)
                new_alpha = self.update_alpha(
                    structural_matrix,
                    attributes_matrix,
                    centroids,
                    labels,
                    alpha_type=alpha_type,
                )
                distance_matrix = combine_alpha(
                    structural_matrix, attributes_matrix, new_alpha
                )
                labels = compute_cellules(distance_matrix, centroids)
                iter_ct += 1

            kmeans_criteria = compute_kmeans_criteria(
                distance_matrix, labels, centroids
            )

            return {
                "centroids": centroids,
                "labels": labels,
                "iter_ct": iter_ct,
                "kmeans_criteria": kmeans_criteria,
                "alpha": new_alpha,
            }


class EmbeddedKMeans(GraphKMeans):
    """
    K-means implementation for embedded (attributed) graph clustering.

    Args:
        structural_matrix (np.ndarray): Matrix of topological distances.
        attributes_matrix (np.ndarray): Matrix of attributes distances.
        k (int): Number of clusters/classes.
        centroids_init (list[int]): List of k integers for initial centroids.
        max_iter (int, optional): Maximum number of iterations. Default is 100.
    """

    def __init__(
        self,
        k: int,
        centroids_init: list,
        max_iter: int = 100,
    ):
        """
        Initialize the clustering and run the k-medoids algorithm.

        Args:
            structural_matrix (array): Structural distance matrix.
            attributes_matrix (array): Attributes distance matrix.
            k (int): Number of clusters.
            centroids_init (list[int]): Initial centroid indices.
            alpha (bool | float, optional): True if alpha optimized,
                float (between 0 and 1) otherwise.
                Default is non-alpha optimized version, with alpha = 0.5
            max_iter (int, optional): Maximum iterations. Default is 100.
        """
        super().__init__(k, centroids_init, max_iter)
        self.centroids_init = centroids_init

    def init_alpha(self, matrix):
        """
        Initialize alpha values.
        
        Args:
            matrix (np.ndarray): Distance matrix.
        
        Returns:
            list: Initial alpha values.
        """
        return [0.5 for i in range(matrix.shape[0])]

    def update_alpha(self, diff_matrix, attributes_matrix, centroids, labels):
        """
        Update alpha values for each node.
        Args:
            diff_matrix (np.ndarray): Difference between structural and attributes matrices.
            attributes_matrix (np.ndarray): Attributes distance matrix.
            centroids (np.ndarray): Current centroids.
            labels (np.ndarray): Current labels.
        Returns:
            np.ndarray: Updated alpha values."""
        n = diff_matrix.shape[0]
        alphas = np.zeros(n)

        for i in range(n):
            centroid_i = centroids[labels[i]]
            dist_i = centroid_i - attributes_matrix[i]
            diff_i = diff_matrix[i]
            numerateur = np.dot(dist_i, diff_i)
            denominateur = np.dot(diff_i, diff_i)
            alpha_i = numerateur / denominateur if denominateur != 0 else 0.5
            alphas[i] = np.clip(alpha_i, 0, 1)
        return alphas

    def update_centroids(self, distance_matrix, labels):
        """
        Update centroids based on current labels.
        """
        centroids = np.zeros((self.k, distance_matrix.shape[1]))
        for i in range(self.k):
            cluster_points = distance_matrix[labels == i]
            centroids[i] = cluster_points.mean(axis=0)
        return centroids

    def update_labels(self, distance_matrix, centroids):
        """
        Update labels based on current centroids.
        """
        distances = cdist(distance_matrix, centroids, metric="euclidean")
        return np.argmin(distances, axis=1)

    def partitioning_simple(self, structural_matrix):
        """
        Partitioning grapg without attributes.

        Args:
            structural_matrix (np.ndarray): Matrix of topological distances.

        Returns:
            tuple: (centroids, labels, iterations, kmeans criteria)
        """
        distance_matrix = structural_matrix
        init = distance_matrix[self.centroids_init]

        kk_results = KMeans(n_clusters=self.k, init=init, max_iter=self.max_iter).fit(
            distance_matrix
        )
        return {
            "centroids": kk_results.cluster_centers_,
            "labels": kk_results.labels_,
            "iter_ct": kk_results.n_iter_,
            "kmeans_criteria": kk_results.inertia_,
        }

    def partitioning(self, structural_matrix, attributes_matrix, alpha: bool | float):
        """
        Partitioning graph with attributes.

        Args:
            structural_matrix (np.ndarray): Matrix of topological distances.
            attributes_matrix (np.ndarray): Matrix of attributes distances.
            alpha (bool | float, optional): True if alpha optimized,
                float (between 0 and 1) otherwise.

        Returns:
            tuple: (centroids, labels, iterations, kmeans criteria)
        """
        default_alpha = [0.5 if isinstance(alpha, bool) and alpha is True else alpha]
        distance_matrix = combine_alpha(
            structural_matrix, attributes_matrix, default_alpha
        )
        init = distance_matrix[self.centroids_init]

        # Non alpha optimized
        if isinstance(alpha, float):
            kk_results = KMeans(
                n_clusters=self.k, init=init, max_iter=self.max_iter
            ).fit(distance_matrix)
            return {
                "centroids": kk_results.cluster_centers_,
                "labels": kk_results.labels_,
                "iter_ct": kk_results.n_iter_,
                "kmeans_criteria": kk_results.inertia_,
                "alpha": alpha,
            }

        # Alpha optimized
        if alpha is True:
            centroids = init
            labels = self.update_labels(distance_matrix, centroids)
            old_labels = np.array([])
            iter_ct = 0
            diff_matrix = structural_matrix - attributes_matrix
            while (iter_ct < self.max_iter) & (
                not self.same_assignment(old_labels, labels)
            ):
                old_labels = labels
                centroids = self.update_centroids(distance_matrix, labels)
                alphas = self.update_alpha(
                    diff_matrix, attributes_matrix, centroids, labels
                )
                distance_matrix = combine_alpha(
                    structural_matrix, attributes_matrix, alphas
                )
                labels = self.update_labels(distance_matrix, centroids)
                iter_ct += 1

            return {
                "centroids": centroids,
                "labels": labels,
                "iter_ct": iter_ct,
                "kmeans_criteria": None,
                "alpha": alphas,
            }
        warnings.warn("unknown alpha option")
        return None
