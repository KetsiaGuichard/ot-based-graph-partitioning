"""
Evaluation metric (internal, external or method-related) helpers for clustering experiments.
"""

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import (
    adjusted_rand_score,
    silhouette_samples,
)
from src.clustering.utils import compute_kmeans_criteria, labels_to_transport_plan
from .utils import prediction_to_partition, gw_loss, fgw_loss


class MethodEvaluation:
    """Compute FGW/GW related metrics for a single clustering."""

    def __init__(self, distance_matrix, cellules, c1, c2):
        self.ot = self.compute_transport_plan(cellules)
        self.c2 = self.compute_c2(c2, self.ot)
        self.kmeans_criteria = compute_kmeans_criteria(distance_matrix, cellules)
        self.gw_loss_isolated = gw_loss(c1, self.c2, self.ot)
        self.c1 = c1

    def compute_transport_plan(self, labels):
        """Compute transportation plan from labels

        Args:
            labels (np.ndarray): array of labels (length: n_samples)
        
        Returns:
            np.ndarray: transportation plan (shape: n_samples x n_classes)
        """
        ot = labels_to_transport_plan(labels)
        return ot[:, ~np.all(ot == 0, axis=0)]

    def compute_c2(self, c2, ot):
        """Adapt target matrix size to ot size by removing empty classes if needed
        
        Args:
            c2 (np.ndarray): target matrix of shape (k, k)
            ot (np.ndarray): transport plan of shape (n, k)
        
        Returns:
            np.ndarray: adapted target matrix of shape (k', k')
        """
        diff = c2.shape[1] - ot.shape[1]
        if diff > 0:
            for _ in range(diff):
                c2 = np.delete(np.delete(c2, 0, axis=0), 0, axis=1)
        return c2

    def compute_fgw_loss(self, c1, m_ab):
        """
        Compute FGW loss for given source matrices and source attributes
        
        Args:
            c1 (np.ndarray): source structural distance matrix (shape: n, n)
            m_ab (np.ndarray): source attribute distance matrix (shape: n, n)
        """
        return fgw_loss(c1, self.c2, self.ot, m_ab)

    def get_df_shape(self, method_name, true_c2):
        """
        Get evaluation dataframe for non-attributed graph
        
        Args:
            method_name (str): name of the method
            true_c2 (np.ndarray): true target matrix (shape: k, k)
        
        Returns:
            pd.DataFrame: dataframe with loss metrics
        """
        true_c2 = self.compute_c2(true_c2, self.ot)
        results = pd.DataFrame(
            {
                "method": method_name,
                "kmeans_criteria": [self.kmeans_criteria],
                "gw_loss_isolated": [self.gw_loss_isolated],
                "gw_loss_true": gw_loss(self.c1, true_c2, self.ot),
            }
        )
        return results

    def get_df_attributed(self, method_name, m_ab):
        """
        Get evaluation dataframe for attributed graph
        
        Args:
            method_name (str): name of the method
            m_ab (np.ndarray): source attribute distance matrix (shape: n, n)
        
        Returns:
            pd.DataFrame: dataframe with loss metrics
        """
        results = pd.DataFrame(
            {
                "method": method_name,
                "kmeans_criteria": [self.kmeans_criteria],
                "gw_loss_isolated": [self.gw_loss_isolated],
                "fgw_loss": self.compute_fgw_loss(self.c1, m_ab),
            }
        )
        return results


class InternalEvaluation:
    """Internal metrics: modularity and silhouette (precomputed distance matrix).
    
    Args:
        graph (nx.Graph): input graph
        predicted_labels (list(int)): predicted labels for each node
        distance_matrix (np.ndarray): precomputed distance matrix (shape: n_samples x n_samples)
    """

    def __init__(self, graph, predicted_labels, distance_matrix):
        self.predicted_labels = predicted_labels
        self.partition = (
            prediction_to_partition(predicted_labels)
            if hasattr(predicted_labels, "max")
            else []
        )
        self.modularity = self.get_modularity(graph)
        self.silhouette = self.get_silhouette(distance_matrix)

    def get_modularity(self, graph):
        """Compute modularity of the partition on the graph."""
        if graph is None:
            return None
        return nx.algorithms.community.modularity(graph, self.partition)

    def get_silhouette(self, distance_matrix):
        """Compute silhouette score from precomputed distance matrix."""
        if distance_matrix is None:
            return None
        return np.mean(
            silhouette_samples(
                distance_matrix, self.predicted_labels, metric="precomputed"
            )
        )

    def get_df(self, method_name):
        """Get evaluation dataframe for internal metrics"""
        results = pd.DataFrame(
            {
                "method": method_name,
                "modularity": [self.modularity],
                "silhouette": [self.silhouette],
            }
        )
        return results


class ExternalEvaluation:
    """External metric: ARI
    
    Args:
        true_labels (list(int)): ground truth labels
        predicted_labels (list(int)): predicted labels
    
    Returns:
        pd.DataFrame: dataframe with ARI metric
    """

    def __init__(self, true_labels, predicted_labels):
        self.ari = adjusted_rand_score(true_labels, predicted_labels)
        self.k = len(set(predicted_labels))

    def get_df(self, method_name):
        """Get evaluation dataframe for ARI metric"""
        results = pd.DataFrame(
            {"method": method_name, "ARI": [self.ari], "k": [self.k]}
        )
        return results
