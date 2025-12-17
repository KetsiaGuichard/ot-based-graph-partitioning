"""
Partitioning attributed graphs using Gromov-Wasserstein (GW) methods.
Provides method both for semi-relaxed GW and fused GW clustering,
including barycenter computation, transport plan initialization, 
and iterative optimization.
"""

from abc import ABC, abstractmethod
import warnings
from ot import gromov
import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist

from src.clustering.utils import (
    compute_barycenters,
    compute_m_generic,
    compute_m_medoid,
    transport_plan_to_hard,
    transport_plan_to_labels,
)
from src.distances import combine_alpha


class GWClustering(ABC):
    """
    Abstract base class for partitioning attributed graphs with GW-type methods.

    Args:
        c1 (np.ndarray): Matrix of structure, e.g., adjacency matrix.
        k (int): Number of clusters/classes.
        weights (list[float], optional): Weights of each node, default to uniform.
        g0 (np.ndarray or str, optional): Initial transport plan or initialization method.
    """

    def __init__(self, n, k: int, weights=None, g0="kmeans"):
        """
        Initialize GWClustering with structure matrix,
        number of clusters, weights, and initial transport plan.

        Args:
            c1 (np.ndarray): Structure matrix.
            k (int): Number of clusters.
            weights (list[float] or None): Node weights.
            g0 (np.ndarray or str): Initial transport plan or method name.
        """
        self.n = n
        self.k = k
        self.weights = self.compute_weights(weights)
        self.c2 = self.identity_graph(k)
        self.g0 = self.compute_g0(g0)

    def identity_graph(self, k):
        """
        Create identity matrix of size k for identification graph.

        Args:
            k (int): Number of clusters/classes.

        Returns:
            np.ndarray: Identity matrix of shape (k, k).
        """
        return np.identity(k)

    def target_c(self, k: int, value: float = 1.0) -> np.ndarray:
        """
        Create target structure: kxk matrix with same value except on the diagonal (0).
        This structural matrix represents equidistant nodes.

        Args:
            k (int): Number of clusters/classes.
            value (float): Replacement value for +inf entries in the result.

        Returns:
            np.ndarray: Target structure matrix (shape (k,k)).
        """
        target_graph = nx.from_numpy_array(self.identity_graph(k))
        target_c = nx.floyd_warshall_numpy(target_graph)
        target_c = np.asarray(target_c, dtype=float)
        target_c[~np.isfinite(target_c)] = value
        return target_c

    def compute_g0(self, g0):
        """
        Validate and return an initial transport plan or a valid strategy string.

        If `g0` is a 2D array it must have shape (n, k0) where n equals the
        number of source nodes (self.n). Strings accepted are POT init names.

        Args:
            g0 (np.ndarray or str): Initial transport plan or method name.

        Returns:
            np.ndarray or str: Initial transport plan or method name.
        """
        if isinstance(g0, np.ndarray):
            return g0
        if isinstance(g0, str):
            valid = [
                "product",
                "random_product",
                "random",
                "fluid",
                "spectral",
                "kmeans",
            ]
            if g0 in valid:
                return g0
        warnings.warn("g0 unknown or invalid, defaulting to 'kmeans'")
        return "kmeans"

    def compute_weights(self, weights) -> np.ndarray:
        """
        Compute node weights. Defaults to uniform weights if not specified.

        Args:
            weights (None or array-like): If None, uniform weights are returned.

        Returns:
            np.ndarray: Normalized weights of shape (n,).
        """
        if weights is None:
            return np.ones(self.n, dtype=float) / float(self.n)
        arr = np.asarray(weights, dtype=float)
        if arr.ndim != 1:
            raise ValueError("weights must be a 1-D array-like of length n")
        if arr.shape[0] != self.n:
            raise ValueError(f"weights length {arr.shape[0]} does not match n ({self.n})")
        total = arr.sum()
        if total <= 0 or np.isnan(total):
            raise ValueError("weights must sum to a positive number")
        return arr / float(total)

    def update_ot(self, t):
        """
        Remove empty columns from transport plan matrix.

        Args:
            t (np.ndarray): Current transport plan matrix.

        Returns:
            np.ndarray: Transport plan matrix without empty columns.
        """
        idx = np.where(t.sum(axis=0) != 0)[0]
        return t[:, idx]

    def update_k(self, t):
        """
        Update number of clusters based on non-empty columns in transport plan.

        Args:
            t (np.ndarray): Current transport plan matrix.

        Returns:
            int: Number of non-empty clusters.
        """
        return len(np.where(t.sum(axis=0) != 0)[0])

    def compute_clusters(self, ot):
        """
        Get cluster assignments by taking argmax over transport plan.

        Returns:
            np.ndarray: Array of cluster indices for each node.
        """
        return ot.argmax(axis=1)

    @abstractmethod
    def partitioning(self):
        """
        Abstract method for partitioning the graph. Must be implemented in subclasses.
        """


class SemiRelaxedGWClustering(GWClustering):
    """
    Partition attributed graph with semi-relaxed Gromov-Wasserstein (GW).

    Args:
        distance_matrix (np.ndarray): Structure matrix (e.g., adjacency matrix).
        k (int): Number of clusters/classes.
        weights (list[float], optional): Node weights, default to uniform.
        g0 (np.ndarray or str, optional): Initial transport plan or method name.
        target_c (np.ndarray, optional): Target structure matrix.
    """

    def __init__(
        self,
        n: int,
        k: int,
        weights=None,
        g0="kmeans",
        target_c=None,
        value=1,
        max_iter=100,
    ):
        """
        Initialize SemiRelaxedGWClustering
        with structure matrix, clusters, weights, and initial transport plan.

        Args:
            distance_matrix (np.ndarray): Structure matrix.
            k (int): Number of clusters.
            weights (list[float] or None): Node weights.
            g0 (np.ndarray or str): Initial transport plan or method name.
            target_c (np.ndarray or None): Target structure matrix.
        """
        super().__init__(n, k, weights, g0)
        self.c2 = self.define_target_c(k, value, target_c)
        self.max_iter = max_iter
        self.value = value

    def define_target_c(self, k, value, target_c):
        """
        Define target structure matrix. If not provided, use default.

        Args:
            k (int): Number of clusters/classes.
            value (float): Replacement value for +inf entries.
            target_c (np.ndarray or None): Target structure matrix.

        Returns:
            np.ndarray: Target structure matrix.
        """
        if target_c is None:
            return self.target_c(k, value)
        return target_c

    def update_alpha(
        self,
        structural_matrix,
        attributes_matrix,
        k,
        ot,
    ):
        """
        Compute best global alpha (weighting parameter between structure and attributes)
        for a given transportat plan.

        Args:
            structural_matrix (np.ndarray): Structural distance matrix.
            attributes_matrix (np.ndarray): Attributes distance matrix.
            k (int): Number of clusters/classes.
            ot (np.ndarray): Transport plan (nxk)

        Returns:
            np.ndarray: Target structure matrix.
        """
        c2 = self.target_c(k, self.value)
        numerator = 0
        denominator = 0
        for i in range(self.n):
            for j in range(self.n):
                for l in range(k):
                    for m in range(k):
                        ds_ij = structural_matrix[i, j]
                        da_ij = attributes_matrix[i, j]
                        r_lm = c2[l, m]
                        s_ij = ds_ij - da_ij
                        r_ijlm = da_ij - r_lm
                        numerator += s_ij * r_ijlm * ot[i, l] * ot[j, m]
                denominator += s_ij**2 * self.weights[i] * self.weights[j]
        alpha = -numerator / denominator if denominator != 0 else 0.5
        return np.clip(alpha, 0, 1)

    def partitioning_simple(self, structural_matrix, embedded=False):
        """
        Partition using semi-relaxed GW.

        Args:
            g0 (np.ndarray or str): Initial transport plan or method name.

        Returns:
            np.ndarray: Transport plan matrix.
        """
        if embedded:
            structural_matrix = cdist(structural_matrix, structural_matrix)
            structural_matrix = structural_matrix / structural_matrix.max()
        ot_mat = gromov.semirelaxed_gromov_wasserstein(
            structural_matrix,
            self.c2,
            self.weights,
            symmetric=True,
            log=False,
            G0=self.g0,
        )
        labels = transport_plan_to_labels(ot_mat)
        centroids = compute_barycenters(structural_matrix, labels)
        return {"ot": ot_mat, "labels": labels, "centroids": centroids}

    def partitioning(
        self,
        structural_matrix,
        attributes_matrix,
        embedded=False,
        alpha=0.5,
    ):
        """
        Partition using semi-relaxed FGW.

        Args:
            structural_matrix (np.ndarray): Structure matrix.
            attributes_matrix (np.ndarray): Attributes matrix.
            alpha (bool | float): True if alpha optimized,
                float (between 0 and 1) otherwise.
                Default is non-alpha optimized version, with alpha = 0.5

        Returns:
            np.ndarray: Transport plan matrix.
        """
        default_alpha = [0.5 if isinstance(alpha, bool) and alpha is True else alpha]
        distance_matrix = combine_alpha(
            structural_matrix, attributes_matrix, default_alpha
        )
        if embedded:
            distance_matrix = cdist(distance_matrix, distance_matrix)
            distance_matrix = distance_matrix / distance_matrix.max()

        if isinstance(alpha, float):
            results = self.partitioning_simple(distance_matrix)
            results["alpha"] = alpha
            return results
        iter_ct = 0
        old_ot = None
        ot = gromov.semirelaxed_gromov_wasserstein(
            distance_matrix,
            self.c2,
            self.weights,
            symmetric=True,
            log=False,
            G0=self.g0,
        )
        while (iter_ct < self.max_iter) & (not np.array_equal(old_ot, ot)):
            ot = self.update_ot(ot)
            old_ot = ot
            k = self.update_k(ot)
            alpha = self.update_alpha(
                structural_matrix, attributes_matrix, k, ot
            )
            distance_matrix = combine_alpha(structural_matrix, attributes_matrix, alpha)
            if embedded:
                distance_matrix = cdist(distance_matrix, distance_matrix)
                distance_matrix = distance_matrix / distance_matrix.max()
            ot = gromov.semirelaxed_gromov_wasserstein(
                distance_matrix,
                self.target_c(k, self.value),
                self.weights,
                symmetric=True,
                log=False,
                G0=old_ot,
            )
            iter_ct += 1
        labels = transport_plan_to_labels(ot)
        centroids = compute_barycenters(structural_matrix, labels)
        return {
            "ot": ot,
            "labels": labels,
            "centroids": centroids,
            "iter_ct": iter_ct,
            "alpha": alpha,
        }


class SemiRelaxedFGWClustering(GWClustering):
    """
    Partition attributed graph with semi-relaxed Fused Gromov-Wasserstein (FGW).
    Adds heuristics for curves and histograms.

    Args:
        attributes (dict): Dictionary of attribute arrays (functions, histograms).
        c1 (np.ndarray): Structure matrix (e.g., shortest path).
        k (int): Number of clusters/classes.
        weights (list[float], optional): Node weights, default to uniform.
        g0 (np.ndarray or str, optional): Initial transport plan or method name.
    """

    def __init__(self, n, k: int, weights=None, g0="kmeans", g0_attributes=None):
        """
        Initialize SemiRelaxedFGWClustering
        with attributes, structure matrix, clusters, weights, and initial transport plan.

        Args:
            attributes (dict): Attribute arrays.
            c1 (np.ndarray): Structure matrix.
            k (int): Number of clusters.
            alpha (float): Proportion of structure strength.
            weights (list[float] or None): Node weights.
            g0 (np.ndarray or str): Initial transport plan or method name.
        """
        super().__init__(n, k, weights, g0)
        self.tasks_distances = {}
        self.tasks_barycenters = {}
        self.beta_weights = None
        self.beta_powers = None
        self.alpha = None
        self.g0_attributes = g0_attributes

    def define_method(
        self, weights, powers, tasks_distances=None, tasks_barycenters=None, alpha=None
    ):
        """
        Define method for barycenter and metric computation.
        If custom barycenters are used, tasks_distances and tasks_barycenters should be defined.
        If medoid, distances_matrices should be defined.

        Args:
            weights (dict): Weights of each attribute (must sum to 1).
            powers (dict): Power of each attribute distance matrix.
            tasks_distances (dict, optional): Needed if custom barycenters used.
            tasks_barycenters (dict, optional): Needed if custom barycenters used.
            distances_matrices (dict, optional): Needed for medoid.
        """
        self.tasks_distances = tasks_distances if tasks_distances is not None else {}
        self.tasks_barycenters = (
            tasks_barycenters if tasks_barycenters is not None else {}
        )
        self.beta_weights = weights
        self.beta_powers = powers
        self.alpha = alpha if alpha is not None else 0.5

    def compute_m(self, ot, distances_attributes, medoid):
        """
        Compute metric matrix M between barycenters and attributes.

        Args:
            ot (np.ndarray): Transport plan matrix.
            distances_attributes (dict): Distance matrices for each attribute.
            medoid (bool): Use medoid barycenters if True.

        Returns:
            np.ndarray: Metric matrix M.
        """
        if not medoid:
            m = compute_m_generic(
                ot,
                distances_attributes,
                {
                    "tasks_distances": self.tasks_distances,
                    "tasks_barycenters": self.tasks_barycenters,
                    "weights": self.beta_weights,
                    "powers": self.beta_powers,
                },
            )
        else:
            m = compute_m_medoid(
                ot,
                distances_attributes,
                self.beta_weights,
                self.beta_powers,
            )
        return m

    def partitioning(self, c1, c2, m, g0):
        """
        Partition using semi-relaxed FGW.

        Args:
            c1 (np.ndarray): Structural source matrix
            c2 (np.ndarray): Structural target matrix
            m (np.ndarray): Metric cost matrix between features across domains.
            g0 (np.ndarray or str): Initial transport plan or method name.

        Returns:
            np.ndarray: Transport plan matrix.
        """
        ot_mat, log = gromov.semirelaxed_fused_gromov_wasserstein(
            m,
            c1,
            c2,
            self.weights,
            alpha=self.alpha,
            symmetric=True,
            log=True,
            G0=g0,
        )
        srfgw_dist = log["srfgw_dist"]
        return ot_mat, srfgw_dist

    def iterate(
        self,
        structural_matrix,
        attributes_matrices_dict,
        value=1,
        iterations=10,
        medoid=False,
    ):
        """
        Iteratively compute optimal transport plan and barycenters.
        Recalculate barycenters and transport plan for a given number of iterations.

        Args:
            iterations (int): Number of iterations.
            medoid (bool): Use medoid barycenters if True.

        Returns:
            dict: Contains lists of transport plans ('ot'), k values ('k'), and metric matrices ('m').
        """
        ot_list = [self.g0]
        k_list = [self.k]
        m_list = []
        m = self.compute_m(self.g0_attributes, attributes_matrices_dict, medoid)
        c2 = self.target_c(self.k, value)
        ot, srfgw_dist = self.partitioning(structural_matrix, c2, m, self.g0)
        m_list.append(m)
        ot_list.append(ot)
        if self.alpha < 1:
            for _ in range(1, iterations):
                k = self.update_k(ot)
                ot = self.update_ot(ot)
                m = self.compute_m(ot, attributes_matrices_dict, medoid)
                c2 = self.target_c(k, value)
                ot, srfgw_dist = self.partitioning(structural_matrix, c2, m, ot)
                if np.array_equal(ot, ot_list[-1]):
                    break
                m_list.append(m)
                k_list.append(k)
                ot_list.append(ot)
        hard_t = transport_plan_to_hard(ot)
        ot_list.append(hard_t)
        m = self.compute_m(hard_t, attributes_matrices_dict, medoid)
        m_list.append(m)
        return {"ot": ot_list, "k": k_list, "m": m_list, "srfgw_dist": srfgw_dist}
