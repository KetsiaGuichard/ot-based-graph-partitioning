"""Evaluation utility helpers for experiments.
Includes parameter grids, partition conversions, and loss/target helpers.
"""

import itertools
import numpy as np
import networkx as nx
import pandas as pd


def parameters_to_df(parameters: dict) -> pd.DataFrame:
    """Transform parameters into a dataframe.

    Given a parameters mapping containing keys 'structure_force', 'c' and
    'epsilon', return the product of these parameters as a dataframe with
    columns ['structure_force', 'c', 'epsilon'].
    """
    product_parameters = itertools.product(
        parameters["structure_force"], parameters["c"], parameters["epsilon"]
    )
    return pd.DataFrame(
        list(product_parameters), columns=["structure_force", "c", "epsilon"]
    )


def prediction_to_partition(labels: np.ndarray) -> list:
    """Transform a label vector into a partition (list of index lists).

    Example: labels = [0,1,0,2] -> [[0,2],[1],[3]]
    """
    classes_ct = range(int(labels.max()) + 1)
    partition = [np.where(labels == cluster)[0].tolist() for cluster in classes_ct]
    return partition


def eval_df(
    evaluation_table: pd.DataFrame, parameters: pd.DataFrame, infos: dict
) -> pd.DataFrame:
    """Concatenate parameter/info columns to an evaluation table.

    Returns a new dataframe containing parameters (repeated), the evaluation
    table and additional infos (repeated) side-by-side.
    """
    tmp_parameters = pd.concat([parameters] * len(evaluation_table), ignore_index=True)
    tmp_infos = pd.DataFrame([infos] * len(evaluation_table))
    evaluation_table = pd.concat(
        [
            tmp_parameters,
            evaluation_table.reset_index(drop=True),
            tmp_infos.reset_index(drop=True),
        ],
        axis=1,
    )
    return evaluation_table


def full_matrix(matrix, indices, size):
    """
    Complete a nxk matrix to a nxk' matrix by adding columns containing zeros

    Args:
        matrix (np.array(n,k)): matrix to be completed
        indices (list of size k): at which columns the current matrix should be?
        size (int): future size of the matrix (k')
    """
    n = matrix.shape[0]
    new_matrix = np.zeros((n, size))
    new_matrix[:, indices] = matrix
    return new_matrix


def distance_mixing(sbm_matrix):
    """
    Compute the shortest path distance matrix from a mixing matrix
    The mixing matrix is transformed in a graph, where the distance on the edges
    equals 1/probability that an edge exists (if the probability is large, the distance is short)
    This distance matrix will be used as a "true" c2 for srGW.

    Args:
        sbm_matrix (np.array(n,k)): matrix to be completed
    """
    mask = sbm_matrix != 0
    inv_matrix = np.zeros_like(sbm_matrix)
    inv_matrix[mask] = 1 / sbm_matrix[mask]
    graph = nx.from_numpy_array(inv_matrix)
    c2 = nx.floyd_warshall_numpy(graph)
    np.fill_diagonal(c2, 0)
    return c2 / c2.max()


def true_c(distance_matrix, labels):
    """Compute "true" target matrix from distance matrix and labels
    
    Args:
        distance_matrix (np.ndarray): precomputed distance matrix (shape: n_samples x n_samples)
        labels (list(int)): ground truth labels
    
    Returns:
        np.ndarray: kxk matrix with average distances between classes
    """
    labels = np.array(labels)
    unique_labels = np.unique(labels)
    k = len(unique_labels)
    k_distances = np.zeros((k, k))

    for i, label_i in enumerate(unique_labels):
        indices_i = np.where(labels == label_i)[0]
        for j, label_j in enumerate(unique_labels):
            indices_j = np.where(labels == label_j)[0]
            submatrix = distance_matrix[np.ix_(indices_i, indices_j)]
            if i == j:
                if len(indices_i) > 1:
                    mask = ~np.eye(len(indices_i), dtype=bool)
                    mean_distance = submatrix[mask].mean()
                else:
                    mean_distance = 0.0
            else:
                mean_distance = submatrix.mean()
            k_distances[i, j] = mean_distance

    return k_distances


def target_c(k, value=1):
    """Create target structure
    This target is a kxk matrix with 1, except on the diagonal

    Args:
        k (int): number of class

    Returns:
        array(k,k): matrix with 1, except on the diagonal (0)
    """
    target = np.full((k, k), value) - value * np.identity(k)
    return target


def gw_loss(c1, c2, ot):
    """Compute GW loss for a given transportation plan (squared loss)
    Args:
        c1 (np.array(n,n)): Metric cost matrix in the source space
        c2 (np.array(k,k)): Metric cost matrix in the target space
        ot (np.array(n,k)): Optimal transportation matrix.

    Returns:
        float: GW loss
    """
    n = c1.shape[0]
    k = c2.shape[0]
    h1 = ot.sum(axis=1)
    h2 = ot.sum(axis=0)
    c = np.dot(np.dot(c1**2, np.reshape(h1, (-1, 1))), np.ones((1, k))) + np.dot(
        np.ones((n, 1)), np.dot(np.reshape(h2, (-1, 1)).T, (c2**2))
    )
    hc1 = c1
    hc2 = 2 * c2
    return np.sum((c - np.dot(np.dot(hc1, ot), hc2)) * ot)


def fgw_loss(c1, c2, ot, m_ab, alpha=0.5):
    """
    Compute FGW loss for a given transportation plan (squared loss)

    Args:
        c1 (np.array(n,n)): Structural distance matrix in the source space
        c2 (np.array(k,k)): Structural distance matrix in the target space
        ot (np.array(n,k)): Optimal transportation matrix.
        m_ab (np.array(n,k)): Attributes distance matrix between features across domains
        alpha (float, optional): Default to 0.5. Tradeoff between attributes and structure.

        Returns:
            float: FGW loss
    """
    gwl = gw_loss(c1, c2, ot)
    fl = np.sum(m_ab * ot)
    return alpha * gwl + (1 - alpha) * fl
