"""Distance computation utilities for graphs, time series and histograms.
Includes DTW, Wasserstein and graph shortest-path distance helpers.
"""

import itertools
import numpy as np
import networkx as nx
from dtw import dtw
from dtaidistance.dtw import distance_matrix_fast, distance_fast
from scipy.stats import wasserstein_distance

import matplotlib.pyplot as plt
import seaborn as sns


def distance_hausdorff_average(d):
    """Compute one of hausdorff distance variants

    Args:
        d (np.array): distance matrix

    Return:
        float: 0.5*(max_min_a_b + max_min_b_a)
    """
    max_min_a_b = d.min(axis=1).max()
    max_min_b_a = d.min(axis=0).max()
    return 0.5 * (max_min_a_b + max_min_b_a)


def distance_ensemble(ensemble, distance_function, normalize=True):
    """Compute distance between ensembles
    Distance function must be pair-wise.

    Args:
        ensemble (list): list of elements
        distance_function (function): function of distance between elements
        normalize (bool, optional): should the distance matrix be normalized? Default to True
    """
    n = len(ensemble)
    distances = np.zeros([n, n])
    for i in range(n):
        for j in range(i, n):
            d = distance_function(ensemble[i], ensemble[j])
            distances[i, j] = d
            distances[j, i] = d
    if normalize:
        distances = distances / distances.max()
    return distances


def distance_hausdorff_dtw_pairwise(list_a, list_b, window_size=0.05):
    """Compute Hausdorff distance with DTW distances between pairs of lists

    Args:
        list_a (list): list of elements for A
        list_b (list): list of elements for B
        function_array (np.array): array with time series or functional values
        window_size (float): proportion of window for DTW

    Returns:
        np.array(n_series, n_series): (normalized) DTW matrix
    """
    len_a = len(list_a)
    len_b = len(list_b)
    length_out = list_a[0].shape[0]
    distances = np.zeros([len_a, len_b])
    for a in range(len_a):
        for b in range(len_b):
            distances[a, b] = distance_fast(
                list_a[a], list_b[b], window=int(window_size * length_out)
            )
    return distance_hausdorff_average(distances)


def distance_graph(graph, weight=None, normalized=True):
    """Compute distance (shortest path) of a graph
    Result is proposed via pandas df and array matrix

    Args:
        graph (networkX Object): graph
        normalized (boolean, optionnal): should the distance matrix be normalized? Default to True

    Returns:
        np.array(n_nodes, n_nodes): (normalized) shortest distance matrix
    """
    shortest_path = nx.floyd_warshall_numpy(graph, weight=weight)
    if normalized:
        shortest_path = (shortest_path - shortest_path.min()) / (
            shortest_path.max() - shortest_path.min()
        )
    return shortest_path


def distance_functions_dtw(function_array, window_size=0.05, normalized=True):
    """Compute distance (DTW with window size) of functional data
    Also work for time series

    Args:
        function_array (np.array): array with time series or functional values
        window_size (float): proportion of window for DTW
        normalized (boolean, optionnal): should the distance matrix be normalized? Default to True

    Returns:
        np.array(n_series, n_series): (normalized) DTW matrix
    """
    n = function_array.shape[0]
    length_out = function_array.shape[1]
    dtw_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distance = dtw(
                function_array[i],
                function_array[j],
                window_type="sakoechiba",
                window_args={"window_size": int(window_size * length_out)},
            ).distance
            dtw_matrix[i, j] = distance
    if normalized:
        dtw_matrix = (dtw_matrix - dtw_matrix.min()) / (
            dtw_matrix.max() - dtw_matrix.min()
        )
    return dtw_matrix


def distance_functions_dtwai(function_array, window_size=0.05, normalized=True):
    """Compute distance (DTW with window size) of functional data
    Also work for time series

    Args:
        function_array (np.array): array with time series or functional values
        window_size (float): proportion of window for DTW
        normalized (boolean, optionnal): should the distance matrix be normalized? Default to True

    Returns:
        np.array(n_series, n_series): (normalized) DTW matrix
    """
    n = function_array.shape[0]
    length_out = function_array.shape[1]
    dtw_matrix = distance_matrix_fast(
        function_array, window=int(window_size * length_out)
    )
    if n > 1 and normalized:
        dtw_matrix = (dtw_matrix - dtw_matrix.min()) / (
            dtw_matrix.max() - dtw_matrix.min()
        )
    return dtw_matrix


def distance_functions_dtwai_barycenters(
    att_array, barycenters, window_size=0.05, normalized=True
):
    """Compute distance (DTW with window size) of functional data
    Also work for time series

    Args:
        att_array (np.array): array with time series or functional values
        window_size (float): proportion of window for DTW
        normalized (boolean, optionnal): should the distance matrix be normalized? Default to True

    Returns:
        np.array(n_series, n_series): (normalized) DTW matrix
    """
    n = att_array.shape[0]
    k = barycenters.shape[0]
    length_out = att_array.shape[1]

    dtw_matrix = np.zeros((n, k))

    for i in range(n):
        for j in range(k):
            dtw_matrix[i, j] = distance_fast(
                att_array[i], barycenters[j], window=int(window_size * length_out)
            )
    if n > 1 and k > 0 and normalized:
        dtw_matrix = (dtw_matrix - dtw_matrix.min()) / (
            dtw_matrix.max() - dtw_matrix.min()
        )
    return dtw_matrix


def distance_histograms(hist_array, normalized=True):
    """Compute Wasserstein Distance between histograms

    Args:
        hist_array (np.array(n, support)): Array of histograms
        normalized (boolean, optionnal): should the distance matrix be normalized? Default to True

    Returns:
        np.array: matrix of distance
    """
    n = hist_array.shape[0]
    support_len = hist_array.shape[1]
    hist_distance = np.zeros((n, n))
    x = np.arange(support_len)

    for i in range(n):
        for j in range(n):
            hist_distance[i, j] = wasserstein_distance(
                x, x, hist_array[i], hist_array[j]
            )
    if n > 1 and normalized:
        hist_distance = (hist_distance - hist_distance.min()) / (
            hist_distance.max() - hist_distance.min()
        )
    return hist_distance


def distance_histograms_fast(hist_array, normalized=True):
    """Compute Wasserstein Distance between histograms (fast version)

    Args:
        hist_array (np.array(n, support)): Array of histograms
        normalized (boolean, optionnal): should the distance matrix be normalized? Default to True

    Returns:
        np.array: matrix of distance
    """
    n = hist_array.shape[0]
    hist_distance = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            d = np.abs(np.cumsum(hist_array[i]) - np.cumsum(hist_array[j])).sum()
            hist_distance[i, j] = d
            hist_distance[j, i] = d

    if normalized:
        hist_distance = (hist_distance - hist_distance.min()) / (
            hist_distance.max() - hist_distance.min()
        )
    return hist_distance


def distance_hausdorff_histograms_pairwise(list_a, list_b):
    """Compute Hausdorff distance with Wasserstein distances between pairs of lists of histograms

    Args:
        list_a (list): list of elements for A
        list_b (list): list of elements for B
        normalized (boolean, optionnal): should the distance matrix be normalized? Default to True

    Returns:
        np.array(n_series, n_series): (normalized) Hausdorff distance matrix
    """
    len_a = len(list_a)
    len_b = len(list_b)
    distances = np.zeros([len_a, len_b])
    for a in range(len_a):
        for b in range(a, len_b):
            d = np.abs(np.cumsum(list_a[a]) - np.cumsum(list_b[b])).sum()
            distances[a, b] = d
            distances[b, a] = d
    return distance_hausdorff_average(distances)


def distance_histograms_fast_barycenters(att_array, barycenters, normalized=True):
    """Compute Wasserstein Distance between histograms (fast version)

    Args:
        att_array (np.array(n, support)): Array of histograms
        barycenters (np.array(k, support)): Array of barycenters
        normalized (boolean, optionnal): should the distance matrix be normalized? Default to True

    Returns:
        np.array: matrix of distance
    """
    n = att_array.shape[0]
    k = barycenters.shape[0]
    hist_distance = np.zeros((n, k))

    for i in range(n):
        for j in range(k):
            d = np.abs(np.cumsum(att_array[i]) - np.cumsum(barycenters[j])).sum()
            hist_distance[i, j] = d

    if normalized:
        hist_distance = (hist_distance - hist_distance.min()) / (
            hist_distance.max() - hist_distance.min()
        )
    return hist_distance


def distance_sum(distance_structural, distance_fun, distance_hist, alpha=0.5):
    """Compute the weighted sum of three distance matrices"""
    iter_matrices = itertools.product(distance_structural, distance_hist, distance_fun)
    distance_total = [
        alpha * matrices[0] + (1 - alpha) * ((matrices[1] + matrices[2]) / 2)
        for matrices in iter_matrices
    ]
    for i in range(len(distance_total)):
        distance_total[i] = distance_total[i] / distance_total[i].max()
    return distance_total


def distance_sum_attributes(distance_structural, distance_attributes, alpha=0.5):
    """Compute the weighted sum of two distance matrices"""
    iter_matrices = itertools.product(distance_structural, distance_attributes)
    distance_total = [
        alpha * matrices[0] + (1 - alpha) * matrices[1] for matrices in iter_matrices
    ]
    for i in range(len(distance_total)):
        distance_total[i] = distance_total[i] / distance_total[i].max()
    return distance_total


def combine_alpha(distance_structural, distance_attributes, alpha):
    """Compute the weighted sum of two distance matrices using an alpha matrix or scalar.

    Args:
        distance_structural (np.array (n, n)): Distance matrix computed from the graph structure.
        distance_attributes (np.array (n, n)): Distance matrix computed from node attributes.
        alpha (float or np.array (n, n)): Weighting coefficient(s).
            If a scalar, the same weight is applied to all elements.
            If an array, must have the same shape as the distance matrices.

    Returns:
        np.array (n, n): Combined distance matrix computed as:
            alpha * distance_structural + (1 - alpha) * distance_attributes
    """
    if np.isscalar(alpha):
        distance_alpha = alpha * distance_structural + (1 - alpha) * distance_attributes
    else:
        alpha = np.asarray(alpha)
        distance_alpha = (
            distance_structural * alpha[:, np.newaxis]
            + distance_attributes * (1 - alpha)[:, np.newaxis]
        )
    distance_alpha = distance_alpha / distance_alpha.max()
    return distance_alpha


def plot_distance_heatmap(distance_matrix, title="Distance Matrix", show=True, ax=None):
    """Plot Heatmap of a distance matrix

    Args:
        distance_matrix (np.array): A matrix of distances
        title (str, optionnal): title of the graph
    """
    sns.heatmap(
        distance_matrix,
        cmap="viridis",
        cbar_kws={"label": "Distance"},
        square=True,
        ax=ax,
    )
    plt.title(title)
    if show:
        plt.show()
