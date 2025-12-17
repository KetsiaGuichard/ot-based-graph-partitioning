"""
Clustering utility functions for graph partitioning and barycenter computation.

This module provides functions for k-means criteria, centroid and barycenter computation,
transport plan conversions, and metric aggregation for attributed graphs.
"""

import numpy as np


def compute_kmeans_criteria(distance_matrix, labels, centroids=None):
    """
    Compute the Frechet k-means criteria for a distance matrix, labels, and centroids.
    # TODO: Add weights
    Args:
        distance_matrix (np.ndarray): Matrix of distances (topological and attributes combined).
        labels (np.ndarray): Array of classified node labels.
        centroids (list[int], optional): List of indices of centroids/barycenters.
            If None, barycenters are computed.

    Returns:
        float: Inner squared distance between elements of classes and centroids/barycenters.
    """
    if centroids is None:
        centroids = compute_barycenters(distance_matrix, labels)
    kmeans_criteria = 0
    unique_labels = np.unique(labels)
    for idx_centroid, i in enumerate(unique_labels):
        idx = np.where(labels == i)[0]
        kmeans_criteria += np.sum(
            np.square(distance_matrix[idx, centroids[idx_centroid]])
        )
    return kmeans_criteria


def compute_centroids(distance_matrix, indices):
    """
    Compute the index of the centroid (min sum of distances) in a group.
    # TODO Add weights
    Args:
        distance_matrix (np.ndarray): Distance matrix.
        indices (list[int]): Indices of the group.

    Returns:
        int: Index of the centroid within the group.
    """
    data_group = distance_matrix[np.ix_(indices, indices)]
    return np.argmin(np.sum(data_group, axis=1))


def compute_cellules(distance_matrix, centroids):
    """
    Assign each node to the closest centroid.
    # TODO Add weights
    Args:
        distance_matrix (np.ndarray): Distance matrix.
        centroids (list[int]): Indices of centroids.

    Returns:
        np.ndarray: Array of assigned cluster labels.
    """
    return np.argmin(distance_matrix[centroids], axis=0)


def centroids_to_ot(distance_matrix, centroids):
    """
    Compute transport plan from centroids by assigning each node to the closest centroid.
    
    Args:
        distance_matrix (np.ndarray): Distance matrix.
        centroids (list[int]): Indices of centroids.
        
    Returns:
        np.ndarray: Transport plan matrix.
    """
    labels = compute_cellules(distance_matrix, centroids)
    return labels_to_transport_plan(labels)


def compute_barycenters(distance_matrix, labels):
    """
    Get Frechet barycenters of clusters.

    Args:
        distance_matrix (np.ndarray): Distance matrix.
        labels (np.ndarray): Array of cluster labels.

    Returns:
        list[int]: Indices of barycenters for each class.
    """
    barycenters = []
    for i in set(labels):
        idx = np.where(labels == i)[0]
        idx_barycenter = compute_centroids(distance_matrix**2, idx)
        barycenters.append(idx[idx_barycenter])
    return barycenters


def labels_to_transport_plan(labels):
    """
    Convert cluster labels to a hard transport plan matrix (one-hot encoding).

    Args:
        labels (np.ndarray): Array of cluster labels.

    Returns:
        np.ndarray: Transport plan matrix (hard clustering).
    """
    one_hot_matrix = np.eye(np.max(labels) + 1)[labels]
    return one_hot_matrix / one_hot_matrix.sum()


def transport_plan_to_labels(transport_plan):
    """
    Convert a transport plan matrix to hard cluster labels.

    Args:
        transport_plan (np.ndarray): Transport plan matrix.

    Returns:
        np.ndarray: Array of cluster labels.
    """
    return np.argmax(transport_plan, axis=1)


def transport_plan_to_soft(transport_plan):
    """
    Convert a transport plan matrix to a soft clustering dictionary.

    Args:
        transport_plan (np.ndarray): Transport plan matrix.

    Returns:
        dict: Soft clustering as {cluster_id: {row_idx: weight, ...}, ...}
    """
    rows, cols = np.nonzero(transport_plan)
    values = transport_plan[rows, cols]
    soft_clustering = {}
    for i, k, w in zip(rows, cols, values):
        if k not in soft_clustering:
            soft_clustering[k] = {}
        soft_clustering[k][int(i)] = float(w)
    return soft_clustering


def transport_plan_to_hard(transport_plan):
    """
    Convert a soft transport plan to a hard transport plan (one-hot per row).

    Args:
        transport_plan (np.ndarray): Transport plan matrix.

    Returns:
        np.ndarray: Hard transport plan matrix.
    """
    t_hard = np.zeros_like(transport_plan, dtype=int)
    for i, row in enumerate(transport_plan):
        j = np.argmax(row)
        t_hard[i, j] = 1
    return t_hard


def barycenter_medoid(distance_matrix):
    """
    Compute the medoid index (min sum of distances) in a group.

    Args:
        distance_matrix (np.ndarray): Distance matrix within a group.

    Returns:
        int: Index of the medoid.
    """
    return distance_matrix.sum(axis=1).argmin()


def final_m(m_dict, weights, powers):
    """
    Weighted sum of metric matrices with powers.

    Args:
        m_dict (dict): Dictionary of metric matrices (one per attribute).
        weights (dict): Weights for each attribute.
        powers (dict): Power for each attribute's metric matrix.

    Returns:
        np.ndarray: Aggregated metric matrix.
    """
    total = sum(weights.values())
    if total != 1:
        weights = {k: v / total for k, v in weights.items()}
    result = sum(weights[idx] * m_dict[idx] ** powers[idx] for idx in m_dict)
    return result


def compute_barycenters_ot_medoid(transport_plan, distance_matrices):
    """
    Compute barycenters (medoids) for each attribute using a transport plan.
    Attributes can be of different types.
    Their name should be consistent with tasks names.

    Args:
        transport_plan (np.ndarray): Transport plan matrix.
        distance_matrices (dict): Dictionary of distance matrices per attribute.

    Returns:
        dict: Barycenter indices per attribute.
    """
    barycenters = {}
    labels = transport_plan_to_labels(transport_plan)
    for attribute_name in distance_matrices.keys():
        barycenter_attribute = []
        for i in set(labels):
            idx_group = np.where(labels == i)[0]
            distance_intra = distance_matrices[attribute_name][
                np.ix_(idx_group, idx_group)
            ]
            id_min = barycenter_medoid(distance_intra)
            barycenter_attribute.append(idx_group[id_min])
        barycenters[attribute_name] = barycenter_attribute
    return barycenters


def compute_barycenters_ot_medoid_soft(transport_plan, distance_matrices):
    """
    Compute barycenters (medoids) for each attribute using a soft transport plan.
    Attributes can be of different types.
    Their name should be consistent with tasks names.
    Barycenters choice should be consistent with defined functions.
    Barycenters packages must be previously loaded.

    Args:
        transport_plan (np.ndarray): Transport plan matrix.
        distance_matrices (dict): Dictionary of distance matrices per attribute.

    Returns:
        dict: Barycenter indices per attribute.
    """
    barycenters = {}
    soft_clusters = transport_plan_to_soft(transport_plan)
    for attribute_name, d_matrix in distance_matrices.items():
        barycenter_attribute = []
        for _, cluster in soft_clusters.items():
            idx_group = np.array(list(cluster.keys()))
            weights = np.array(list(cluster.values()))
            if len(idx_group) == 0:
                barycenter_attribute.append(None)
                continue
            distance_intra = d_matrix[np.ix_(idx_group, idx_group)]
            costs = np.dot(distance_intra, weights)
            id_min = np.argmin(costs)
            barycenter_attribute.append(idx_group[id_min])
        barycenters[attribute_name] = barycenter_attribute
    return barycenters


def compute_barycenters_ot_generic(transport_plan, attributes, tasks_barycenters):
    """
    Compute barycenters using custom functions for each attribute (hard clustering).
    Attributes can be of different types.
    Their name should be consistent with tasks names.
    Barycenters choice should be consistent with defined functions.
    Barycenters packages must be previously loaded.

    Args:
        transport_plan (np.ndarray): Transport plan matrix.
        attributes (dict): Dictionary of attribute arrays.
        tasks_barycenters (dict): Format: attribute_name:
            {"function": func, "kwargs": {}, "data_name": "s"}

    Returns:
        dict: Barycenters per attribute.
    """
    barycenters = {}
    labels = transport_plan_to_labels(transport_plan)
    for attribute_name in tasks_barycenters.keys():
        attributes_tmp = attributes[attribute_name]
        barycenter_attribute = []
        for i in set(labels):
            idx_group = np.where(labels == i)[0]
            bary_function = tasks_barycenters[attribute_name]
            kwargs = bary_function["kwargs"]
            cell_attribute = attributes_tmp[idx_group]
            data_name = bary_function["data_name"]
            kwargs[data_name] = cell_attribute
            barycenter_attribute.append(bary_function["function"](**kwargs))
        barycenters[attribute_name] = barycenter_attribute
    return barycenters


def compute_barycenters_ot_generic_soft(transport_plan, attributes, tasks_barycenters):
    """
    Compute barycenters using custom functions for each attribute (soft clustering).
    Attributes can be of different types.
    Their name should be consistent with tasks names.
    Barycenters choice should be consistent with defined functions.
    Barycenters packages must be previously loaded.

    Args:
        transport_plan (np.ndarray): Transport plan matrix.
        attributes (dict): Dictionary of attribute arrays.
        tasks_barycenters (dict): Format: attribute_name:
            {"function": func, "kwargs": {}, "data_name": "s"}

    Returns:
        dict: Barycenters per attribute.
    """
    barycenters = {}
    soft_clusters = transport_plan_to_soft(transport_plan)
    for attribute_name, bary_function in tasks_barycenters.items():
        attributes_tmp = attributes[attribute_name]
        barycenter_attribute = []
        for _, cluster in soft_clusters.items():
            idx_group = np.array(list(cluster.keys()))
            weights = np.array(list(cluster.values()))
            if len(idx_group) == 0:
                barycenter_attribute.append(None)
                continue
            cell_attribute = attributes_tmp[idx_group]
            kwargs = dict(bary_function["kwargs"])
            data_name = bary_function["data_name"]
            kwargs[data_name] = cell_attribute
            kwargs["weights"] = weights
            barycenter_attribute.append(bary_function["function"](**kwargs))
        barycenters[attribute_name] = barycenter_attribute
    return barycenters


def compute_m_generic(transport_plan, attributes, config):
    """
    Compute the aggregated metric matrix between barycenters and attributes (generic version).
    Attributes can be of different types.
    Their name should be consistent with tasks names.
    Barycenters choice should be consistent with defined functions.
    Barycenters packages must be previously loaded.

    Args:
        transport_plan (np.ndarray): Transport plan matrix.
        attributes (dict): Dictionary of attribute arrays.
        config (dict): Dictionary containing keys:
            - tasks_distances (dict): Format: attribute_name: {"function": func, "kwargs": {}}
            - tasks_barycenters (dict): Format: attribute_name:
                {"function": func, "kwargs": {}, "data_name": "s"}
            - weights (dict): Weights for each attribute (should sum to 1).
            - powers (dict): Power for each attribute's metric matrix.

    Returns:
        np.ndarray: Aggregated metric matrix.
    """
    tasks_distances = config["tasks_distances"]
    tasks_barycenters = config["tasks_barycenters"]
    weights = config["weights"]
    powers = config["powers"]
    barycenters = compute_barycenters_ot_generic_soft(
        transport_plan, attributes, tasks_barycenters
    )
    m_dict = {}
    for attribute_name in tasks_distances.keys():
        distance_function = tasks_distances[attribute_name]
        kwargs = distance_function["kwargs"]
        kwargs["barycenters"] = np.array(barycenters[attribute_name])
        kwargs["att_array"] = attributes[attribute_name]
        m_dict[attribute_name] = distance_function["function"](**kwargs)
    m = final_m(m_dict, weights, powers)
    return m


def compute_m_medoid(transport_plan, distance_matrices, weights, powers):
    """
    Compute the aggregated metric matrix between barycenters and attributes (medoid version).
    Attributes can be of different types. Their name should be consistent with tasks names.
    Barycenters choice should be consistent with distance functions.
        Packages must be previously loaded.

    Args:
        transport_plan (np.ndarray): Transport plan matrix.
        distance_matrices (dict): Dictionary of distance matrices per attribute.
        weights (dict): Weights for each attribute (should sum to 1).
        powers (dict): Power for each attribute's metric matrix.

    Returns:
        np.ndarray: Aggregated metric matrix.
    """
    barycenters = compute_barycenters_ot_medoid_soft(transport_plan, distance_matrices)
    m_dict = {}
    for attribute_name in distance_matrices.keys():
        distance_matrix = distance_matrices[attribute_name]
        idx = barycenters[attribute_name]
        m_dict[attribute_name] = distance_matrix[:, idx]
    m = final_m(m_dict, weights, powers)
    return m
