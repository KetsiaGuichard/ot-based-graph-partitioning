"""
Simulation utilities for constructing group counts and proportions.
Provides functions to generate proportions and convert them to counts per group.
"""

from random import randint
import numpy as np
from scipy.stats import dirichlet


def generate_proportions(alpha1_range, alpha2_range, step):
    """
    Generate all combinations of proportions (alpha1, alpha2, alpha3)
    such that alpha1 + alpha2 + alpha3 = 1 and
    alpha1 in alpha1_range, alpha2 in alpha2_range, alpha3 >= alpha2
    
    Args:
        alpha1_range (tuple): (min, max) range for alpha1
        alpha2_range (tuple): (min, max) range for alpha2
        step (float): step size for generating values within the ranges
    
    Returns:
        list of list: list of [alpha1, alpha2, alpha3] combinations"""
    alpha1_values = np.arange(alpha1_range[0], alpha1_range[1], step)
    alpha1_values = np.arange(alpha1_range[0], alpha1_range[1] + step / 2, step)
    combinations = []
    for a1 in alpha1_values:
        alpha2_values = np.arange(a1, alpha2_range[1], step)
        for a2 in alpha2_values:
            a3 = 1 - a1 - a2
            if a3 >= a2:
                combinations.append([round(a1, 3), round(a2, 3), round(a3, 3)])
    return combinations


def prop_to_ct(nodes_ct, prop_group):
    """Convert proportions to counts per group

    Args:
        nodes_ct (int): number of nodes
        prop_group (list(float)): list of proportions per group
    
    Returns:
        list(int): Number of nodes per group
    """
    k = len(prop_group)
    prop_group = np.array(prop_group)
    ct = np.round(prop_group * nodes_ct).astype(int)
    while sum(ct) < nodes_ct:
        ct[randint(0, k - 1)] += 1
    while sum(ct) > nodes_ct:
        ct[randint(0, k - 1)] -= 1
    return ct


def ct_group_alpha(
    nodes_ct, alpha1_range=(0.1, 1 / 3), alpha2_range=(0.1, 1 / 3), step=0.1
):
    """Get different number of nodes per group
    (How to divise n in 3 groups according to different proportions)
    
    Args:
        nodes_ct (int): number of nodes
        alpha1_range (tuple, optional): range for proportion of group 1. Defaults to (0.1, 1/3).
        alpha2_range (tuple, optional): range for proportion of group 2. Defaults to (0.1, 1/3).
        step (float, optional): step size for generating proportions. Defaults to 0.1.
    
    Returns:
        list(list(int)): List of different numbers of nodes per group"""
    repartition_list = generate_proportions(alpha1_range, alpha2_range, step)
    return [prop_to_ct(nodes_ct, repartition) for repartition in repartition_list]


def ct_group(nodes_ct, k, equilibre=10):
    """Get number of nodes per group
    (How to divise n in k groups)

    Args:
        nodes_ct (int): number of nodes
        k (int): number of groups
        equilibre (int, optional): the greater, the less pronounced the imbalance

    Returns:
        list(int): Number of nodes per group
    """
    prop_group = dirichlet.rvs([equilibre] * k, size=1)[0]
    ct = prop_to_ct(nodes_ct, prop_group)
    return ct


def ct_group_alt(groups, alterate=None):
    """
    Divisive = One SBM classes shares different attributes.
    Aggregative = Two SBM classes share same attributes.
    """
    if alterate == "divisive":
        idx = np.random.randint(len(groups))
        return np.concatenate(
            (groups[:idx], ct_group(groups[idx], 2), groups[idx + 1 :])
        )
    if alterate == "aggregative":
        return np.concatenate(([np.sum(groups[:2])], groups[2:]))
    return groups
