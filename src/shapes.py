"""Helpers to build and validate SBM mixing matrices for synthetic graphs.
Includes connectivity checks and several template mixing matrix shapes.
"""

import numpy as np


def normalize_symmetric(mixing_matrix):
    """Normalize and symmetrize a matrix"""
    normalized_mixing_matrix = mixing_matrix / mixing_matrix.sum(axis=0)
    symmetric_mixing_matrix = (
        normalized_mixing_matrix + normalized_mixing_matrix.T
    ) / 2
    return symmetric_mixing_matrix


def check_connectivity(probability_matrix):
    """Check if a probability matrix represents a related network
    The probability matrix could represent a related network, however,
    depending on the sizes of the classes and the probability in the
    matrix, the resulting SBM network could be non related.
    This function represents only an early check for this issue.

    Args:
        probability_matrix (matrix): probability matrix for SBM

    Returns
        bool: True if the network is related, false otherwise
    """
    adjacency = (probability_matrix > 0).astype(int)
    degree = np.diag(adjacency.sum(axis=1))
    laplacian = degree - adjacency
    eigen_values = np.linalg.eigvalsh(laplacian)
    return np.sum(np.isclose(eigen_values, 0, atol=1e-8)) == 1


class MixingMatrix:
    """Define mixing matrix and check connectivity

    Args:
        mixing_matrix (matrix): matrix if already defined
        k (int): number of group
        structure_strength (float): probability of a link between two nodes of the same group.
    """

    def __init__(self, mixing_matrix, k, structure_strength):
        self.k = k
        self.structure_strength = structure_strength
        self.mixing_matrix = mixing_matrix

    def define_mixing_matrix(
        self, models=None, inter_group=0.1, zero_ratio=0.3, type_graph="ref"
    ):
        """Define the mixing matrix for SBM

        Args:
            mixing_matrix(np.array(float)): mixing matrix (k,k) for SBM. 
                Default to use structure strength.
            models (matrix): list of random weights to be used
            inter_groups (float, optional): the greater, the stronger 
                could be the inter-class structure.
            zero_ratio (float, optional): proportion of zeros in the matrix, used in sparse case
            type_graph (string, optional): type of mixing matrix. 
                Could be "ref", "sparse", "chain", "star" or "donut".

        Returns:
            np.array: symmetric probability matrix
        """
        if len(self.mixing_matrix) > 0:
            matrix = self.mixing_matrix
        else:
            if models is None:
                models = []
            connectivity = False
            while connectivity is False:
                match type_graph:
                    case "ref":
                        matrix = self.define_mixing_matrix_ref(inter_group, models)
                    case "sparse":
                        matrix = self.define_mixing_matrix_sparse(
                            inter_group, zero_ratio
                        )
                    case "chain":
                        matrix = self.define_mixing_matrix_chain(inter_group)
                    case "star":
                        matrix = self.define_mixing_matrix_star(inter_group)
                    case "donut":
                        matrix = self.define_mixing_matrix_donut(inter_group)
                connectivity = check_connectivity(matrix)
        return matrix

    def define_mixing_matrix_ref(self, inter_group, models):
        """Define the mixing matrix for SBM

        Args:
            inter_groups (float): the greater, the stronger could be the inter-class structure.

        Returns:
            np.array: symmetric probability matrix
        """
        if len(models) > 0:
            random_weights = models
        else:
            random_weights = np.round(
                np.random.uniform(0, inter_group, (self.k, self.k)), 2
            )
        mixing_matrix = random_weights + self.structure_strength * np.eye(self.k)
        return normalize_symmetric(mixing_matrix)

    def define_mixing_matrix_chain(self, inter_group):
        """Define the mixing matrix for SBM with a chain format
        Group k is linked with and only with k-1 and k+1

        Args:
            inter_group (float): probability of a link between two nodes of different groups.

        Returns:
            np.array: symmetric probability matrix
        """
        matrix = np.zeros((self.k, self.k))
        for i in range(self.k):
            matrix[i, i] = self.structure_strength
            if i > 0:
                matrix[i, i - 1] = np.round(np.random.uniform(0.01, inter_group, 1), 2)
            if i < self.k - 1:
                matrix[i, i + 1] = np.round(np.random.uniform(0.01, inter_group, 1), 2)
        return normalize_symmetric(matrix)

    def define_mixing_matrix_star(self, inter_group):
        """Define the mixing matrix for SBM with a star format
        Group k is linked with and only with group 1

        Args:
            inter_group (float): probability of a link between two nodes of different groups.

        Returns:
            np.array: symmetric probability matrix
        """
        matrix = np.zeros((self.k, self.k))
        matrix[0, 1:] = np.round(np.random.uniform(0.01, inter_group, 1), 2)
        matrix[1:, 0] = np.round(np.random.uniform(0.01, inter_group, 1), 2)
        np.fill_diagonal(matrix, self.structure_strength)
        return normalize_symmetric(matrix)

    def define_mixing_matrix_donut(self, inter_group):
        """Define the mixing matrix for SBM with a donut format
        Group k is linked with and only with k-1 and k+1
        Group 1 is also linked with final group k

        Args:
            inter_group (float): probability of a link between two nodes of different groups.

        Returns:
            np.array: symmetric probability matrix
        """
        matrix = np.zeros((self.k, self.k))
        for i in range(self.k):
            matrix[i, i] = self.structure_strength
            matrix[i, (i - 1) % self.k] = np.round(
                np.random.uniform(0.01, inter_group, 1), 2
            )
            matrix[i, (i + 1) % self.k] = np.round(
                np.random.uniform(0.01, inter_group, 1), 2
            )
        return normalize_symmetric(matrix)

    def define_mixing_matrix_sparse(self, inter_group, zero_ratio=0.3):
        """Define the mixing matrix for SBM without a specific shape
        but also without full connectivity

        Args:
            inter_group (float): probability of a link between two nodes of different groups.
            zero_ratio (float, optional): proportion of zeros.

        Returns:
            np.array: symmetric probability matrix to be normalized
        """
        # Get indices of the upper triangular side
        triu_indices = np.triu_indices(self.k, k=1)  # k=1 to exclude diagonal
        idx_ct = len(triu_indices[0])
        zeros_ct = int(zero_ratio * idx_ct)

        # Generate values
        values = np.round(np.random.uniform(0.01, inter_group, idx_ct), 2)

        # Random zeros
        zero_positions = np.random.choice(idx_ct, size=zeros_ct, replace=False)
        values[zero_positions] = 0.0

        # Matrix generation
        matrix = np.zeros((self.k, self.k))
        matrix[triu_indices] = values
        matrix = matrix + matrix.T
        np.fill_diagonal(matrix, self.structure_strength)

        return normalize_symmetric(matrix)
