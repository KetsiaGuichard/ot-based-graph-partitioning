"""
Simulation helpers to build geometric and stochastic-block-model graphs with attributes.

This module provides small classes used to create toy graphs for experiments:
"""

import itertools
import networkx as nx
import numpy as np
import seaborn as sns
from sklearn.metrics.pairwise import euclidean_distances

from src.distances import (
    distance_graph,
    distance_functions_dtwai,
    distance_histograms_fast,
)
from src.shapes import MixingMatrix
from src.simulation.attributes import SimulatedFunctionalData, SimulatedHistograms


class SimulatedGeometricGraph:
    """Create a simulated geometric graph
    
    Args:
        k_group_ct (list(int)): number of nodes per class.
        means (list(tuple)): list of means for each class (2D points).
        stds (list(int)): list of standard deviations for each class.
    """
    def __init__(
        self,
        k_group_ct,
        means,
        stds,
    ):
        self.k = len(k_group_ct)
        self.means = means
        self.stds = stds
        self.k_group_ct = k_group_ct
        self.x, self.y, self.distance_matrix = self.scatter(means, stds)
        self.pos = None

    def connect_graph(self, graph, distance_matrix):
        """
        Ensure that the graph is connected by adding edges between components.
        
        Args:
            graph (nx.Graph): The input graph which may be disconnected.
            distance_matrix (np.ndarray): The distance matrix used to find closest nodes.
        
        Returns:
            nx.Graph: A connected graph.
        """
        connected_list = list(nx.connected_components(graph))
        for component in range(len(connected_list)):
            for next_component in range(component + 1, len(connected_list)):
                subset_distances = distance_matrix[
                    np.ix_(
                        list(connected_list[component]),
                        list(connected_list[next_component]),
                    )
                ]
                min_distance = np.unravel_index(
                    subset_distances.argmin(), subset_distances.shape
                )
                vertice_component = list(connected_list[component])[min_distance[0]]
                vertice_next_component = list(connected_list[next_component])[
                    min_distance[1]
                ]
                graph.add_edge(
                    vertice_component,
                    vertice_next_component,
                    length=distance_matrix[vertice_component, vertice_next_component],
                )
        return graph

    def create_graph(self, radius):
        """
        Create geometric graph based on a distance threshold (radius).
        
        Args:
            radius (float): Distance threshold for connecting nodes.
        
        Returns:
            nx.Graph: The created geometric graph.
        """
        graph = nx.Graph()
        for i in range(len(self.x)):
            graph.add_node(i)
            for j in range(i + 1, len(self.x)):
                if self.distance_matrix[i, j] <= radius:
                    graph.add_edge(i, j, length=self.distance_matrix[i, j])
        graph = self.connect_graph(graph, self.distance_matrix)
        if self.pos is None:
            self.pos = nx.spring_layout(graph)
        return graph

    def get_true_labels(self):
        """
        Return true labels
        """
        return np.repeat(np.arange(len(self.k_group_ct)), self.k_group_ct)

    def scatter(self, means: list[tuple], stds: list[int]):
        """
        Scatter points according to Gaussian distributions for each class.
        
        Args:
            means (list(tuple)): list of means for each class (2D points).
            stds (list(int)): list of standard deviations for each class.
        
        Returns:
            np.ndarray: x coordinates of points.
            np.ndarray: y coordinates of points.
            np.ndarray: distance matrix of points.
        """
        samples = [
            np.random.normal(loc=mean, scale=std, size=(n, 2))
            for mean, std, n in zip(means, stds, self.k_group_ct)
        ]
        points = np.vstack(samples)
        x, y = points[:, 0], points[:, 1]
        distance_matrix = euclidean_distances(np.stack((x, y), axis=1))
        return x, y, distance_matrix


class SimulatedGraph:
    """Create a simulated structure thanks to Stochastic Block Model

    Args:
        groups (list(int)): number of nodes per class.
        type_graph(str, optional): "ref", "chain", "star", "donut"
        structure_strength (float, optional): the greater, 
            the stronger is the intra-class structure. Default 2.
        mixing_matrix(np.array(float), optional): mixing matrix (k,k) for SBM. 
            Default to use structure strength.
    """

    def __init__(
        self,
        groups,
        type_graph: str = "ref",
        structure_strength: float = 2,
        mixing_matrix=np.array([]),
    ):
        self.k = len(groups)
        self.structure_strength = structure_strength
        self.groups = groups
        self.mixing_matrix = self.create_mixing_matrix(
            mixing_matrix, structure_strength, type_graph
        )
        self.graph = self.create_graph()
        self.graph_model = self.__get_graph_base()

    def __get_graph_base(self):
        """
        Def mixing matrix (basis for further simulation on structure_force)

        Returns:
            np.array: SBM basis without diagonal addition
        """
        random_weights = np.round(np.random.uniform(0, 0.15, (self.k, self.k)), 2)
        return random_weights

    def create_mixing_matrix(self, mixing_matrix, structure_strength, type_graph):
        """
        Generate mixing matrix
        """
        mix = MixingMatrix(mixing_matrix, self.k, structure_strength)
        return mix.define_mixing_matrix(type_graph=type_graph)

    def create_graph(self):
        """Create networkx graph and give it some colors for visualisation

        Returns:
            Networkx Object: graph with colors associated with nodes
        """
        non_connected = True
        while non_connected:  # force to connect graph
            graph = nx.stochastic_block_model(
                self.groups.tolist(), self.mixing_matrix.tolist()
            )
            non_connected = not nx.is_connected(graph)

        colors = sns.color_palette("Set2", self.k)
        for i, (block_start, block_size) in enumerate(
            zip(np.cumsum([0] + self.groups.tolist()), self.groups)
        ):
            for node in range(block_start, block_start + block_size):
                graph.nodes[node]["color"] = colors[i]
                graph.nodes[node]["group"] = i

        return graph

    def get_colors(self):
        """Get colors of graph
        Returns:
            list: list of k colors
        """
        colors = [self.graph.nodes[node]["color"] for node in self.graph.nodes]
        return colors

    def get_k(self):
        """
        Get number of classes
        """
        return len(self.groups)

    def get_true_labels(self):
        """
        Return true labels
        """
        return np.repeat(np.arange(len(self.groups)), self.groups)

    def test_structure_force(self, structure_force: list, type_graph="ref", fixed=True):
        """
        Create several graph distance matrices

        Args:
            structure_force (list(float)): parameters to be tested
            type_graph (str, optional): "ref", "chain", "star" or "donut". Default to "ref".
            fixed (boolean, optional): should the seed be the same at each iteration?
                Default to True

        Returns:
            list(array): matrices of distance
            list(array): mixing matrices
        """
        k = self.get_k()
        list_distances_graph = list([])
        list_mixing_matrix = list([])
        for structure in structure_force:
            if fixed:
                models = self.graph_model
            else:
                models = None
            mix = MixingMatrix(mixing_matrix=[], k=k, structure_strength=structure)
            mixing_matrix = mix.define_mixing_matrix(
                models=models, type_graph=type_graph
            )
            graph = SimulatedGraph(
                groups=self.groups,
                type_graph=type_graph,
                structure_strength=structure,
                mixing_matrix=mixing_matrix,
            ).graph
            tmp_dist = distance_graph(graph)
            list_distances_graph.append(tmp_dist)
            list_mixing_matrix.append(mixing_matrix)
        return list_distances_graph, list_mixing_matrix

    def plot(self, ax=None):
        """Plot simulated graph"""
        colors = self.get_colors()
        nx.draw(self.graph, node_size=20, node_color=colors, with_labels=False, ax=ax)


class SimulatedAttributedGraph:
    """Create Graph with attributed data
    Some parameters can be tested

    Args:
        k (int): number of classes
        n (int): number of nodes
        type_graph (str, optional): "ref", "chain", "star", "donut"
        alterate (str, optional): "divisive" or "aggregative", default to None
        equilibre (int, optional): the greater, the less pronounced the imbalance
    """

    def __init__(self, k: int, groups: np.array = None):
        self.k = k
        self.groups = groups
        self.graph_model = self.__get_graph_base()

    def get_n(self):
        """
        Get number of nodes
        """
        return int(sum(self.groups))

    def get_k(self):
        """
        Get number of classes
        """
        return self.k

    def __get_graph_base(self):
        """
        Def mixing matrix (basis for further simulation on structure_force)

        Returns:
            np.array: SBM basis without diagonal addition
        """
        random_weights = np.round(np.random.uniform(0, 0.15, (self.k, self.k)), 2)
        return random_weights

    def get_true_labels(self):
        """
        Return true labels
        """
        labels_base = self.groups
        return np.repeat(np.arange(len(labels_base)), labels_base)

    def test_structure_force(self, structure_force: list, type_graph="ref", fixed=True):
        """
        Create several graph distance matrices

        Args:
            structure_force (list(float)): parameters to be tested
            type_graph (str, optional): "ref", "chain", "star" or "donut". Default to "ref".
            fixed (boolean, optional): should the seed be the same at each iteration?
                Default to True

        Returns:
            list(array): matrices of distance
            list(array): mixing matrices
        """
        list_distances_graph = list([])
        list_mixing_matrix = list([])
        for structure in structure_force:
            if fixed:
                models = self.graph_model
            else:
                models = None
            mix = MixingMatrix(mixing_matrix=[], k=self.k, structure_strength=structure)
            mixing_matrix = mix.define_mixing_matrix(
                models=models, type_graph=type_graph
            )
            graph = SimulatedGraph(
                groups=self.groups,
                type_graph=type_graph,
                structure_strength=structure,
                mixing_matrix=mixing_matrix,
            ).graph
            tmp_dist = distance_graph(graph)
            list_distances_graph.append(tmp_dist)
            list_mixing_matrix.append(mixing_matrix)
        return list_distances_graph, list_mixing_matrix

    def test_epsilon(self, epsilon_list: list):
        """
        Crée plusieurs matrices de distance de fonctions en changeant la perturbation

        Args:
            epsilon_list (list(float)): parameters to be tested
            fixed (boolean): should the seed be the same at each iteration? Default to True.

        Returns:
            list(np.array): list of distances
            list(np.array): histograms defined
        """
        list_distances_fun = list([])
        list_fun = list([])
        for epsilon in epsilon_list:
            models = None
            tmp_fun = SimulatedFunctionalData(
                self.groups, epsilon, models
            ).functions_array
            tmp_dist = distance_functions_dtwai(tmp_fun)
            list_distances_fun.append(tmp_dist)
            list_fun.append(tmp_fun)
        return list_distances_fun, list_fun

    def test_c(self, c_list: list):
        """
        Create several distance matrix between histograms
        While conserving the same models of distribution per classes

        Args:
            c_list (list(float)): parameters to be tested
            fixed (boolean): should the seed be the same at each iteration? Default to True.

        Returns:
            list(np.array): list of distances
            list(np.array): histograms defined
        """
        list_distances_hist = list([])
        list_hist = list([])
        for c in c_list:
            models = None
            tmp_hist = SimulatedHistograms(self.groups, models, c=c).hist_array
            tmp_dist = distance_histograms_fast(tmp_hist)
            list_distances_hist.append(tmp_dist)
            list_hist.append(tmp_hist)
        return list_distances_hist, list_hist

    def test_c_epsilon(self, c_list: list, epsilon_list: list, product=True):
        """
        Create several distance matrix between histograms and curves
        While conserving the same models of distribution per classes

        Args:
            c_list (list(float)): parameters to be tested (histograms)
            epsilon_list (list(float)): parameters to be tested (curves)
            fixed (boolean): should the seed be the same at each iteration? Default to True.
            product (boolean): should attributes be combined or simply added ?
                Default to combined (True).

        Returns:
            list(np.array): list of distances (aggregated)
            list(np.array): histograms defined
            list(np.array): curves defined
        """
        list_distances_hist, list_hist = self.test_c(c_list)
        list_distances_fun, list_fun = self.test_epsilon(epsilon_list)
        if product:
            list_distances_attributes = [
                (h + f) / 2
                for h, f in itertools.product(list_distances_hist, list_distances_fun)
            ]
        else:
            list_distances_attributes = [
                (list_distances_hist[i] + list_distances_fun[i]) / 2
                for i in range(len(list_distances_hist))
            ]
        list_distances_attributes = [i / i.max() for i in list_distances_attributes]
        return (
            list_distances_attributes,
            list_distances_hist,
            list_distances_fun,
            list_hist,
            list_fun,
        )
