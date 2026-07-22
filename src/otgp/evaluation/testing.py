"""
Testing framework for graph clustering experiments.
Provides methods for both non attributed and attributed graph clustering methods evaluation.
"""

import os
from abc import ABC
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import SpectralClustering

from src.otgp.clustering import (
    ClusteringInit,
    FrechetKMeans,
    EmbeddedKMeans,
    SemiRelaxedGWClustering,
    SemiRelaxedFGWClustering,
    transport_plan_to_labels,
    compute_m_generic,
    compute_m_medoid,
)
from src.otgp.distances import combine_alpha
from .utils import (
    eval_df,
    distance_mixing,
    heat_kernel,
    modularity_numpy_fast,
)
from .metrics import ExternalEvaluation


class BaseTestingMethods(ABC):
    """Base testing class with core shared helpers.

    Args:
        n (int): number of individuals
        k (int): number of classes
        parameters (dict): parameters for evaluation dataframe
    """

    def __init__(self, n, k, parameters):
        self.k = k
        self.n = n
        self.parameters = parameters

    def start_initialisation_random(self, distance_matrix):
        """Random initialisation of centroids and transport plan."""
        init = ClusteringInit(self.n, self.k, distance_matrix)
        centroids_init, init_ot = init.random_init()
        return centroids_init, init_ot

    def start_initialisation(self, distance_matrix, embedded=False):
        """Kmeans++ initialisation of centroids and transport plan.

        Args:
            distance_matrix (np.ndarray): distance matrix
            embedded (bool): whether to use embedded kmeans++ init

        Returns:
            list(int): list of initial centroids
            np.ndarray: initial transport plan
        """
        init = ClusteringInit(self.n, self.k, distance_matrix)
        if embedded:
            centroids_init, init_ot = init.embedded_kmeanspp_init()
        else:
            centroids_init, init_ot = init.kmeanspp_init()
        return centroids_init, init_ot

    def frechet_kmeans_testing(self, distance_matrix, centroids_init):
        """Frechet kmeans testing with given distance matrix and centroids init."""
        frechet_kmeans = FrechetKMeans(k=self.k, centroids_init=centroids_init)
        partition = frechet_kmeans.partitioning_simple(
            structural_matrix=distance_matrix
        )
        return partition["labels"]

    def embedded_kmeans_testing(self, distance_matrix, centroids_init):
        """Embedded kmeans testing with given distance matrix and centroids init."""
        embedded_kmeans = EmbeddedKMeans(k=self.k, centroids_init=centroids_init)
        partition = embedded_kmeans.partitioning_simple(
            structural_matrix=distance_matrix
        )
        return partition["labels"]

    def srgw_testing(
        self,
        distance_matrix,
        c2=None,
        value=1,
        target_type="distance",
        embedded=False,
        g0="kmeans",
    ):
        """Semi-relaxed GW testing with given distance matrix and target C2."""
        srgw = SemiRelaxedGWClustering(
            n=self.n, k=self.k, g0=g0, target_c=c2, target_type=target_type, value=value
        )
        partition = srgw.partitioning_simple(
            structural_matrix=distance_matrix, embedded=embedded
        )
        return partition["labels"]

    def distance_to_affinity(self, distance_matrix, k=7):
        """
        Transform a distance matrix into an affinity matrix, using local kernel strategy.

        Args:
            distance_matrix (np.ndarray): distance matrix to be transformed.
            k (int): number of neighbors considered (default to 7).

        Returns:
            np.ndarray: Affinity matrix.
        """
        n = distance_matrix.shape[0]
        sorted_d = np.sort(distance_matrix, axis=1)
        sigma = sorted_d[:, k]
        similarity = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    similarity[i, j] = np.exp(
                        -distance_matrix[i, j] ** 2 / (sigma[i] * sigma[j])
                    )
        similarity = similarity / similarity.max()
        return similarity

    def distance_to_affinity_gaussian(self, distance_matrix, sigma=None):
        """
        Transform a distance matrix into an affinity matrix, using gaussian kernel.

        Args:
            distance_matrix (np.ndarray): distance matrix to be transformed.
            sigma (float): std considered for gaussian kernel.

        Returns:
            np.ndarray: Affinity matrix.
        """
        distance = distance_matrix.astype(float)
        if sigma is None:
            sigma = np.median([distance > 0])
        similarity = np.exp(-(distance**2) / (2 * sigma**2))
        np.fill_diagonal(similarity, 1.0)
        similarity = similarity / similarity.max()
        return similarity

    def spectral_testing(self, input_matrix):
        """Spectral clustering testing with given distance or normalized laplacian matrix."""
        spectral_clustering = SpectralClustering(
            n_clusters=self.k,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=0,
        )
        return spectral_clustering.fit_predict(input_matrix)

    def sr_spec_gw(self, laplacian, adjacency, ts=np.linspace(0.1, 3, 5)):
        """Semi-relaxed GW (heat kernel) with parameter optimization"""
        mod_dict = dict.fromkeys(ts, 0)
        labels_dict = dict.fromkeys(ts, 0)
        for t in ts:
            heat_kernel_matrix = heat_kernel(laplacian, t)
            labels_dict[t] = SemiRelaxedGWClustering(
                self.n, self.k, g0="kmeans", target_c=np.eye(self.k)
            ).partitioning_simple(heat_kernel_matrix)["labels"]
            mod_dict[t] = modularity_numpy_fast(adjacency, labels_dict[t])
        return labels_dict[max(mod_dict, key=mod_dict.get)]

    def saving(self, eval_df_obj, folder="data"):
        """Save evaluation dataframe to CSV file.

        Args:
            eval_df_obj (pd.DataFrame): evaluation dataframe to save
            folder (str): folder to save the CSV file in
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(folder, exist_ok=True)
        filename = f"{folder}/{timestamp}.csv"
        with open(filename, "w") as f:
            eval_df_obj.to_csv(f, index=False)
            f.flush()
            os.fsync(f.fileno())


class SimpleGraphTesting(BaseTestingMethods):
    """Testing for simple (unattributed) graphs."""

    def testing_init(self, r1_total, infos, true_labels=None, save=False):
        """
        Testing methods for simple graphs with given distance matrices.

        Args:
            r1_total (list of np.ndarray): list of distance matrices
            infos (dict): additional information for evaluation dataframe
            true_labels (list): ground truth labels
            save (bool): whether to save the evaluation dataframe

        Returns:
            pd.DataFrame: evaluation dataframe
        """
        evaluation = pd.DataFrame()
        for i in range(len(r1_total)):
            distance_matrix = r1_total[i]["distance"]

            embedded_distance = cdist(distance_matrix, distance_matrix)
            embedded_distance = embedded_distance / embedded_distance.max()

            # Init
            centroids_init_random, init_ot_random = self.start_initialisation_random(
                distance_matrix
            )
            centroids_init, init_ot = self.start_initialisation(
                distance_matrix, embedded=False
            )
            centroids_init_embedded, init_ot_embedded = self.start_initialisation(
                distance_matrix, embedded=True
            )
            _, init_ot_embedded2 = self.start_initialisation(
                embedded_distance, embedded=True
            )

            cellules = [
                # Frechet kmeans
                self.frechet_kmeans_testing(distance_matrix, centroids_init_random),
                self.frechet_kmeans_testing(distance_matrix, centroids_init),
                self.frechet_kmeans_testing(distance_matrix, centroids_init_embedded),
                # Embedded Frechet kmeans
                self.embedded_kmeans_testing(distance_matrix, centroids_init_random),
                self.embedded_kmeans_testing(distance_matrix, centroids_init),
                self.embedded_kmeans_testing(distance_matrix, centroids_init_embedded),
                # srGW max
                self.srgw_testing(
                    distance_matrix,
                    value=np.max(distance_matrix),
                    g0=init_ot_random,
                ),
                self.srgw_testing(
                    distance_matrix,
                    value=np.max(distance_matrix),
                    g0=init_ot,
                ),
                self.srgw_testing(
                    distance_matrix,
                    value=np.max(distance_matrix),
                    g0=init_ot_embedded,
                ),
                self.srgw_testing(
                    distance_matrix,
                    value=np.max(distance_matrix),
                    g0=init_ot_embedded2,
                ),
                # srGW mean
                self.srgw_testing(
                    distance_matrix,
                    value=np.mean(distance_matrix),
                    g0=init_ot_random,
                ),
                self.srgw_testing(
                    distance_matrix,
                    value=np.mean(distance_matrix),
                    g0=init_ot,
                ),
                self.srgw_testing(
                    distance_matrix,
                    value=np.mean(distance_matrix),
                    g0=init_ot_embedded,
                ),
                self.srgw_testing(
                    distance_matrix,
                    value=np.mean(distance_matrix),
                    g0=init_ot_embedded2,
                ),
                # Embedded srGW max
                self.srgw_testing(
                    distance_matrix,
                    value=np.max(distance_matrix),
                    g0=init_ot_random,
                    embedded=True,
                ),
                self.srgw_testing(
                    distance_matrix,
                    value=np.max(distance_matrix),
                    g0=init_ot,
                    embedded=True,
                ),
                self.srgw_testing(
                    distance_matrix,
                    value=np.max(distance_matrix),
                    g0=init_ot_embedded,
                    embedded=True,
                ),
                self.srgw_testing(
                    distance_matrix,
                    value=np.max(distance_matrix),
                    g0=init_ot_embedded2,
                    embedded=True,
                ),
                # srGW mean
                self.srgw_testing(
                    distance_matrix,
                    value=np.mean(distance_matrix),
                    g0=init_ot_random,
                    embedded=True,
                ),
                self.srgw_testing(
                    distance_matrix,
                    value=np.mean(distance_matrix),
                    g0=init_ot,
                    embedded=True,
                ),
                self.srgw_testing(
                    distance_matrix,
                    value=np.mean(distance_matrix),
                    g0=init_ot_embedded,
                    embedded=True,
                ),
                self.srgw_testing(
                    distance_matrix,
                    value=np.mean(distance_matrix),
                    g0=init_ot_embedded2,
                    embedded=True,
                ),
            ]
            labels = [
                "Frechet kmeans - random init",
                "Frechet kmeans - kmeans++ init",
                "Frechet kmeans - embedded kmeans++ init",
                "Embedded kmeans - random init",
                "Embedded kmeans - kmeans++ init",
                "Embedded kmeans - embedded kmeans++ init",
                "srGW isolated nodes (max) - random init",
                "srGW isolated nodes (max) - kmeans++ init",
                "srGW isolated nodes (max) - embedded kmeans++ init",
                "srGW isolated nodes (max) - embedded embedded kmeans++ init",
                "srGW isolated nodes (mean) - random init",
                "srGW isolated nodes (mean) - kmeans++ init",
                "srGW isolated nodes (mean) - embedded kmeans++ init",
                "srGW isolated nodes (mean) - embedded embedded kmeans++ init",
                "Embedded srGW isolated nodes (max) - random init",
                "Embedded srGW isolated nodes (max) - kmeans++ init",
                "Embedded srGW isolated nodes (max) - embedded kmeans++ init",
                "Embedded srGW isolated nodes (max) - embedded embedded kmeans++ init",
                "Embedded srGW isolated nodes (mean) - random init",
                "Embedded srGW isolated nodes (mean) - kmeans++ init",
                "Embedded srGW isolated nodes (mean) - embedded kmeans++ init",
                "Embedded srGW isolated nodes (mean) - embedded embedded kmeans++ init",
            ]

            external_evaluation = pd.DataFrame()
            for method in range(len(cellules)):
                external_evaluation_tmp = ExternalEvaluation(
                    true_labels, cellules[method]
                ).get_df(labels[method])
                external_evaluation = pd.concat(
                    [external_evaluation, external_evaluation_tmp]
                )
            tmp_evaluation = external_evaluation
            parameters_tmp = {k: v[i] for k, v in self.parameters.items()}
            tmp_evaluation = eval_df(tmp_evaluation, parameters_tmp, infos)
            evaluation = pd.concat([evaluation, tmp_evaluation])

        if save:
            self.saving(evaluation)

        return evaluation

    def testing_simple(
        self, r1_total, infos, mixing_matrix, c_target, true_labels=None, save=False
    ):
        """
        Testing methods for simple graphs with given distance matrices.

        Args:
            r1_total (list of dict of np.ndarray): dict of source matrices
            infos (dict): additional information for evaluation dataframe
            mixing_matrix (list of np.ndarray): list of mixing matrices for true target c2
            true_labels (list): ground truth labels
            save (bool): whether to save the evaluation dataframe

        Returns:
            pd.DataFrame: evaluation dataframe
        """
        evaluation = pd.DataFrame()
        for i in range(len(r1_total)):
            distance_matrix = r1_total[i]["distance"]
            adjacency_matrix = r1_total[i]["adjacency"]
            affinity_matrix = self.distance_to_affinity(distance_matrix)

            # laplacian_matrix = R1_total[i]['laplacian']
            embedded_distance = cdist(distance_matrix, distance_matrix)
            embedded_distance = embedded_distance / embedded_distance.max()

            # Init
            centroids_init_embedded, init_ot_embedded = self.start_initialisation(
                distance_matrix, embedded=True
            )
            _, init_ot_embedded2 = self.start_initialisation(
                embedded_distance, embedded=True
            )
            # centroids_init, init_ot = self.start_initialisation(distance_matrix)
            # centroids_random, init_ot_random = self.start_initialisation_random(
            #    distance_matrix
            # )
            c_sim = distance_mixing(mixing_matrix[i])
            embedded_c_sim = cdist(c_sim, c_sim)
            embedded_c_sim = embedded_c_sim / embedded_c_sim.max()

            cellules = [
                self.frechet_kmeans_testing(distance_matrix, centroids_init_embedded),
                self.embedded_kmeans_testing(distance_matrix, centroids_init_embedded),
                self.srgw_testing(
                    distance_matrix,
                    value=np.max(distance_matrix),
                    g0=init_ot_embedded,
                ),
                self.srgw_testing(
                    distance_matrix,
                    value=np.mean(distance_matrix),
                    g0=init_ot_embedded,
                ),
                self.srgw_testing(distance_matrix, c2=c_sim, g0=init_ot_embedded),
                self.srgw_testing(distance_matrix, c2=c_target[i], g0=init_ot_embedded),
                self.srgw_testing(
                    distance_matrix,
                    value=np.max(embedded_distance),
                    g0=init_ot_embedded2,
                    embedded=True,
                ),
                self.srgw_testing(
                    distance_matrix,
                    value=np.mean(embedded_distance),
                    g0=init_ot_embedded2,
                    embedded=True,
                ),
                self.srgw_testing(
                    adjacency_matrix, target_type="adjacency", g0="kmeans"
                ),
                self.spectral_testing(affinity_matrix),
                self.spectral_testing(adjacency_matrix),
                # self.sr_spec_GW(
                #    laplacian_matrix,
                #    adjacency_matrix
                # )
            ]
            labels = [
                "Frechet kmeans - embedded kmeans++ init",
                "Embedded kmeans - embedded kmeans++ init",
                "srGW - isolated nodes (max) - embedded kmeans++ init",
                "srGW - isolated nodes (mean) - embedded kmeans++ init",
                "srGW true target (proba) - embedded kmeans++ init",
                "srGW true target (median) - embedded kmeans++ init",
                "Embedded srGW - isolated nodes (max) - embedded embedded kmeans++ init",
                "Embedded srGW - isolated nodes (mean) - embedded embedded kmeans++ init",
                "adjacency matrix - kmeans",
                "spectral clustering affinity - no init",
                "spectral clustering adjacency - no init",
                # "srSpecGW - kmeans"
            ]

            external_evaluation = pd.DataFrame()
            for method in range(len(cellules)):
                external_evaluation_tmp = ExternalEvaluation(
                    true_labels, cellules[method]
                ).get_df(labels[method])
                external_evaluation = pd.concat(
                    [external_evaluation, external_evaluation_tmp]
                )
            tmp_evaluation = external_evaluation
            parameters_tmp = {k: v[i] for k, v in self.parameters.items()}
            tmp_evaluation = eval_df(tmp_evaluation, parameters_tmp, infos)
            evaluation = pd.concat([evaluation, tmp_evaluation])

        if save:
            self.saving(evaluation)

        return evaluation


class AttributedGraphTesting(BaseTestingMethods):
    """Testing for attributed graphs. Stores attributes, distances and graph distances."""

    def compute_m(self, ot, fgw_config, distances, attributes, medoid):
        """Compute M matrix for srfgw clustering.

        Args:
            ot (np.ndarray): current transport plan
            fgw_config (dict): configuration for fgW
            distances (list of np.ndarray): list of distance matrices for attributes
            attributes (np.ndarray): attribute matrix
            medoid (bool): whether to use medoid computation

        Returns:
            np.ndarray: computed M matrix
        """
        if not medoid:
            m = compute_m_generic(
                ot=ot,
                attributes=attributes,
                tasks_distances=fgw_config["tasks_distances"],
                tasks_barycenters=fgw_config["tasks_barycenters"],
                weights=fgw_config["weights"],
                powers=fgw_config["powers"],
            )
        else:
            m = compute_m_medoid(
                ot=ot,
                distance_matrices=distances,
                weights=fgw_config["weights"],
                powers=fgw_config["powers"],
            )
        return m

    def frechet_kmeans_testing(
        self,
        structural_matrix,
        attributes_matrix,
        centroids_init,
        alpha=0.5,
        alpha_type=None,
    ):
        """
        Frechet kmeans testing with given structural and attribute matrices and centroids init.

        Args:
            structural_matrix (np.ndarray): structural distance matrix
            attributes_matrix (np.ndarray): attribute distance matrix
            centroids_init (list): list of initial centroids
            alpha (float): tradeoff parameter
            alpha_type (str): type of alpha combination

        Returns:
            list: predicted labels
        """
        frechet_kmeans = FrechetKMeans(k=self.k, centroids_init=centroids_init)
        partition = frechet_kmeans.partitioning(
            structural_matrix, attributes_matrix, alpha=alpha, alpha_type=alpha_type
        )
        return partition["labels"]

    def embedded_kmeans_testing(
        self, structural_matrix, attributes_matrix, centroids_init, alpha=0.5
    ):
        """
        Embedded kmeans testing with given structural and attribute matrices and centroids init.

        Args:
            structural_matrix (np.ndarray): structural distance matrix
            attributes_matrix (np.ndarray): attribute distance matrix
            centroids_init (list): list of initial centroids
            alpha (float): tradeoff parameter

        Returns:
            list: predicted labels
        """
        embedded_kmeans = EmbeddedKMeans(k=self.k, centroids_init=centroids_init)
        partition = embedded_kmeans.partitioning(
            structural_matrix, attributes_matrix, alpha
        )
        return partition["labels"]

    def srgw_testing(
        self,
        structural_matrix,
        attributes_matrix,
        value=1,
        target_type="distance",
        g0="kmeans",
        embedded=False,
        alpha=0.5,
    ):
        """
        Semi-relaxed GW testing with given structural and attribute matrices.

        Args:
            structural_matrix (np.ndarray): structural distance matrix
            attributes_matrix (np.ndarray): attribute distance matrix
            value (float): value for equidistant nodes target matrix
            g0 (str or np.ndarray): initialization method or transport plan
            embedded (bool): whether to use embedded version
            alpha (float): tradeoff parameter
            alpha_type (str): type of alpha combination

        Returns:
            list: predicted labels
        """
        srgw = SemiRelaxedGWClustering(
            n=self.n, k=self.k, g0=g0, value=value, target_type=target_type
        )
        partition = srgw.partitioning(
            structural_matrix,
            attributes_matrix,
            alpha=alpha,
            embedded=embedded,
        )
        return partition["labels"]

    def srfgw_testing(
        self,
        structural_matrix,
        attributes_matrices_dict,
        fgw_config,
        g0_attributes,
        value=1,
        target_type="distance",
        g0="kmeans",
        medoid=False,
    ):
        """
        Semi-relaxed FGW testing with given structural and attribute matrices.

        Args:
            structural_matrix (np.ndarray): structural distance matrix
            attributes_matrices_dict (dict): dictionary of attribute distance matrices
            value (float): value for equidistant nodes target matrix
            fgw_config (dict): configuration for fgW
            g0_attributes (str or np.ndarray): initialization method or transport plan
                for attributes
            g0 (str or np.ndarray): initialization method or transport plan for structure
            medoid (bool): whether to use medoid computation

        Returns:
            list: predicted labels
            float: srfgw distance
        """
        srfgw = SemiRelaxedFGWClustering(
            n=self.n,
            k=self.k,
            g0=g0,
            g0_attributes=g0_attributes,
            target_type=target_type,
            value=value,
        )
        srfgw.define_method(
            weights=fgw_config["weights"],
            powers=fgw_config["powers"],
            tasks_distances=fgw_config.get("tasks_distances", None),
            tasks_barycenters=fgw_config.get("tasks_barycenters", None),
            alpha=fgw_config.get("alpha", None),
        )
        ot_fused_clustering = srfgw.iterate(
            structural_matrix,
            attributes_matrices_dict,
            iterations=10,
            medoid=medoid,
        )
        srfgw_dist = ot_fused_clustering["srfgw_dist"]
        labels = transport_plan_to_labels(ot_fused_clustering["ot"][-1])
        return labels, srfgw_dist

    def testing_attributed(
        self,
        r1_total,
        attributes_distances,
        infos,
        fgw_config,
        true_labels=None,
        save=False,
        medoid=True,
    ):
        """
        Testing methods for attributed graphs
            with given structural and attribute distances.

        Args:
            structural_distances (list of np.ndarray): list of structural distance matrices
            r1_total (list of dict): list of dictionaries of attribute distance matrices
            infos (dict): additional information for evaluation dataframe
            fgw_config (dict): configuration for fgW
            true_labels (list): ground truth labels
            save (bool): whether to save the evaluation dataframe
            medoid (bool): whether to use medoid computation

        Returns:
            pd.DataFrame: evaluation dataframe
        """
        evaluation = pd.DataFrame()
        for i in range(len(r1_total)):
            # Get matrices
            structural_distance_matrix = r1_total[i]["distance"]
            adjacency_matrix = r1_total[i]["adjacency"]
            attributes_matrices_dict = attributes_distances[i]
            attributes_matrix = np.mean(
                np.stack(list(attributes_matrices_dict.values())), axis=0
            )
            attributes_matrix = attributes_matrix / attributes_matrix.max()
            distance_matrix = combine_alpha(
                structural_distance_matrix, attributes_matrix, alpha=0.5
            )
            affinity_matrix = self.distance_to_affinity(distance_matrix)
            affinity_gaussian = self.distance_to_affinity_gaussian(distance_matrix)

            # Distance matrices normalization
            distance_matrix2 = cdist(distance_matrix, distance_matrix)
            distance_matrix2 = distance_matrix2 / distance_matrix2.max()
            structural_matrix2 = cdist(
                structural_distance_matrix, structural_distance_matrix
            )
            structural_matrix2 = structural_matrix2 / structural_matrix2.max()
            attributes_matrix2 = cdist(attributes_matrix, attributes_matrix)
            attributes_matrix2 = attributes_matrix2 / attributes_matrix2.max()
            attributes_matrices_dict2 = {}
            for key in attributes_matrices_dict:
                val = attributes_matrices_dict[key]
                att_dist = cdist(val, val)
                attributes_matrices_dict2[key] = att_dist / att_dist.max()

            centroids_init_embedded, init_ot_embedded = self.start_initialisation(
                distance_matrix, embedded=True
            )
            _, init_ot_embedded2 = self.start_initialisation(
                distance_matrix2, embedded=True
            )
            _, init_ot_attributes = self.start_initialisation(
                attributes_matrix, embedded=True
            )

            srfgw_labels_mean, _ = self.srfgw_testing(
                structural_distance_matrix,
                attributes_matrices_dict,
                value=np.mean(structural_distance_matrix),
                fgw_config=fgw_config,
                g0=init_ot_embedded,
                g0_attributes=init_ot_embedded,
                medoid=medoid,
            )
            srfgw_labels_mean_att, _ = self.srfgw_testing(
                structural_distance_matrix,
                attributes_matrices_dict,
                value=np.mean(structural_distance_matrix),
                fgw_config=fgw_config,
                g0=init_ot_embedded,
                g0_attributes=init_ot_attributes,
                medoid=medoid,
            )
            srfgw_labels_max, _ = self.srfgw_testing(
                structural_distance_matrix,
                attributes_matrices_dict,
                value=np.max(structural_distance_matrix),
                fgw_config=fgw_config,
                g0=init_ot_embedded,
                g0_attributes=init_ot_embedded,
                medoid=medoid,
            )
            embedded_srfgw_labels_mean, _ = self.srfgw_testing(
                structural_matrix2,
                attributes_matrices_dict,
                value=np.mean(structural_matrix2),
                fgw_config=fgw_config,
                g0=init_ot_embedded2,
                g0_attributes=init_ot_embedded2,
                medoid=medoid,
            )
            embedded_srfgw_labels_max, _ = self.srfgw_testing(
                structural_matrix2,
                attributes_matrices_dict,
                value=np.max(structural_matrix2),
                fgw_config=fgw_config,
                g0=init_ot_embedded2,
                g0_attributes=init_ot_embedded2,
                medoid=medoid,
            )
            srfgw_labels_adj, _ = self.srfgw_testing(
                adjacency_matrix,
                attributes_matrices_dict,
                target_type="adjacency",
                fgw_config=fgw_config,
                g0="kmeans",
                g0_attributes=init_ot_attributes,
                medoid=medoid,
            )
            srfgw_labels_adj_mix, _ = self.srfgw_testing(
                adjacency_matrix,
                attributes_matrices_dict,
                target_type="adjacency",
                fgw_config=fgw_config,
                g0=init_ot_embedded,
                g0_attributes=init_ot_embedded,
                medoid=medoid,
            )

            cellules = [
                self.frechet_kmeans_testing(
                    structural_distance_matrix,
                    attributes_matrix,
                    centroids_init_embedded,
                    alpha=0.5,
                ),
                self.embedded_kmeans_testing(
                    structural_distance_matrix,
                    attributes_matrix,
                    centroids_init_embedded,
                    alpha=0.5,
                ),
                self.srgw_testing(
                    structural_distance_matrix,
                    attributes_matrix,
                    value=np.mean(distance_matrix),
                    g0=init_ot_embedded,
                    alpha=0.5,
                    embedded=False,
                ),
                self.srgw_testing(
                    structural_distance_matrix,
                    attributes_matrix,
                    value=np.mean(distance_matrix2),
                    g0=init_ot_embedded2,
                    alpha=0.5,
                    embedded=True,
                ),
                self.srgw_testing(
                    structural_distance_matrix,
                    attributes_matrix,
                    value=np.max(distance_matrix),
                    g0=init_ot_embedded,
                    alpha=0.5,
                    embedded=False,
                ),
                self.srgw_testing(
                    structural_distance_matrix,
                    attributes_matrix,
                    value=np.max(distance_matrix2),
                    g0=init_ot_embedded2,
                    alpha=0.5,
                    embedded=True,
                ),
                srfgw_labels_mean,
                srfgw_labels_mean_att,
                srfgw_labels_max,
                embedded_srfgw_labels_mean,
                embedded_srfgw_labels_max,
                srfgw_labels_adj,
                srfgw_labels_adj_mix,
                self.spectral_testing(affinity_matrix),
                self.spectral_testing(affinity_gaussian),
            ]
            labels = [
                "Frechet kmeans",
                "Embedded kmeans",
                "srGW (mean)",
                "Embedded srGW (mean)",
                "srGW (max)",
                "Embedded srGW (max)",
                "srfgw (mean) - init mix",
                "srfgw (mean) - init attributes",
                "srfgw (max)",
                "Embedded srfgw (mean)",
                "Embedded srfgw (max)",
                "srfgw adjacency - init attributes",
                "srfgw adjacency - init mix",
                "spectral",
                "spectral - gaussian",
            ]

            external_evaluation = pd.DataFrame()
            for method in range(len(cellules)):
                external_evaluation_tmp = ExternalEvaluation(
                    true_labels, cellules[method]
                ).get_df(labels[method])
                external_evaluation = pd.concat(
                    [external_evaluation, external_evaluation_tmp]
                )

            tmp_evaluation = external_evaluation
            parameters_tmp = {k: v[i] for k, v in self.parameters.items()}
            tmp_evaluation = eval_df(tmp_evaluation, parameters_tmp, infos)
            evaluation = pd.concat([evaluation, tmp_evaluation])

        if save:
            self.saving(evaluation)

        return evaluation
