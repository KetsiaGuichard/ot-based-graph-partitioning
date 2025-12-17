"""
Simulate functional data and histograms for attributed graph experiments.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy.interpolate import BSpline
from scipy.stats import dirichlet


class SimulatedFunctionalData:
    """Create Functional Data with a structure

    Args:
        k_group_ct (list(int)): number of nodes per class.
        epsilon (float): importance of perturbation inside a class.
        model_functions (array, optional): model of coefficients type. Default to None.
        length_out (int, optional): support length, default to 50
        support (tuple(int), optional): [min,max] value of support, default to (1, 16)
        degree (int, optional): Function degree. Default to 3.
    """

    def __init__(
        self,
        k_group_ct,
        epsilon: float,
        model_functions: list = None,
        degree: int = 3,
        length_out=73,
        support=(1, 16),
    ):
        self.k_group_ct = k_group_ct
        self.k = len(self.k_group_ct)
        self.epsilon = epsilon
        self.degree = degree
        self.x = self.get_x(support, length_out)
        self.knots = self.get_knots(support)
        self.n_basis = len(self.knots) + self.degree + 1
        self.models_funct = self.get_model_functions(model_functions)
        self.functions_df, self.functions_array = self.generate_functions()

    def get_x(self, support, length_out):
        """Get support
        Args:
            support (list(int)): min and max value of support
            length_out (int): support size
        Returns:
            list(float): support
        """
        return np.linspace(
            min(support) - self.degree - 1, max(support) + self.degree + 1, length_out
        )

    def get_knots(self, support):
        """Get knots
        Args:
            support (list(int)): min and max value of support
            degree (int): Function degree

        Returns:
            list(float): List of knots
        """
        return np.arange(
            min(support) - self.degree - 1, max(support) + self.degree + 1, 1
        )

    def get_model_functions(self, model_functions):
        """Generate k different coefficients set

        Args:
            model_functions (list(array)): if none, new models are generated

        Returns:
            list(array): k list of coefficients
        """
        if model_functions is not None:
            return model_functions
        model_functions = list([])
        for _ in range(self.k):
            model_functions.append(np.random.uniform(0, 1, self.n_basis))
        return model_functions

    def generate_functions(self):
        """Generate n different functions, with small differences inside groups.

        Returns :
            spline_df (data.frame): dataframe with function values over the support
            spline_array (np.array): same with np.array format
        """

        spline_data = []
        for i in range(self.k):
            theta_tmp = self.models_funct[i]
            for j in range(self.k_group_ct[i]):
                theta_alt = theta_tmp + np.random.uniform(
                    -self.epsilon, self.epsilon, len(theta_tmp)
                )
                spline_alt = BSpline(
                    self.knots, theta_alt, self.degree, extrapolate=False
                )(self.x)
                for xi, yi in zip(self.x, spline_alt):
                    spline_data.append([i + 1, j + 1, xi, yi])

        spline_df = pd.DataFrame(spline_data, columns=["group", "version", "x", "y"])
        spline_df = spline_df[~spline_df.y.isna()]
        spline_df["id"] = spline_df.groupby(["group", "version"]).ngroup()

        spline_array = spline_df.pivot(
            index=["group", "version"], columns="x", values="y"
        )
        spline_array = spline_array.reset_index(drop=True).to_numpy()

        return spline_df, spline_array

    def plot(self, maxi: int = 5):
        """Plot splines per group

        Args:
            max (int, optional): maximal number of splines plotted per group
        """
        g = sns.FacetGrid(
            data=self.functions_df[self.functions_df.version <= maxi],
            col="group",
            row="version",
            margin_titles=True,
            despine=False,
            hue="group",
            palette=sns.color_palette("Set2", self.k),
            sharex=True,
            sharey=True,
            height=0.9,
            aspect=4,
        )

        g.map(sns.lineplot, "x", "y")
        g.set_axis_labels("x", "y")
        g.set_titles(row_template="Version {row_name}", col_template="Group {col_name}")
        g.set(ylim=(0, 1))
        g.tight_layout()
        g.add_legend()
        plt.show()


class SimulatedHistograms:
    """Create Histograms with perturbations per group

    Args:
        k_group_ct (list(int)): number of nodes per class.
        models (np.array, optional): Initial models, default to None.
        support (int, optional): histogram support, default to 10.
        uni_parameters (tuple(int), optional): parameters of generative law. Default to (2,5).
        c (int, optional): dirichlet alpha. If high, low perturbations. Default to 1000.
    """

    def __init__(
        self, k_group_ct, models=None, support: int = 10, uni_parameters=(2, 5), c=1000
    ):
        self.k_group_ct = k_group_ct
        self.k = len(self.k_group_ct)
        self.support = support
        self.models = self.generate_k_models(uni_parameters, models)
        self.hist_df, self.hist_array = self.generate_alt_models(c)

    def generate_k_models(self, uni_parameters, models):
        """Generate k histograms, one for each group

        Args:
            uni_parameters (list(int)): parameters of uniform law
            models (np.array, optional): Initial models, default to None.

        Returns:
            np.array: k value of histograms of size support
        """
        if models is None:
            min_parameters = min(uni_parameters)
            max_parameters = max(uni_parameters)
            alphas = dirichlet.rvs(
                np.random.uniform(min_parameters, max_parameters, self.support),
                size=self.k,
            )
            return alphas
        return models

    def generate_alt_models(self, c):
        """Generate perturbations

        Args:
            c (int): dirichlet alpha. If high, low perturbations.

        Returns:
            hist_df (data.frame): dataframe with histogram values
            hist_array (np.array): same with np.array format
        """
        hist_df = []
        for i in range(self.k):
            alt = dirichlet.rvs(self.models[i] * c, size=self.k_group_ct[i])
            tmp = pd.DataFrame(
                alt.T, columns=[j + 1 for j in range(self.k_group_ct[i])]
            )
            tmp["x"] = np.arange(1, self.support + 1)
            tmp["group"] = i + 1
            hist_df.append(
                tmp.melt(id_vars=["x", "group"], var_name="version", value_name="value")
            )
        hist_df = pd.concat(hist_df)
        hist_df["id"] = hist_df.groupby(["group", "version"]).ngroup()

        grouped_hist = hist_df.groupby("id")["value"].apply(list)
        hist_array = np.array(grouped_hist.tolist())

        return hist_df, hist_array

    def plot(self, maxi: int = 5):
        """Plot histograms per group

        Args:
            max (int, optional): maximal number of splines plotted per group
        """
        g = sns.FacetGrid(
            data=self.hist_df[self.hist_df.version <= maxi],
            col="group",
            row="version",
            margin_titles=True,
            despine=False,
            hue="group",
            palette=sns.color_palette("Set2", self.k),
            sharex=True,
            sharey=True,
            height=1,
            aspect=4,
        )

        g.map(sns.barplot, "x", "value")
        g.set_axis_labels("x", "value")
        g.set_titles(col_template="Group {col_name}", row_template="Version {row_name}")
        g.tight_layout()
        plt.show()
