# (Attributed) Graph Partitioning with semi-relaxed (Fused) Gromov-Wasserstein

This repository implements simulation and clustering methods based on semi-relaxed (Fused) Gromov–Wasserstein (sr(F)GW) and k‑Fréchet means for attributed and non-attributed graphs.

## Main functionalities
- Simulation of synthetic graphs (SBM-like mixing matrices) and node attributes (functional data and histograms).
- Distance computation: graph shortest-path distances, DTW for functional data, and Wasserstein for histograms.
- Clustering implementations: k‑Fréchet means, embedded k‑means, Semi-Relaxed GW (srGW) and Semi-Relaxed Fused GW (srFGW).
- Evaluation for automated experiments and metrics (internal/external).
- Example notebooks that reproduce experiments (located in top-level notebooks and methods_example.ipynb).

## Repository layout
- `src/`: project Python package
  - `src/simulation`: graph and attribute simulators (`graph.py`, `attributes.py`, `utils.py`)
  - `src/clustering`: clustering algorithms and helpers (`gw_clustering.py`, `kmeans.py`, `init_strategies.py`, `utils.py`)
  - `src/evaluation`: evaluation utilities (`utils.py`, `metrics.py`, `testing.py`)
  - `src/distances.py`: distance computation helpers
  - `src/shapes.py`: SBM creation with constraints 
- `methods_example.ipynb`: example workflows
- `simulations.ipynb`: simulations for (non)-attributed graph clustering evaluation
- `figure_table_generation.ipynb`: generation of figures used in the article

## Requirements & install
- Python 3.8+ recommended.
- Install dependencies (in a virtual environment):

  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt

Notable dependencies: numpy, scipy, pandas, scikit-learn, networkx, matplotlib, seaborn, POT (`ot`), dtaidistance, dtw.