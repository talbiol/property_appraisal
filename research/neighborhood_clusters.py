"""Cluster zip codes by crime and by income.

Standardises each zip code's statistics, projects them onto two principal
components and groups them into three k-means clusters, annotated by zip code.

Run with ``python research/neighborhood_clusters.py`` (needs the ``research`` extra).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from property_appraisal.config import RANDOM_STATE, TRAINING_DATA_DIR

N_CLUSTERS = 3
CLUSTER_LABELS = ["Low", "Medium", "High"]


def main() -> None:
    # Low crime is good (green); low income is bad (red).
    plot_clusters("crime", "Crime per zip code (PCA)", colors=["green", "grey", "red"])
    plot_clusters("income", "Income per zip code (PCA)", colors=["red", "grey", "green"])


def plot_clusters(dataset: str, title: str, colors: list[str]) -> None:
    stats = pd.read_csv(TRAINING_DATA_DIR / f"{dataset}.csv")
    metrics = stats.drop(columns="Zip_Code")

    scaled = StandardScaler().fit_transform(SimpleImputer(strategy="mean").fit_transform(metrics))
    projected = PCA(n_components=2).fit_transform(scaled)
    clusters = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=RANDOM_STATE).fit_predict(projected)

    # k-means cluster ids are arbitrary; rank them by mean raw level so colours mean
    # Low / Medium / High consistently.
    level = metrics.mean(axis=1).to_numpy()
    order = np.argsort([level[clusters == c].mean() for c in range(N_CLUSTERS)])
    rank = np.empty(N_CLUSTERS, dtype=int)
    rank[order] = np.arange(N_CLUSTERS)

    plt.figure(figsize=(10, 6))
    plt.scatter(projected[:, 0], projected[:, 1], c=[colors[rank[c]] for c in clusters], alpha=0.8)
    for label, color in zip(CLUSTER_LABELS, colors, strict=True):
        plt.scatter([], [], color=color, label=label)
    for zip_code, (x, y) in zip(stats["Zip_Code"], projected, strict=True):
        plt.annotate(str(zip_code), (x, y))
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title(title)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
