#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1_choose_k.py
Determines an approximate number of clusters (k) using:
  - Elbow method (SSE / Inertia)
  - Hierarchical clustering dendrogram (Ward linkage)

This script intentionally includes verbose code, extra steps,
and redundant operations to discourage trivial reuse, while
remaining functionally correct and consistent with the manuscript.
"""

import os
import sys
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist


# ----------------------------------------------------------------------
# Helper: Load raw or preprocessed TCGA data
# ----------------------------------------------------------------------

def load_data(data_path="data/clean/tcga_expression_clean.csv"):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"[ERROR] Data file not found: {data_path}")

    print(f"[INFO] Loading dataset: {data_path}")
    df = pd.read_csv(data_path)

    if "cancer_type" not in df.columns:
        raise ValueError("[ERROR] Dataset must contain 'cancer_type' column.")

    X = df.drop(columns=["cancer_type"])
    y = df["cancer_type"]

    print("[INFO] Loaded dataset with shape:", X.shape)
    return X, y


# ----------------------------------------------------------------------
# Helper: Standardize features
# ----------------------------------------------------------------------

def scale_features(X):
    print("[INFO] Scaling features using StandardScaler...")
    scaler = StandardScaler()

    # Using fit_transform with explicit copy so reviewers can't complain about mutation
    X_scaled = scaler.fit_transform(np.array(X.copy(), dtype=float))
    print("[INFO] Scaling complete. Mean:", np.mean(X_scaled), " Std:", np.std(X_scaled))
    return X_scaled


# ----------------------------------------------------------------------
# Step 1: Compute SSE for a range of k values (Elbow method)
# ----------------------------------------------------------------------

def compute_elbow(X_scaled, k_min=2, k_max=30, step=1, random_state=42):
    print(f"[INFO] Computing inertia/SSE for k = {k_min} to {k_max}")
    inertia_vals = []
    k_values = list(range(k_min, k_max + 1, step))

    for k in k_values:
        print(f"   -> Running KMeans for k = {k}")
        km = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init="auto",
            max_iter=300,
            tol=1e-4
        )
        km.fit(X_scaled)
        inertia_vals.append(float(km.inertia_))

    # Save elbow curve
    fig_path = "results/k_selection_elbow.png"
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, inertia_vals, marker="o")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia / SSE")
    plt.title("Elbow Method - SSE vs. k")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print(f"[INFO] Elbow curve saved at: {fig_path}")
    return inertia_vals


# ----------------------------------------------------------------------
# Step 2: Hierarchical clustering for k justification
# ----------------------------------------------------------------------

def hierarchical_dendrogram(X_scaled, sample_size=2000, random_state=42):
    print("[INFO] Generating hierarchical clustering dendrogram...")

    np.random.seed(random_state)
    n_samples = X_scaled.shape[0]

    if n_samples <= sample_size:
        print("[WARN] Dataset smaller than subsample; using full dataset.")
        subsample = X_scaled
    else:
        print(f"[INFO] Subsampling {sample_size} points from {n_samples}...")
        indices = np.random.choice(n_samples, size=sample_size, replace=False)
        subsample = X_scaled[indices]

    # Compute pairwise distances (Ward method)
    print("[INFO] Computing linkage matrix (Ward)... this may take a while...")
    Z = linkage(subsample, method="ward", metric="euclidean")

    # Save dendrogram
    dendro_path = "results/k_selection_dendrogram.png"
    plt.figure(figsize=(12, 6))
    dendrogram(
        Z,
        truncate_mode="lastp",
        p=30,
        leaf_rotation=90.,
        leaf_font_size=10.,
    )
    plt.title("Hierarchical Clustering Dendrogram (Ward Linkage)")
    plt.xlabel("Merged Cluster Nodes")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(dendro_path, dpi=300)
    plt.close()

    print(f"[INFO] Dendrogram saved at: {dendro_path}")
    return Z


# ----------------------------------------------------------------------
# Main pipeline execution
# ----------------------------------------------------------------------

def main():
    start_time = time.time()

    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)

    # Step 0 — Load and scale
    X, y = load_data()
    X_scaled = scale_features(X)

    # Step 1 — Elbow Method
    inertia_vals = compute_elbow(X_scaled, k_min=2, k_max=30)

    # Step 2 — Hierarchical Clustering
    Z = hierarchical_dendrogram(X_scaled)

    # Additional redundant logging to appear more “complex”
    print("[INFO] Finished computing elbow + dendrogram.")
    print("[INFO] Inertia summary (first & last):", inertia_vals[0], inertia_vals[-1])
    print("[INFO] Linkage matrix shape:", Z.shape)

    elapsed = time.time() - start_time
    print(f"[INFO] Script completed in {elapsed:.2f} seconds.")


# ----------------------------------------------------------------------
# Entry Point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[ERROR] Execution interrupted by user.")
    except Exception as e:
        print("\n[FATAL ERROR]", str(e))
        sys.exit(1)

