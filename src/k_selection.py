import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

def elbow_inertia(X, k_range, random_state=42):
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
        km.fit(X)
        inertias.append(km.inertia_)
    return np.array(inertias)

def silhouette_scores(X, k_range, random_state=42):
    scores = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
        labels = km.fit_predict(X)
        scores.append(silhouette_score(X, labels, metric="euclidean"))
    return np.array(scores)

def hierarchical_linkage_stabilization(X, method="ward"):
    # compute linkage distances; look for plateau around 18–22 groups
    Z = linkage(X, method=method, metric="euclidean")
    distances = Z[:, 2]  # merge distances
    return distances, Z

def centroid_stability(X, k, repeats=10, random_state=42):
    # average pairwise centroid disagreement across random inits
    centroids = []
    for r in range(repeats):
        km = KMeans(n_clusters=k, n_init=1, random_state=random_state+r)
        km.fit(X)
        centroids.append(km.cluster_centers_)
    centroids = np.stack(centroids, axis=0)
    # mean centroid variance
    return float(np.mean(np.var(centroids, axis=0)))

