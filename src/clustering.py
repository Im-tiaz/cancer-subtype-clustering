import numpy as np
from sklearn.cluster import KMeans

def run_kmeans(X, k, random_state=42):
    km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
    labels = km.fit_predict(X)
    return labels, km.cluster_centers_

