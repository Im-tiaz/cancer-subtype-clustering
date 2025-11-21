import numpy as np
from collections import defaultdict

def compute_label_centroids(X, true_labels):
    centroids = {}
    for lab in np.unique(true_labels):
        idx = np.where(true_labels == lab)[0]
        centroids[lab] = X[idx].mean(axis=0)
    return centroids

def dominant_cluster_centroid(X, assignments, cid):
    return X[assignments == cid].mean(axis=0)

def subtype_distance_flags(X, true_labels, assignments, purity_info):
    """
    purity_info: dict cid -> dominant_label (from purity calc)
    Returns list of (sample_id_idx, sc_label, dc_label, D1, D2, flag)
    """
    label_centroids = compute_label_centroids(X, true_labels)
    out = []
    for cid, dc_label in purity_info.items():
        dc_cent = dominant_cluster_centroid(X, assignments, cid)
        idx = np.where(assignments == cid)[0]
        for i in idx:
            sc_label = true_labels[i]
            if sc_label == dc_label:
                continue
            D1 = np.linalg.norm(X[i] - dc_cent)
            D2 = np.linalg.norm(X[i] - label_centroids[sc_label])
            flag = D1 < D2  # paper uses D1 ≪ D2 as evidence
            out.append((i, sc_label, dc_label, float(D1), float(D2), flag))
    return out

