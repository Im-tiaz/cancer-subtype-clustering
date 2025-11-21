import numpy as np
from collections import Counter
from sklearn.cluster import KMeans

def cluster_purity(cluster_label_indices, true_labels):
    labs = true_labels[cluster_label_indices]
    if len(labs) == 0:
        return 0.0, None
    c = Counter(labs)
    dominant, count = c.most_common(1)[0]
    return count / len(labs), dominant

def recursive_refine(
    X, true_labels, base_assignments, purity_thr=0.95,
    min_cluster_size=30, max_recursions=6, random_state=42
):
    assignments = base_assignments.copy()
    next_cluster_id = assignments.max() + 1
    history = []

    for depth in range(max_recursions):
        changed = False
        for cid in np.unique(assignments):
            idx = np.where(assignments == cid)[0]
            purity, dom = cluster_purity(idx, true_labels)
            history.append((depth, cid, len(idx), purity, dom))

            if len(idx) < min_cluster_size or purity >= purity_thr:
                continue

            # subcluster this heterogeneous cluster
            subX = X[idx]
            # heuristic: split into 2 subclusters
            km = KMeans(n_clusters=2, n_init="auto", random_state=random_state+depth)
            sublabels = km.fit_predict(subX)

            # reassign with new global ids
            for s in [0, 1]:
                s_idx = idx[sublabels == s]
                assignments[s_idx] = next_cluster_id
                next_cluster_id += 1

            changed = True

        if not changed:
            break

    return assignments, history

