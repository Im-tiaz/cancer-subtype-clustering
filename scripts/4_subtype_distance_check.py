import numpy as np, pandas as pd
from src.io_utils import load_npy, save_csv
from src.recursive_subclustering import cluster_purity
from src.subtype_distance import subtype_distance_flags

def main():
    X = load_npy("data/processed/X.npy")
    true_labels = load_npy("data/processed/labels.npy")
    refined = pd.read_csv("results/subclusters/refined_assignments.csv")["cluster"].values

    purity_info = {}
    for cid in np.unique(refined):
        idx = np.where(refined == cid)[0]
        purity, dom = cluster_purity(idx, true_labels)
        if dom is not None:
            purity_info[cid] = dom

    flags = subtype_distance_flags(X, true_labels, refined, purity_info)
    df = pd.DataFrame(flags, columns=["sample_idx","orig_label","dominant_label","D1","D2","D1_lt_D2"])
    save_csv("results/subclusters/subtype_distance_flags.csv", df)

if __name__ == "__main__":
    main()

