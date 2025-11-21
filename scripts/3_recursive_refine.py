import argparse, numpy as np, pandas as pd
from collections import Counter
from src.io_utils import load_npy, save_csv
from src.recursive_subclustering import recursive_refine, cluster_purity

def main(args):
    X = load_npy(args.X)
    true_labels = load_npy(args.labels)
    base = pd.read_csv("results/clusters/kmeans_assignments.csv")["cluster"].values

    refined, hist = recursive_refine(
        X, true_labels, base,
        purity_thr=args.purity,
        min_cluster_size=args.min_size,
        max_recursions=args.max_rec
    )

    save_csv("results/subclusters/refined_assignments.csv",
             pd.DataFrame({"cluster": refined}))

    hist_df = pd.DataFrame(hist, columns=["depth","cluster_id","size","purity","dominant_label"])
    save_csv("results/subclusters/refine_history.csv", hist_df)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--X", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--purity", type=float, default=0.95)
    ap.add_argument("--min_size", type=int, default=30)
    ap.add_argument("--max_rec", type=int, default=6)
    main(ap.parse_args())

