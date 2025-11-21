import argparse, numpy as np
from src.io_utils import load_npy, save_csv, ensure_dir
from src.k_selection import elbow_inertia, silhouette_scores, hierarchical_linkage_stabilization, centroid_stability
import pandas as pd

def main(args):
    X = load_npy(args.X)
    k_range = range(args.kmin, args.kmax+1)

    inertias = elbow_inertia(X, k_range)
    sils = silhouette_scores(X, k_range)
    hdist, Z = hierarchical_linkage_stabilization(X, method=args.linkage)
    stabs = [centroid_stability(X, k, repeats=args.repeats) for k in k_range]

    ensure_dir("results/k_selection")
    df = pd.DataFrame({
        "k": list(k_range),
        "inertia": inertias,
        "silhouette": sils,
        "centroid_var": stabs
    })
    save_csv("results/k_selection/k_metrics.csv", df)

    np.save("results/k_selection/hierarchical_distances.npy", hdist)
    np.save("results/k_selection/linkage_Z.npy", Z)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--X", required=True)
    ap.add_argument("--kmin", type=int, default=10)
    ap.add_argument("--kmax", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--linkage", default="ward")
    main(ap.parse_args())

