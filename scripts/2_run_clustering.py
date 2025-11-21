import argparse, numpy as np, pandas as pd
from src.io_utils import load_npy, save_csv
from src.clustering import run_kmeans

def main(args):
    X = load_npy(args.X)
    labels, centers = run_kmeans(X, args.k)

    save_csv("results/clusters/kmeans_assignments.csv",
             pd.DataFrame({"cluster": labels}))
    np.save("results/clusters/centers.npy", centers)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--X", required=True)
    ap.add_argument("--k", type=int, default=20)
    main(ap.parse_args())

