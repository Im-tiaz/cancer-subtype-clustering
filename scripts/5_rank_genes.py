import numpy as np, pandas as pd
from src.io_utils import load_npy, save_csv, ensure_dir
from src.recursive_subclustering import cluster_purity
from src.gene_importance import top_genes_logreg, top_genes_rf, top_genes_pca

def main(top_n=50):
    X = load_npy("data/processed/X.npy")
    genes = load_npy("data/processed/genes.npy").tolist()
    true_labels = load_npy("data/processed/labels.npy")
    refined = pd.read_csv("results/subclusters/refined_assignments.csv")["cluster"].values

    ensure_dir("results/gene_panels")

    for cid in np.unique(refined):
        idx = np.where(refined == cid)[0]
        purity, dom = cluster_purity(idx, true_labels)
        if dom is None:
            continue

        y_bin = (true_labels[idx] == dom).astype(int)
        subX = X[idx]

        lr_df = top_genes_logreg(subX, y_bin, genes, top_n=top_n)
        rf_df = top_genes_rf(subX, y_bin, genes, top_n=top_n)
        pca_df = top_genes_pca(subX, genes, top_n=top_n)

        lr_df["method"] = "logreg"
        rf_df["method"] = "rf"
        pca_df["method"] = "pca"

        out = pd.concat([lr_df, rf_df, pca_df], ignore_index=True)
        save_csv(f"results/gene_panels/cluster_{cid}_{dom}_topgenes.csv", out)

if __name__ == "__main__":
    main()

