import argparse
import numpy as np
import pandas as pd
from src.io_utils import load_expression, load_labels, save_npy, save_csv
from src.preprocess import preprocess_expression

def main(args):
    expr = load_expression(args.expr)
    labels = load_labels(args.labels)

    expr = expr.loc[labels.index]  # align
    X, genes, samples = preprocess_expression(expr)

    save_npy("data/processed/X.npy", X)
    save_npy("data/processed/genes.npy", np.array(genes))
    save_npy("data/processed/samples.npy", np.array(samples))
    save_npy("data/processed/labels.npy", labels.values)

    save_csv("data/processed/expr_preprocessed.csv",
             pd.DataFrame(X, index=samples, columns=genes))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--expr", required=True)
    ap.add_argument("--labels", required=True)
    main(ap.parse_args())

