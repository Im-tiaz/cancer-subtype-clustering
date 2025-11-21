import numpy as np
import pandas as pd
from pathlib import Path

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def load_expression(expr_path):
    df = pd.read_csv(expr_path, index_col=0)
    return df

def load_labels(labels_path):
    lab = pd.read_csv(labels_path)
    return lab.set_index("sample_id")["cancer_type"]

def save_npy(path, arr):
    ensure_dir(Path(path).parent)
    np.save(path, arr)

def load_npy(path):
    return np.load(path, allow_pickle=True)

def save_csv(path, df):
    ensure_dir(Path(path).parent)
    df.to_csv(path)

