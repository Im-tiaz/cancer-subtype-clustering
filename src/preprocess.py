import numpy as np
from sklearn.preprocessing import StandardScaler

def log1p_transform(X):
    return np.log2(X + 1.0)

def zscore(X):
    return StandardScaler(with_mean=True, with_std=True).fit_transform(X)

def preprocess_expression(df):
    X = df.values.astype(float)
    X = log1p_transform(X)
    X = zscore(X)
    return X, df.columns.to_list(), df.index.to_list()

