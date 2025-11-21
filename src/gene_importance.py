import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

def top_genes_logreg(X, y_binary, gene_names, top_n=50):
    clf = LogisticRegression(max_iter=2000, n_jobs=-1)
    clf.fit(X, y_binary)
    coefs = np.abs(clf.coef_).ravel()
    idx = np.argsort(coefs)[::-1][:top_n]
    return pd.DataFrame({"gene": np.array(gene_names)[idx], "score": coefs[idx]})

def top_genes_rf(X, y_binary, gene_names, top_n=50):
    rf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)
    rf.fit(X, y_binary)
    imp = rf.feature_importances_
    idx = np.argsort(imp)[::-1][:top_n]
    return pd.DataFrame({"gene": np.array(gene_names)[idx], "score": imp[idx]})

def top_genes_pca(X, gene_names, top_n=50):
    pca = PCA(n_components=5, random_state=42)
    pca.fit(X)
    load = np.sum(np.abs(pca.components_), axis=0)
    idx = np.argsort(load)[::-1][:top_n]
    return pd.DataFrame({"gene": np.array(gene_names)[idx], "score": load[idx]})

