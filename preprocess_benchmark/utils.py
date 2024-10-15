
import os.path
import scipy.io as sio
import scipy.sparse as sp
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfTransformer


def aug_normalized_adjacency(adj, add_loops=True):
    if add_loops:
        adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def row_normalize(mx, add_loops=True):
    if add_loops:
        mx = mx + sp.eye(mx.shape[0])
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def is_close(a, b, c):
    return np.abs(a - b) < c


def read_dataset(dataset):
    data = sio.loadmat(os.path.join('../../real_data/data', f'{dataset}.mat'))
    features = data['fea'].astype(float)
    adj = data['W']
    adj = adj.astype(float)
    if not sp.issparse(adj):
        adj = sp.csc_matrix(adj)
    if sp.issparse(features):
        features = features.toarray()
    labels = data['gnd'].reshape(-1) - 1
    n_classes = len(np.unique(labels))
    return adj, features, labels, n_classes


def preprocess_dataset(adj, features, row_norm=True, sym_norm=True, feat_norm=True, tf_idf=False):
    if sym_norm:
        adj = aug_normalized_adjacency(adj, True)
    if row_norm:
        adj = row_normalize(adj, True)

    if tf_idf:
        features = TfidfTransformer().fit_transform(features).toarray()
    if feat_norm:
        features = normalize(features)
    return adj, features

