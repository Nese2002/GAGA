"""Loading of the YelpChi and Amazon fraud datasets"""

import os
import zipfile
import urllib.request
import numpy as np
from scipy import io as sio
from scipy import sparse

_DATASETS = {
    "yelp": {
        "url": "https://data.dgl.ai/dataset/FraudYelp.zip",
        "mat": "YelpChi.mat",
        "relations": ["net_rsr", "net_rtr", "net_rur"],
    },
    "amazon": {
        "url": "https://data.dgl.ai/dataset/FraudAmazon.zip",
        "mat": "Amazon.mat",
        "relations": ["net_upu", "net_usu", "net_uvu"],
    },
}

# The first 3305 Amazon nodes are unlabelled and excluded from every split.
_AMAZON_LABELLED_START = 3305


def _download(name, cache_dir):
    """Fetch and unzip the ``.mat`` file for ``name``; returns its local path."""
    spec = _DATASETS[name]
    os.makedirs(cache_dir, exist_ok=True)
    mat_path = os.path.join(cache_dir, spec["mat"])
    if os.path.exists(mat_path):
        return mat_path

    zip_path = os.path.join(cache_dir, name + ".zip")
    print(f"Downloading {name} from {spec['url']} ...")
    urllib.request.urlretrieve(spec["url"], zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(cache_dir)
    os.remove(zip_path)
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"{spec['mat']} not found after extracting {zip_path}")
    return mat_path


def _row_normalize(feats):
    """Row-normalize the feature matrix """
    rowsum = np.asarray(feats.sum(axis=1)).flatten() + 0.01
    inv = np.power(rowsum, -1)
    inv[np.isinf(inv)] = 0.0
    return (sparse.diags(inv).dot(feats)).astype(np.float32)


def _split_indices(n_nodes, name, train_size, val_size, seed):
    """Reproduce the benchmark split: shuffle once, then slice train/test/val."""
    index = np.arange(n_nodes)
    if name == "amazon":
        index = np.arange(_AMAZON_LABELLED_START, n_nodes)

    index = np.random.RandomState(seed).permutation(index)
    n = len(index)
    train_end = int(train_size * n)
    val_start = n - int(val_size * n)

    train_ids = index[:train_end]
    val_ids = index[val_start:]
    test_ids = index[train_end:val_start]
    return train_ids, val_ids, test_ids


def load_dataset(name, root="./data_cache", train_size=0.4, val_size=0.1,
                 seed=42, norm_feat=True):
    """Load a fraud graph"""
    if name not in _DATASETS:
        raise ValueError(f"unknown dataset {name!r}, expected one of {list(_DATASETS)}")

    mat_path = _download(name, root)
    mat = sio.loadmat(mat_path)

    features = mat["features"]
    features = features.todense() if sparse.issparse(features) else features
    features = np.asarray(features, dtype=np.float32)
    if norm_feat:
        features = _row_normalize(features)
    features = np.asarray(features, dtype=np.float32)

    labels = np.asarray(mat["label"]).squeeze().astype(np.int64)

    adjacencies = [mat[rel].tocsr().astype(np.float32)
                   for rel in _DATASETS[name]["relations"]]

    train_ids, val_ids, test_ids = _split_indices(
        features.shape[0], name, train_size, val_size, seed)

    print(f"[{name}] nodes={features.shape[0]} feat_dim={features.shape[1]} "
          f"relations={len(adjacencies)}")
    print(f"  train={len(train_ids)} (pos={int(labels[train_ids].sum())})  "
          f"val={len(val_ids)}  test={len(test_ids)}")

    return {
        "features": features,
        "labels": labels,
        "adjacencies": adjacencies,
        "train_ids": train_ids.astype(np.int64),
        "val_ids": val_ids.astype(np.int64),
        "test_ids": test_ids.astype(np.int64),
        "n_relations": len(adjacencies),
        "n_classes": 2,
        "feat_dim": features.shape[1],
    }
