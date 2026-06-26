"""Graph-to-sequence transformation (the heart of GAGA).

For every node we walk its neighbourhood up to ``n_hops`` along each relation and
summarise the neighbours found at each hop into three group vectors:

* ``h_benign``  - mean feature of neighbours that are *training* nodes labelled 0
* ``h_fraud``   - mean feature of neighbours that are *training* nodes labelled 1
* ``h_unknown`` - mean feature of every other neighbour (val/test/unlabelled)

Only training labels are revealed, and a node never sees its own label, so no
label leakage occurs. The per-node sequence is::

    [ center | hop1(benign,fraud,unknown) | hop2(...) | ... ]   (repeated per relation)

with total length ``R * (1 + n_hops * (n_classes + 1))``.

Building the sequences is the one expensive, one-off step, so the result is
cached to disk and the work is spread across processes.
"""

import os
import multiprocessing as mp

import numpy as np
from scipy import sparse
from tqdm import tqdm


# Globals populated inside each worker process (inherited via fork). Keeping the
# big arrays here avoids re-pickling them for every task.
_CTX = {}


def _init_context(features, labels, adjacencies, train_mask,
                  n_hops, n_groups, grp_norm):
    _CTX.update(
        features=features,
        labels=labels,
        adjacencies=adjacencies,
        train_mask=train_mask,
        n_hops=n_hops,
        n_groups=n_groups,
        grp_norm=grp_norm,
        feat_dim=features.shape[1],
    )


def _neighbors(adj, nodes):
    """Unique 1-hop neighbours of a set of nodes under a single relation."""
    if len(nodes) == 0:
        return np.empty(0, dtype=np.int64)
    return np.unique(adj[nodes].indices)


def _aggregate(idx):
    """Mean feature over ``idx`` (zeros if empty), optionally group-normalised."""
    feat_dim = _CTX["feat_dim"]
    if len(idx) == 0:
        return np.zeros(feat_dim, dtype=np.float32)
    mean = _CTX["features"][idx].mean(axis=0)
    if _CTX["grp_norm"]:
        mean = mean / np.sqrt(len(idx))
    return mean.astype(np.float32)


def _group_block(center, neighbors):
    """Split neighbours into (benign, fraud, unknown) and mean each group."""
    if len(neighbors) == 0:
        return np.zeros((_CTX["n_groups"], _CTX["feat_dim"]), dtype=np.float32)

    train_mask, labels = _CTX["train_mask"], _CTX["labels"]
    train_nb = neighbors[train_mask[neighbors]]
    train_nb = train_nb[train_nb != center]  # never reveal the node's own label

    pos = train_nb[labels[train_nb] == 1]
    neg = train_nb[labels[train_nb] == 0]
    unknown = np.setdiff1d(neighbors, train_nb, assume_unique=True)

    return np.stack([_aggregate(neg), _aggregate(pos), _aggregate(unknown)], axis=0)


def _node_sequence(center):
    """Build the full grouped sequence for one node across all relations."""
    rows = []
    for adj in _CTX["adjacencies"]:
        rows.append(_CTX["features"][center][None, :])  # center token
        frontier = np.array([center], dtype=np.int64)
        for _ in range(_CTX["n_hops"]):
            frontier = _neighbors(adj, frontier)
            rows.append(_group_block(center, frontier))
    return np.concatenate(rows, axis=0)


def _process_block(bounds):
    start, end = bounds
    seq_len = len(_CTX["adjacencies"]) * (1 + _CTX["n_hops"] * _CTX["n_groups"])
    out = np.zeros((end - start, seq_len, _CTX["feat_dim"]), dtype=np.float32)
    for i, center in enumerate(range(start, end)):
        out[i] = _node_sequence(center)
    return start, out


def build_sequences(data, n_hops, grp_norm=False, add_self_loop=False,
                    n_workers=None, cache_file=None):
    """Turn a loaded dataset into the (N, S, E) sequence tensor used for training.

    Parameters
    ----------
    data : dict
        Output of :func:`gaga.data.load_dataset`.
    n_hops : int
        Number of hops K to expand around each node.
    cache_file : str, optional
        If given, load from here when present, otherwise save the result here.

    Returns
    -------
    np.ndarray of shape (n_nodes, R * (1 + n_hops * (n_classes + 1)), feat_dim).
    """
    if cache_file and os.path.exists(cache_file):
        print(f"Loading cached sequences from {cache_file}")
        return np.load(cache_file)

    features = data["features"]
    labels = data["labels"]
    adjacencies = data["adjacencies"]
    n_nodes = features.shape[0]
    n_groups = data["n_classes"] + 1

    if add_self_loop:
        eye = sparse.eye(n_nodes, format="csr", dtype=np.float32)
        adjacencies = [(adj + eye).tocsr() for adj in adjacencies]

    train_mask = np.zeros(n_nodes, dtype=bool)
    train_mask[data["train_ids"]] = True

    n_workers = n_workers or min(mp.cpu_count(), 8)
    seq_len = len(adjacencies) * (1 + n_hops * n_groups)
    print(f"Building sequences: nodes={n_nodes} seq_len={seq_len} "
          f"hops={n_hops} workers={n_workers}")

    init_args = (features, labels, adjacencies, train_mask, n_hops, n_groups, grp_norm)
    sequences = np.zeros((n_nodes, seq_len, features.shape[1]), dtype=np.float32)

    # Split nodes into one contiguous block per worker.
    block = n_nodes // n_workers + 1
    bounds = [(s, min(s + block, n_nodes)) for s in range(0, n_nodes, block)]

    if n_workers == 1:
        _init_context(*init_args)
        for b in tqdm(bounds, desc="sequences"):
            start, chunk = _process_block(b)
            sequences[start:start + chunk.shape[0]] = chunk
    else:
        with mp.Pool(n_workers, initializer=_init_context, initargs=init_args) as pool:
            for start, chunk in tqdm(pool.imap_unordered(_process_block, bounds),
                                     total=len(bounds), desc="sequences"):
                sequences[start:start + chunk.shape[0]] = chunk

    if cache_file:
        os.makedirs(os.path.dirname(cache_file) or ".", exist_ok=True)
        np.save(cache_file, sequences)
        print(f"Cached sequences to {cache_file}")

    return sequences
