"""The GAGA Transformer classifier.

Each token of a node's sequence is a feature vector with a known *role*: which
hop it came from, which relation, and which group (benign / fraud / unknown). We
inject that structure through three learned embeddings, run a standard
Transformer encoder over the sequence, then read out the per-relation center
tokens and concatenate them for classification.
"""

import torch
import torch.nn as nn


class SequenceEncoder(nn.Module):
    """Project raw features and add hop / relation / group embeddings."""

    def __init__(self, feat_dim, emb_dim, n_hops, n_relations, n_classes, dropout):
        super().__init__()
        n_groups = n_classes + 1
        base_len = 1 + n_hops * n_groups  # tokens contributed by one relation

        self.input_proj = nn.Sequential(nn.Linear(feat_dim, emb_dim), nn.ReLU())
        self.hop_emb = nn.Embedding(n_hops + 1, emb_dim)
        self.relation_emb = nn.Embedding(n_relations, emb_dim)
        self.group_emb = nn.Embedding(n_groups, emb_dim)
        self.dropout = nn.Dropout(dropout)

        # Per-token index patterns; they depend only on the dimensions, so we
        # build them once and register as buffers (moved with .to(device)).
        hop_block = [0] + [h for h in range(1, n_hops + 1) for _ in range(n_groups)]
        grp_block = [n_classes] + list(range(n_groups)) * n_hops  # center -> "unknown"

        hop_idx, grp_idx, rel_idx = [], [], []
        for r in range(n_relations):
            hop_idx += hop_block
            grp_idx += grp_block
            rel_idx += [r] * base_len

        self.register_buffer("hop_idx", torch.tensor(hop_idx, dtype=torch.long))
        self.register_buffer("grp_idx", torch.tensor(grp_idx, dtype=torch.long))
        self.register_buffer("rel_idx", torch.tensor(rel_idx, dtype=torch.long))

    def forward(self, x):
        # x: (N, S, feat_dim) -> (N, S, emb_dim), then add the three encodings.
        tokens = self.input_proj(x)
        pos = (self.hop_emb(self.hop_idx)
               + self.relation_emb(self.rel_idx)
               + self.group_emb(self.grp_idx))
        return self.dropout(tokens + pos.unsqueeze(0))


class GAGA(nn.Module):
    def __init__(self, feat_dim, emb_dim, n_classes, n_hops, n_relations,
                 n_heads, ff_dim, n_layers, dropout=0.1):
        super().__init__()
        self.n_relations = n_relations
        self.base_len = 1 + n_hops * (n_classes + 1)

        self.encoder = SequenceEncoder(feat_dim, emb_dim, n_hops, n_relations,
                                       n_classes, dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)

        # Concatenate the center token of every relation, then classify.
        self.classifier = nn.Linear(emb_dim * n_relations, n_classes)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        h = self.encoder(x)                 # (N, S, E)
        h = self.transformer(h)             # (N, S, E)

        # Center token of each relation sits at multiples of base_len.
        centers = h[:, 0::self.base_len, :]            # (N, n_relations, E)
        agg = centers.reshape(centers.size(0), -1)     # (N, n_relations * E)
        return self.classifier(agg)
