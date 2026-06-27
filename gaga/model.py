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
    """Project raw features and add hop / relation / group embeddings.

    The three learnable encodings can each be switched off independently
    (``use_hop`` / ``use_relation`` / ``use_group``) for the §5.7.2 ablation.
    When the sequences were built without group aggregation (``group_agg=False``)
    each hop is a single mean token, so the group encoding carries no meaning and
    is forced off.
    """

    def __init__(self, feat_dim, emb_dim, n_hops, n_relations, n_classes, dropout,
                 group_agg=True, use_hop=True, use_relation=True, use_group=True):
        super().__init__()
        groups_per_hop = (n_classes + 1) if group_agg else 1
        base_len = 1 + n_hops * groups_per_hop  # tokens contributed by one relation

        # the group encoding only exists when neighbours are split into groups
        use_group = use_group and group_agg
        self.use_hop, self.use_relation, self.use_group = use_hop, use_relation, use_group

        self.input_proj = nn.Sequential(nn.Linear(feat_dim, emb_dim), nn.ReLU())
        self.hop_emb = nn.Embedding(n_hops + 1, emb_dim) if use_hop else None
        self.relation_emb = nn.Embedding(n_relations, emb_dim) if use_relation else None
        self.group_emb = nn.Embedding(groups_per_hop, emb_dim) if use_group else None
        self.dropout = nn.Dropout(dropout)

        # Per-token index patterns; they depend only on the dimensions, so we
        # build them once and register as buffers (moved with .to(device)).
        hop_block = [0] + [h for h in range(1, n_hops + 1) for _ in range(groups_per_hop)]
        if group_agg:
            grp_block = [n_classes] + list(range(groups_per_hop)) * n_hops  # center -> "unknown"
        else:
            grp_block = [0] * base_len

        hop_idx, grp_idx, rel_idx = [], [], []
        for r in range(n_relations):
            hop_idx += hop_block
            grp_idx += grp_block
            rel_idx += [r] * base_len

        self.register_buffer("hop_idx", torch.tensor(hop_idx, dtype=torch.long))
        self.register_buffer("grp_idx", torch.tensor(grp_idx, dtype=torch.long))
        self.register_buffer("rel_idx", torch.tensor(rel_idx, dtype=torch.long))

    def forward(self, x):
        # x: (N, S, feat_dim) -> (N, S, emb_dim), then add the enabled encodings.
        tokens = self.input_proj(x)
        pos = None
        if self.use_hop:
            pos = self.hop_emb(self.hop_idx)
        if self.use_relation:
            r = self.relation_emb(self.rel_idx)
            pos = r if pos is None else pos + r
        if self.use_group:
            g = self.group_emb(self.grp_idx)
            pos = g if pos is None else pos + g
        if pos is not None:
            tokens = tokens + pos.unsqueeze(0)
        return self.dropout(tokens)


class GAGA(nn.Module):
    """GAGA classifier with switchable components for the §5.7 ablation.

    Parameters that control the ablations
    -------------------------------------
    group_agg : bool
        Whether the input sequences use group aggregation (3 group tokens per hop)
        or a single mean token per hop. Must match how ``build_sequences`` was run.
    use_hop / use_relation / use_group : bool
        Toggle each learnable encoding (Table 5).
    backbone : {'transformer', 'mlp', 'none'}
        The sequence encoder applied before readout. ``'transformer'`` is GAGA;
        ``'mlp'`` is a token-wise MLP (no cross-token attention); ``'none'`` skips
        it entirely. Used to isolate the contribution of self-attention (§5.7.1).
    """

    def __init__(self, feat_dim, emb_dim, n_classes, n_hops, n_relations,
                 n_heads, ff_dim, n_layers, dropout=0.1,
                 group_agg=True, use_hop=True, use_relation=True, use_group=True,
                 backbone="transformer"):
        super().__init__()
        self.n_relations = n_relations
        groups_per_hop = (n_classes + 1) if group_agg else 1
        self.base_len = 1 + n_hops * groups_per_hop
        self.backbone = backbone

        self.encoder = SequenceEncoder(feat_dim, emb_dim, n_hops, n_relations,
                                       n_classes, dropout, group_agg=group_agg,
                                       use_hop=use_hop, use_relation=use_relation,
                                       use_group=use_group)

        if backbone == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=emb_dim, nhead=n_heads, dim_feedforward=ff_dim,
                dropout=dropout, batch_first=True)
            self.sequence_model = nn.TransformerEncoder(layer, num_layers=n_layers)
        elif backbone == "mlp":
            # token-wise feed-forward, applied independently per token (no attention)
            self.sequence_model = nn.Sequential(
                nn.Linear(emb_dim, ff_dim), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(ff_dim, emb_dim))
        elif backbone == "none":
            self.sequence_model = nn.Identity()
        else:
            raise ValueError(f"unknown backbone {backbone!r}")

        # Concatenate the center token of every relation, then classify.
        self.classifier = nn.Linear(emb_dim * n_relations, n_classes)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        h = self.encoder(x)                 # (N, S, E)
        h = self.sequence_model(h)          # (N, S, E)

        # Center token of each relation sits at multiples of base_len.
        centers = h[:, 0::self.base_len, :]            # (N, n_relations, E)
        agg = centers.reshape(centers.size(0), -1)     # (N, n_relations * E)
        return self.classifier(agg)
