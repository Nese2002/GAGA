# GAGA — Group AGgregation enhanced TrAnsformer

A clean, dependency-light reimplementation of GAGA for **graph-based fraud
detection** on the YelpChi and Amazon review graphs.

The idea: instead of message passing, summarise each node's multi-hop
neighbourhood into a short **sequence of group vectors** and classify it with a
small **Transformer encoder**. Neighbours at every hop are split into three
groups — *benign*, *fraud* and *unknown* — using only training labels, which lets
the model exploit label information without leakage.

## Pipeline

```
data.py      load .mat graph (features, labels, per-relation adjacencies) + split
   |
sequence.py  per node: K-hop neighbourhood -> grouped sequence  (N, S, E)
   |
model.py     feature projection + hop/relation/group embeddings
             -> Transformer encoder -> concat per-relation center tokens -> MLP
   |
trainer.py   Adam + CrossEntropy, early stop on val AUC, threshold-moving on test
```

Sequence length is `R * (1 + n_hops * (n_classes + 1))` — for Yelp/Amazon with
3 relations, 2 hops and 2 classes that is `3 * (1 + 2*3) = 21` tokens.

## Install

```bash
pip install -r requirements.txt
```

No DGL required — the `.mat` files are downloaded directly and the graph is built
with SciPy sparse matrices.

## Run

```bash
python train.py --config configs/amazon.json            # ~11k nodes, fast
python train.py --config configs/yelp.json              # ~45k nodes
python train.py --config configs/amazon.json --n_runs 5  # mean +/- std
python train.py --config configs/amazon.json --plot      # also save a loss curve
```

The first run downloads the dataset (cached in `./data_cache`) and builds the
sequences (cached in `./seq_cache`); later runs reuse both.

## Outputs

Each run writes to `./checkpoints/`:

* `<dataset>_best.pt` — best-validation model weights (`torch.save` of the
  `state_dict`; reload with `model.load_state_dict(torch.load(path))`).
* `<dataset>_history.json` — per-epoch train loss and validation AUC / F1-macro.
* `<dataset>_curves.png` — training curve, when `--plot` is passed or
  `gaga.plot.plot_history(history)` is called.

## Reproduction targets

Same seed-717 split as the original benchmark. Approximate test results:

| Dataset | AUC   | F1-macro |
|---------|-------|----------|
| Amazon  | ~0.96 | ~0.91    |
| Yelp    | ~0.94 | ~0.90    |

## Colab

`notebooks/GAGA_Colab.ipynb` trains on a GPU. Copy this `Project/` folder into
your Google Drive, then open the notebook in Colab (Runtime → GPU) and run the
cells top to bottom.
