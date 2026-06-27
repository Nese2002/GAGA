"""Training loop, early stopping and final evaluation."""

import copy
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset

from .data import load_dataset
from .sequence import build_sequences
from .model import GAGA
from .metrics import predict, compute_metrics, best_pr_threshold, format_metrics


def _set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _make_loaders(sequences, labels, data, batch_size):
    dataset = TensorDataset(torch.from_numpy(sequences),
                            torch.from_numpy(labels).long())
    loaders = {}
    for split in ("train_ids", "val_ids", "test_ids"):
        subset = Subset(dataset, data[split])
        loaders[split] = DataLoader(subset, batch_size=batch_size,
                                    shuffle=(split == "train_ids"))
    return loaders["train_ids"], loaders["val_ids"], loaders["test_ids"]


def _prepare(config, cache_dir):
    """Build the data split, (cached) sequences, loaders and model for a config.

    Shared by ``train`` and ``evaluate`` so both see the identical split/sequences.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _set_seed(config.get("init_seed", config["seed"]))

    data = load_dataset(config["dataset"], root=config.get("data_root", "./data_cache"),
                        train_size=config["train_size"], val_size=config["val_size"],
                        seed=config["seed"], norm_feat=config.get("norm_feat", True))

    group_agg = config.get("group_agg", True)
    reveal_ids = None
    label_tag = ""
    label_rate = config.get("label_rate")
    if label_rate is not None and label_rate < config["train_size"]:
        train_ids = data["train_ids"]
        keep = int(round(len(train_ids) * label_rate / config["train_size"]))
        reveal_ids = np.random.RandomState(config["seed"]).permutation(train_ids)[:keep]
        label_tag = f"_lr{label_rate}"

    cache_file = os.path.join(
        cache_dir,
        f"{config['dataset']}_h{config['n_hops']}_"
        f"{config['train_size']}_{config['val_size']}_{config['seed']}"
        f"{'_grpnorm' if config.get('grp_norm') else ''}"
        f"{'' if group_agg else '_noga'}{label_tag}.npy")
    sequences = build_sequences(
        data, n_hops=config["n_hops"], grp_norm=config.get("grp_norm", False),
        add_self_loop=config.get("add_self_loop", False),
        group_agg=group_agg, reveal_ids=reveal_ids,
        n_workers=config.get("n_workers"), cache_file=cache_file)

    loaders = _make_loaders(sequences, data["labels"], data, config["batch_size"])

    model = GAGA(feat_dim=data["feat_dim"], emb_dim=config["emb_dim"],
                 n_classes=data["n_classes"], n_hops=config["n_hops"],
                 n_relations=data["n_relations"], n_heads=config["n_heads"],
                 ff_dim=config["ff_dim"], n_layers=config["n_layers"],
                 dropout=config["dropout"],
                 group_agg=group_agg,
                 use_hop=config.get("use_hop", True),
                 use_relation=config.get("use_relation", True),
                 use_group=config.get("use_group", True),
                 backbone=config.get("backbone", "transformer")).to(device)

    return device, data, loaders, model


def train(config, cache_dir="./seq_cache", ckpt_dir="./checkpoints"):
    """Run one full training + evaluation cycle from a config dict    """
    device, data, loaders, model = _prepare(config, cache_dir)
    train_loader, val_loader, test_loader = loaders

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"],
                                 weight_decay=config["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    os.makedirs(ckpt_dir, exist_ok=True)
    run_name = config.get("run_name", config["dataset"])
    ckpt_path = os.path.join(ckpt_dir, f"{run_name}_best.pt")

    best_auc, best_state, best_epoch, patience = -1.0, None, -1, 0
    history = {"epoch": [], "train_loss": [], "val_auc": [], "val_f1_macro": []}
    print(f"Training GAGA on {config['dataset']} ({device})")

    for epoch in range(config["max_epochs"]):
        model.train()
        total_loss = 0.0
        for seq, label in train_loader:
            seq, label = seq.to(device), label.to(device)
            optimizer.zero_grad()
            loss = criterion(model(seq), label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % config.get("eval_interval", 1) == 0:
            avg_loss = total_loss / len(train_loader)
            y_true, y_prob, y_pred = predict(model, val_loader, device)
            val = compute_metrics(y_true, y_prob, y_pred)
            print(f"epoch {epoch:>3d} | loss {avg_loss:.4f} | val {format_metrics(val)}")

            history["epoch"].append(epoch)
            history["train_loss"].append(avg_loss)
            history["val_auc"].append(val["auc"])
            history["val_f1_macro"].append(val["f1_macro"])

            if val["auc"] > best_auc:
                best_auc, best_epoch = val["auc"], epoch
                best_state = copy.deepcopy(model.state_dict())
                torch.save(best_state, ckpt_path)  # persist the best model
                patience = 0
            else:
                patience += 1
                if config.get("early_stop", 0) and patience >= config["early_stop"]:
                    print(f"Early stopping at epoch {epoch} (best val AUC {best_auc:.4f}).")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\nBest val AUC {best_auc:.4f} at epoch {best_epoch}. "
          f"Saved checkpoint to {ckpt_path}")

    history_path = os.path.join(ckpt_dir, f"{run_name}_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f)

    # Pick the operating threshold on validation, then evaluate the test set.
    v_true, v_prob, _ = predict(model, val_loader, device)
    thres = best_pr_threshold(v_true, v_prob)
    t_true, t_prob, t_pred = predict(model, test_loader, device, threshold=thres)
    test = compute_metrics(t_true, t_prob, t_pred)
    print(f"\nTest @ thres={thres:.3f}: {format_metrics(test)}")
    print(f"  f1-fraud={test['f1_fraud']:.4f} f1-benign={test['f1_benign']:.4f} "
          f"recall-macro={test['recall_macro']:.4f}")
    return test, history


def evaluate(config, ckpt_path=None, cache_dir="./seq_cache", ckpt_dir="./checkpoints"):
    """Load a saved checkpoint and evaluate it on the test set — no training.

    Rebuilds the same split / cached sequences as ``train`` (so the test set matches
    what the checkpoint was trained on), reconstructs the model, loads the weights,
    picks the operating threshold on validation and reports the test metrics.

    ``ckpt_path`` defaults to ``{ckpt_dir}/{run_name or dataset}_best.pt``.
    Returns the test-metrics dict.
    """
    device, _, loaders, model = _prepare(config, cache_dir)
    _, val_loader, test_loader = loaders

    run_name = config.get("run_name", config["dataset"])
    if ckpt_path is None:
        ckpt_path = os.path.join(ckpt_dir, f"{run_name}_best.pt")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"Loaded checkpoint {ckpt_path}")

    # Same protocol as train(): tune threshold on val, then score the test set.
    v_true, v_prob, _ = predict(model, val_loader, device)
    thres = best_pr_threshold(v_true, v_prob)
    t_true, t_prob, t_pred = predict(model, test_loader, device, threshold=thres)
    test = compute_metrics(t_true, t_prob, t_pred)
    print(f"Test @ thres={thres:.3f}: {format_metrics(test)}")
    return test


def train_multiple(config, n_runs, cache_dir="./seq_cache"):
    """Train ``n_runs`` times (varying only the init seed) and report mean +/- std.

    The data split and cached sequences stay fixed (they depend on ``config['seed']``);
    each run uses a different ``init_seed`` so weight init and batch order differ.
    Returns ``(runs, summary)`` where ``summary[metric] = (mean, std)``.
    """
    base = config.get("init_seed", config["seed"])
    runs = []
    for i in range(n_runs):
        print(f"\n----- run {i + 1}/{n_runs} (init_seed={base + i}) -----")
        runs.append(train({**config, "init_seed": base + i}, cache_dir=cache_dir)[0])

    summary = {}
    print("\n===== summary over", n_runs, "runs =====")
    for k in runs[0]:
        vals = np.array([r[k] for r in runs])
        mean, std = vals.mean(), vals.std(ddof=1 if n_runs > 1 else 0)
        summary[k] = (mean, std)
        print(f"{k:14s}= {mean:.4f} +/- {std:.4f}")
    return runs, summary
