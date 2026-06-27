"""Evaluation metrics and probability threshold selection."""

import numpy as np
import torch
from sklearn import metrics


@torch.no_grad()
def predict(model, loader, device, threshold=0.5):
    model.eval()
    probs, trues = [], []
    for seq, label in loader:
        logits = model(seq.to(device))
        if logits.dim() == 1:                 # guard against a size-1 last batch
            logits = logits.unsqueeze(0)
        probs.append(torch.sigmoid(logits).cpu())
        trues.append(label)

    prob = torch.cat(probs).numpy()[:, 1]
    y_true = torch.cat(trues).numpy()
    y_pred = (prob >= threshold).astype(np.int64)
    return y_true, prob, y_pred


def best_pr_threshold(y_true, y_prob):
    """Threshold on the precision-recall curve that maximises F1."""
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_prob)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(precision), where=(precision + recall) > 0)
    return thresholds[min(f1.argmax(), len(thresholds) - 1)]


def compute_metrics(y_true, y_prob, y_pred):
    """Collect the headline GAGA metrics into a dict."""
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    gmean = np.sqrt((tp / (tp + fn)) * (tn / (tn + fp))) if (tp + fn) and (tn + fp) else 0.0

    return {
        "auc": metrics.roc_auc_score(y_true, y_prob),
        "f1_macro": metrics.f1_score(y_true, y_pred, average="macro"),
        "gmean": gmean,
        "ap": metrics.average_precision_score(y_true, y_prob),
        "precision_1": metrics.precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_1": metrics.recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_fraud": metrics.f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_benign": metrics.f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        "recall_macro": metrics.recall_score(y_true, y_pred, average="macro"),
    }


def format_metrics(m):
    return (f"AUC={m['auc']:.4f} F1-macro={m['f1_macro']:.4f} "
            f"GMean={m['gmean']:.4f} AP={m['ap']:.4f} "
            f"P(1)={m['precision_1']:.4f} R(1)={m['recall_1']:.4f}")
