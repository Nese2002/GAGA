"""Plotting helpers for training curves."""

import matplotlib.pyplot as plt


def plot_history(history, save_path=None, show=False):
    epochs = history["epoch"]
    fig, ax_loss = plt.subplots(figsize=(7, 4.5))

    # Training loss on the left axis.
    ax_loss.plot(epochs, history["train_loss"], color="tab:red", label="train loss")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("train loss", color="tab:red")
    ax_loss.tick_params(axis="y", labelcolor="tab:red")

    # Validation metrics on the right axis.
    ax_metric = ax_loss.twinx()
    ax_metric.plot(epochs, history["val_auc"], color="tab:blue", label="val AUC")
    ax_metric.plot(epochs, history["val_f1_macro"], color="tab:green",
                   linestyle="--", label="val F1-macro")
    ax_metric.set_ylabel("validation metric", color="tab:blue")
    ax_metric.tick_params(axis="y", labelcolor="tab:blue")

    # Merge legends from both axes.
    lines = ax_loss.get_lines() + ax_metric.get_lines()
    ax_loss.legend(lines, [l.get_label() for l in lines], loc="center right")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Saved training curve to {save_path}")
    if show:
        plt.show()
    return fig
