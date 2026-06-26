"""Command-line entry point for training GAGA.

Examples
--------
    python train.py --config configs/amazon.json
    python train.py --config configs/yelp.json --n_runs 5
    python train.py --config configs/amazon.json --max_epochs 3   # quick smoke test
"""

import argparse
import json

from gaga.trainer import train, train_multiple


def main():
    parser = argparse.ArgumentParser(description="Train GAGA for graph fraud detection.")
    parser.add_argument("--config", required=True, help="Path to a JSON config file.")
    parser.add_argument("--n_runs", type=int, default=1, help="Repeat training N times.")
    parser.add_argument("--max_epochs", type=int, default=None,
                        help="Override max_epochs (handy for a quick smoke test).")
    parser.add_argument("--n_workers", type=int, default=None,
                        help="Processes used when building sequences.")
    parser.add_argument("--cache_dir", default="./seq_cache",
                        help="Directory for cached sequence tensors.")
    parser.add_argument("--plot", action="store_true",
                        help="Save a training-curve PNG next to the checkpoint.")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    if args.max_epochs is not None:
        config["max_epochs"] = args.max_epochs
    if args.n_workers is not None:
        config["n_workers"] = args.n_workers

    if args.n_runs > 1:
        train_multiple(config, args.n_runs, cache_dir=args.cache_dir)
    else:
        _, history = train(config, cache_dir=args.cache_dir)
        if args.plot:
            from gaga.plot import plot_history
            plot_history(history, save_path=f"checkpoints/{config['dataset']}_curves.png")


if __name__ == "__main__":
    main()
