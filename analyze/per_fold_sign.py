"""Per-fold sign ablation: val loss on one fixed validation set with GDSM's mixer replaced by its nearest unsigned
doubly stochastic matrix in a single fold at a time (all other folds as trained), plus the unablated baseline.
usage: python -m analyze.per_fold_sign --out_dir <dir> [--ckpt ckpt.pt --sequences 1000]"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyze.mixer_stats import load, val_batches  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True); parser.add_argument("--ckpt", default="ckpt.pt")
    parser.add_argument("--sequences", type=int, default=1000); parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--block", type=int, default=1024)
    args = parser.parse_args()
    model, checkpoint = load(args.out_dir, "cuda", args.ckpt)
    dataset = checkpoint["config"]["dataset"]; del checkpoint
    model.eval()
    batches = list(val_batches(dataset, args.block, args.batch, args.sequences // args.batch, "cuda", seed=2026))
    folds = [(name, module) for name, module in model.named_modules() if hasattr(module, "radial")]

    def evaluate():
        total = 0.0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for x, y in batches:
                total += model(x, y)[1].item()
        return total / len(batches)

    baseline = evaluate()
    print(f"baseline val {baseline:.4f} ({len(folds)} folds, {args.sequences} sequences)", flush=True)
    for index, (name, module) in enumerate(folds):
        module.ablate_sign = True
        loss = evaluate()
        module.ablate_sign = False
        print(f"fold {index + 1:2d} {name:28s} val {loss:.4f}  delta {loss - baseline:+.4f}", flush=True)


if __name__ == "__main__":
    main()
