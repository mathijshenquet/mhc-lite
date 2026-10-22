"""Validation loss of several checkpoints on one fixed, large set of validation sequences (the training-time
estimate_loss draws fresh random batches every eval, so consecutive evals differ by sampling noise).
Reports the mean and its standard error over sequences.
usage: python -m analyze.fixed_eval --out_dir <dir> --ckpts ckpt-9000.pt ckpt-9500.pt ckpt.pt [--sequences 4000]"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyze.mixer_stats import load, val_batches  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True); parser.add_argument("--ckpts", nargs="+", required=True)
    parser.add_argument("--sequences", type=int, default=4000); parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--block", type=int, default=1024)
    args = parser.parse_args()
    for ckpt in args.ckpts:
        model, checkpoint = load(args.out_dir, "cuda", ckpt)
        config, iteration = checkpoint["config"], checkpoint["iter_num"]
        del checkpoint
        model.eval()
        losses = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for x, y in val_batches(config["dataset"], args.block, args.batch, args.sequences // args.batch, "cuda", seed=2026):
                logits, _ = model(x, y)
                per_sequence = torch.nn.functional.cross_entropy(logits.float().transpose(1, 2), y, reduction="none").mean(-1)
                losses.append(per_sequence)
        values = torch.cat(losses)
        print(f"{ckpt:14s} iter {iteration:6d}: val {values.mean().item():.4f} ± {values.std().item() / values.numel() ** 0.5:.4f} "
              f"(sem, {values.numel()} sequences x {args.block} tokens)", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
