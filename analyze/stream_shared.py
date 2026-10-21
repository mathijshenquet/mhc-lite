"""Shared stream component vs stream differences per fold: the scale that bf16 rounding of GDSM's mixing leaks from.
For every hyper-connection fold, on a few validation sequences: mean per-token norm of the stream mean over the
s streams (shared) and of the deviations from it (differences), and their ratio.  Forward only, fp32, CPU-friendly.
usage: python -m analyze.stream_shared --out_dir <run dir> --ckpts ckpt-500.pt ckpt-1000.pt [--batches 2 --block 512]"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyze.mixer_stats import load, val_batches  # noqa: E402


def measure(out_dir, ckpt, batches, block, device):
    model, checkpoint = load(out_dir, "cpu", ckpt)
    model = model.to(device).float().eval()
    config, iteration = checkpoint["config"], checkpoint["iter_num"]
    del checkpoint
    folds = [(name, module) for name, module in model.named_modules() if name.endswith(("hc_attn", "hc_mlp"))]
    sums = {name: [0.0, 0.0, 0] for name, _ in folds}
    for name, module in folds:
        streams = module.num_residual_streams
        original = module.width_connection

        def recording(residuals, original=original, name=name, streams=streams):
            x = residuals.float().unflatten(0, (-1, streams))          # (b, s, t, d)
            shared = x.mean(dim=1, keepdim=True)
            sums[name][0] += shared.norm(dim=-1).mean().item()
            sums[name][1] += (x - shared).norm(dim=-1).mean().item()
            sums[name][2] += 1
            return original(residuals)

        module.width_connection = recording
    with torch.no_grad():
        for x, y in val_batches(config["dataset"], block, 1, batches, device, seed=7):
            model(x, y)
    rows = []
    for name, _ in folds:
        shared, diff, count = sums[name]
        rows.append((name, shared / count, diff / count))
    return iteration, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--ckpts", nargs="+", required=True)
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--block", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    for ckpt in args.ckpts:
        iteration, rows = measure(args.out_dir, ckpt, args.batches, args.block, args.device)
        picks = [0, 5, 11, 23, 35, 47]
        label = os.path.basename(args.out_dir.rstrip("/")).replace("out-finewebedu-large-", "")
        print(f"{label:16s} iter {iteration:5d} shared/diff per fold " + " ".join(
            f"{rows[i][0].split('.')[-2]}{'a' if rows[i][0].endswith('attn') else 'm'}:{rows[i][1]:.0f}/{rows[i][2]:.0f}={rows[i][1] / max(rows[i][2], 1e-9):.1f}"
            for i in picks if i < len(rows)), flush=True)


if __name__ == "__main__":
    main()
