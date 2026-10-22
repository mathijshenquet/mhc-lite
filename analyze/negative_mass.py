"""How signed is GDSM's realised residual mixer over training?  For each checkpoint, on a few validation sequences:
per fold, the negative mass of B (sum of |negative entries| / sum of |entries|, mean over tokens) and the fraction of
tokens with any negative entry; plus the effective rank of the stream differences (x − x̄) over the feature dimension
(participation ratio of their singular values), which tests whether the differences live in a low-dimensional subspace.
usage: python -m analyze.negative_mass --out_dir <dir> --ckpts ckpt-500.pt ckpt-1000.pt ... [--batches 2 --block 512]"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyze.mixer_stats import load, val_batches  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True); parser.add_argument("--ckpts", nargs="+", required=True)
    parser.add_argument("--batches", type=int, default=2); parser.add_argument("--block", type=int, default=512)
    args = parser.parse_args()
    for ckpt in args.ckpts:
        model, checkpoint = load(args.out_dir, "cpu", ckpt)
        model = model.float().eval(); config, iteration = checkpoint["config"], checkpoint["iter_num"]; del checkpoint
        folds = [(n, m) for n, m in model.named_modules() if n.endswith(("hc_attn", "hc_mlp"))]
        stats = {n: {"neg": [], "any": [], "rank": []} for n, _ in folds}
        for name, module in folds:
            module.record_mixers = True
            original = module.width_connection

            def recording(residuals, original=original, name=name, module=module):
                out = original(residuals)
                mixer = module.last_mixer.float()                                   # (..., s, t) per token
                mixer = mixer.reshape(-1, *mixer.shape[-2:])
                negative = torch.relu(-mixer).sum((-2, -1)); total = mixer.abs().sum((-2, -1)).clamp_min(1e-12)
                stats[name]["neg"].append((negative / total).mean().item())
                stats[name]["any"].append((negative > 1e-6).float().mean().item())
                x = residuals.float().unflatten(0, (-1, module.num_residual_streams))   # (b, s, t, d)
                diffs = (x - x.mean(1, keepdim=True)).reshape(-1, x.shape[-1])
                singular = torch.linalg.svdvals(diffs[:: max(1, diffs.shape[0] // 4096)])
                stats[name]["rank"].append((singular.sum() ** 2 / (singular ** 2).sum()).item() if singular.sum() > 0 else 0.0)
                return out

            module.width_connection = recording
        with torch.no_grad():
            for x, y in val_batches(config["dataset"], args.block, 1, args.batches, "cpu", seed=5):
                model(x, y)
        mean = lambda v: sum(v) / len(v)
        neg = [mean(stats[n]["neg"]) for n, _ in folds]; anyneg = [mean(stats[n]["any"]) for n, _ in folds]
        rank = [mean(stats[n]["rank"]) for n, _ in folds]
        picks = [2, 11, 23, 35, 47]
        print(f"iter {iteration:6d} | neg mass mean {mean(neg):.4f} max {max(neg):.4f} | tokens with a negative entry {mean(anyneg):.3f} | "
              f"diff eff-rank per fold " + " ".join(f"f{i + 1}:{rank[i]:.0f}" for i in picks if i < len(rank)) +
              " | neg mass per fold " + " ".join(f"f{i + 1}:{neg[i]:.3f}" for i in picks if i < len(neg)), flush=True)


if __name__ == "__main__":
    main()
