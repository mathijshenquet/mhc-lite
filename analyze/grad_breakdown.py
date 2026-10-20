"""Gradient norm per parameter group for one batch: are the large gradients in the hyper-connections, the
attention, the MLPs or the embeddings?  usage: python -m analyze.grad_breakdown --out_dir out-finewebedu-large-gdsm-s0 [--batches 2]"""

import argparse
import collections
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyze.mixer_stats import load, val_batches  # noqa: E402


def group(name):
    if "branch_attn" in name:
        return "attn"
    if "branch_mlp" in name:
        return "mlp"
    if "hc_attn" in name or "hc_mlp" in name:
        for key in ("static_alpha", "dynamic_alpha", "dynamic_gate", "static_gate", "static_beta", "dynamic_beta", "norm", "gate_scale", "pre_branch_scale", "residual_scale", "h_post_scale", "dynamic_alpha_scale", "dynamic_beta_scale", "scale"):
            if key in name:
                return f"hc.{key}"
        return "hc.other"
    if "wte" in name or "wpe" in name or "lm_head" in name:
        return "embed"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ckpt", default="ckpt.pt")
    ap.add_argument("--batches", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    model, ckpt = load(args.out_dir, args.device, args.ckpt)
    model.train(); cfg = ckpt["config"]
    sums = collections.defaultdict(float); counts = collections.defaultdict(int); per_layer = collections.defaultdict(float)
    for x, y in val_batches(cfg["dataset"], cfg["block_size"], args.batch_size, args.batches, args.device, seed=1):
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda" if args.device.startswith("cuda") else "cpu", dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss.backward()
        total = 0.0
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            g2 = p.grad.float().pow(2).sum().item(); total += g2
            sums[group(name)] += g2; counts[group(name)] += p.numel()
            if name.startswith("transformer.h."):
                per_layer[int(name.split(".")[2])] += g2
        print(f"batch: loss {loss.item():.3f}  total grad norm {total ** 0.5:.3f}")
    print(f"{'group':20s} {'grad norm':>10s} {'params':>10s}")
    for k in sorted(sums, key=lambda k: -sums[k]):
        print(f"{k:20s} {sums[k] ** 0.5:10.3f} {counts[k]:10d}")
    print("per block grad norm: " + " ".join(f"{i}:{per_layer[i] ** 0.5:.2f}" for i in sorted(per_layer)))


if __name__ == "__main__":
    main()
