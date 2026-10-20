"""What is Adam actually doing per parameter group?  Reads a checkpoint's optimizer state (no forward pass).
usage: python -m analyze.adam_state --out_dir out-finewebedu-large-gdsm-s0 --ckpt ckpt-1000.pt
Per group (attn, mlp, embed, hc.* split by type):
  upd/w     ||next Adam update|| / ||w||   (update-to-weight ratio; healthy ~1e-3)
  snr       mean |m̂| / (√v̂ + eps)        (1 = gradient sign consistent across steps, ~0 = sign flips: oscillation)
  √v̂        RMS of the second-moment estimate (gradient scale Adam has seen)
Assumes the original (decay, no-decay) two-group layout in named_parameters order, as nanoGPT writes it."""

import argparse
import math
import os
import sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyze.mixer_stats import load  # noqa: E402
from analyze.grad_breakdown import group  # noqa: E402


def lr_at(it, cfg):
    lr, mn, wu, dec = cfg["learning_rate"], cfg["min_lr"], cfg["warmup_iters"], cfg["lr_decay_iters"]
    if it < wu:
        return lr * (it + 1) / (wu + 1)
    if it > dec:
        return mn
    return mn + 0.5 * (1 + math.cos(math.pi * (it - wu) / (dec - wu))) * (lr - mn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ckpt", default="ckpt.pt")
    args = ap.parse_args()
    model, ckpt = load(args.out_dir, "cpu", args.ckpt)
    cfg = ckpt["config"]; opt = ckpt["optimizer"]
    named = list(model.named_parameters())
    order = [(n, p) for n, p in named if p.dim() >= 2] + [(n, p) for n, p in named if p.dim() < 2]
    flat_ids = [i for g in opt["param_groups"] for i in g["params"]]
    assert len(flat_ids) == len(order), (len(flat_ids), len(order))
    b1, b2 = opt["param_groups"][0]["betas"]; eps = opt["param_groups"][0]["eps"]; wd = opt["param_groups"][0]["weight_decay"]
    lr = lr_at(ckpt["iter_num"], cfg)
    acc = defaultdict(lambda: [0.0, 0.0, 0.0, 0, 0.0])   # upd², w², snr·n, n, v
    for (n, p), i in zip(order, flat_ids):
        st = opt["state"].get(i)
        if not st:
            continue
        t = float(st["step"]); m = st["exp_avg"].float(); v = st["exp_avg_sq"].float(); w = p.detach().float()
        mh = m / (1 - b1 ** t); vh = v / (1 - b2 ** t); den = vh.sqrt() + eps
        upd = lr * mh / den + (lr * wd * w if p.dim() >= 2 else 0.0)
        a = acc[group(n)]
        a[0] += upd.pow(2).sum().item(); a[1] += w.pow(2).sum().item(); a[2] += (mh.abs() / den).sum().item(); a[3] += w.numel(); a[4] += vh.sum().item()
    print(f"{cfg['hyper_conn_type']} iter {ckpt['iter_num']} lr {lr:.2e}:  group | upd/w | snr |m̂|/√v̂ | RMS √v̂ | params")
    for k in sorted(acc, key=lambda k: -acc[k][0] ** 0.5 / max(acc[k][1], 1e-30) ** 0.5):
        u, w2, snr, cnt, vs = acc[k]
        print(f"  {k:22s} {u ** 0.5 / max(w2, 1e-30) ** 0.5:.2e}  {snr / cnt:.3f}  {(vs / cnt) ** 0.5:.2e}  {cnt}")


if __name__ == "__main__":
    main()
