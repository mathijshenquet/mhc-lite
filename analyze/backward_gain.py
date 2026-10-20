"""Gradient norm w.r.t. the residual stream at every fold boundary: does the backward signal grow with depth?
usage: python -m analyze.backward_gain --out_dir out-finewebedu-large-gdsm-s0 [--batches 2]
Prints ||dL/dx_k|| for the stream entering fold k (k = 0 is the embedding output, last = before the final norm),
normalised by the value at the last fold, so a per-fold backward gain g shows up as g^(depth) growth toward the input."""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyze.mixer_stats import load, val_batches  # noqa: E402


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
    folds = [m for n, m in model.named_modules() if n.endswith(("hc_attn", "hc_mlp"))]
    captured = []

    def hook(module, inputs):
        x = inputs[0]
        x.retain_grad(); captured.append(x)

    handles = [f.register_forward_pre_hook(hook) for f in folds]
    sums = None; n = 0
    for x, y in val_batches(cfg["dataset"], cfg["block_size"], args.batch_size, args.batches, args.device, seed=2):
        captured.clear(); model.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda" if args.device.startswith("cuda") else "cpu", dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss.backward()
        norms = torch.tensor([t.grad.float().norm().item() for t in captured])
        sums = norms if sums is None else sums + norms; n += 1
    for h in handles:
        h.remove()
    norms = sums / n
    rel = norms / norms[-1]
    print(f"{cfg['hyper_conn_type']} iter {ckpt['iter_num']} folds {len(norms)}: ||dL/dx_k|| relative to the last fold, every 4th fold from the input:")
    print("  " + " ".join(f"{k}:{rel[k]:.2f}" for k in range(0, len(rel), 4)) + f"  last:{rel[-1]:.2f}")
    print(f"  input/last ratio {rel[0]:.2f}; mean per-fold backward gain {(rel[0]) ** (1 / max(1, len(rel) - 1)):.4f}; absolute last {norms[-1]:.4f}")


if __name__ == "__main__":
    main()
