"""Is the gradient turning into noise?  Gradient agreement across independent batches at a checkpoint.
usage: python -m analyze.grad_noise --out_dir out-finewebedu-large-gdsm-s0 --ckpt ckpt-1000.pt [--n 16 --block 512]
From n per-batch gradients g_i (running sums only, no gradients stored), per parameter group:
  |g|        mean per-batch gradient norm
  |G|        norm of the true (expected) gradient, unbiased: |G|² ≈ (|Σg|² − Σ|g_i|²) / (n(n−1))
  cos        mean pairwise cosine between per-batch gradients: (|Σ ĝ_i|² − n) / (n(n−1))
  B_noise    gradient noise scale tr(Σ)/|G|² in units of one batch (McCandlish et al.): batches needed for signal ≈ noise
High |g| with small |G| and cos → the gradient is mostly noise."""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyze.mixer_stats import load, val_batches  # noqa: E402


def group(n):
    if n.endswith("_scale"):
        return "gain_scalars"
    if "alpha" in n or "beta" in n or "gate" in n:
        return "hc_logits"
    return "body"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True); ap.add_argument("--ckpt", default="ckpt.pt")
    ap.add_argument("--n", type=int, default=16); ap.add_argument("--block", type=int, default=512)
    args = ap.parse_args()
    model, ckpt = load(args.out_dir, "cpu", args.ckpt); model.train()
    cfg = ckpt["config"]; kind = ckpt["model_args"]["hyper_conn_type"]; it = ckpt["iter_num"]; del ckpt
    named = [(n, p) for n, p in model.named_parameters()]
    groups = sorted({group(n) for n, _ in named}) + ["all"]
    sum_g = {g: None for g in groups}; sq = {g: [] for g in groups}
    from collections import defaultdict
    sum_hat = defaultdict(lambda: None)
    for x, y in val_batches(cfg["dataset"], args.block, 1, args.n, "cpu", seed=10):
        model.zero_grad(set_to_none=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss.backward()
        grads = [p.grad.float() if p.grad is not None else torch.zeros_like(p, dtype=torch.float32) for _, p in named]
        for g in groups:
            sel = [gr for (n, _), gr in zip(named, grads) if g == "all" or group(n) == g]
            nrm2 = sum((t ** 2).sum() for t in sel).item(); sq[g].append(nrm2)
            hat = [t / max(nrm2, 1e-30) ** 0.5 for t in sel]
            sum_hat[g] = hat if sum_hat[g] is None else [a + b for a, b in zip(sum_hat[g], hat)]
            sum_g[g] = [t.clone() for t in sel] if sum_g[g] is None else [a + b for a, b in zip(sum_g[g], sel)]
    n = args.n
    print(f"{kind} iter {it} ({n} batches × {args.block} tokens): group | mean |g| | |G| | mean cos | B_noise")
    for g in groups:
        S2 = sum((t ** 2).sum() for t in sum_g[g]).item(); ssq = sum(sq[g])
        G2 = (S2 - ssq) / (n * (n - 1)); trS = (ssq - S2 / n) / (n - 1)
        H2 = sum((t ** 2).sum() for t in sum_hat[g]).item(); cos = (H2 - n) / (n * (n - 1))
        mean_g = sum(v ** 0.5 for v in sq[g]) / n
        print(f"  {g:13s} {mean_g:.3e}  {max(G2, 0) ** 0.5:.3e}  {cos:+.3f}  {trS / G2 if G2 > 0 else float('inf'):.1f}")


if __name__ == "__main__":
    main()
