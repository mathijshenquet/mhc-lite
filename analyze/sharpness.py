"""Preconditioned sharpness at a checkpoint: top eigenvalue of P^-1 H with Adam's preconditioner P = diag(√v̂+eps),
by power iteration on Hessian-vector products (Cohen et al., adaptive edge of stability: Adam with β1 = 0.9 is stable
while λ_max(P^-1 H) < 38/lr).  Also the same restricted to the hyper-connection gain scalars, and the raw λ_max(H).
usage: python -m analyze.sharpness --out_dir out-finewebedu-large-gdsm-s0 --ckpt ckpt-1000.pt [--iters 20]"""

import argparse
import os
import sys

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyze.mixer_stats import load, val_batches  # noqa: E402
from analyze.adam_state import lr_at  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ckpt", default="ckpt.pt")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--block", type=int, default=0, help="sequence length for the HVP batch (0 = the model block size)")
    args = ap.parse_args()
    dev = args.device
    model, ckpt = load(args.out_dir, "cpu", args.ckpt)   # optimizer state stays on the CPU; only P^-1/2 goes to the GPU
    model = model.to(dev)
    model.train(); cfg = ckpt["config"]; opt = ckpt["optimizer"]
    named = list(model.named_parameters())
    order = [(n, p) for n, p in named if p.dim() >= 2] + [(n, p) for n, p in named if p.dim() < 2]
    flat_ids = [i for g in opt["param_groups"] for i in g["params"]]
    b1, b2 = opt["param_groups"][0]["betas"]; eps = opt["param_groups"][0]["eps"]
    params = [p for _, p in order]; names = [n for n, _ in order]
    dinv = []
    for (n, p), i in zip(order, flat_ids):
        st = opt["state"][i]; t = float(st["step"])
        dinv.append((st["exp_avg_sq"].float().to(dev) / (1 - b2 ** t)).sqrt().add(eps).rsqrt())   # P^-1/2
    scalar_mask = [n.endswith("_scale") and p.dim() <= 1 for n, p in order]
    del ckpt["optimizer"], ckpt["model"]
    x, y = next(iter(val_batches(cfg["dataset"], args.block or cfg["block_size"], args.batch_size, 1, dev, seed=6)))
    with sdpa_kernel(SDPBackend.MATH), torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
        _, loss = model(x, y)
    grads = torch.autograd.grad(loss, params, create_graph=True)

    def hvp(vec):
        return torch.autograd.grad(grads, params, grad_outputs=vec, retain_graph=True)

    def power(precond, mask=None, shift=0.0):
        v = [torch.randn_like(p) for p in params]
        if mask is not None:
            v = [vi if m else torch.zeros_like(vi) for vi, m in zip(v, mask)]
        nrm = torch.sqrt(sum((vi.float() ** 2).sum() for vi in v)); v = [vi / nrm for vi in v]
        lam = 0.0
        for _ in range(args.iters):
            u = [vi * di for vi, di in zip(v, dinv)] if precond else v
            hu = hvp(u)
            w = [h.float() * di for h, di in zip(hu, dinv)] if precond else [h.float() for h in hu]
            if mask is not None:
                w = [wi if m else torch.zeros_like(wi) for wi, m in zip(w, mask)]
            if shift:
                w = [wi + shift * vi for wi, vi in zip(w, v)]
            lam = sum((wi * vi).sum() for wi, vi in zip(w, v)).item()
            nrm = torch.sqrt(sum((wi ** 2).sum() for wi in w)); v = [wi / nrm for wi in w]
        return lam - shift

    lr = lr_at(ckpt["iter_num"], cfg)
    def top(precond, mask=None):
        lam = power(precond, mask)
        # power iteration finds the largest |λ|; if that is negative, shift to get the largest algebraic one
        return power(precond, mask, shift=-lam * 1.0 + 1e-12) if lam < 0 else lam
    full = top(True); scal = top(True, scalar_mask); raw = top(False)
    print(f"{ckpt['model_args']['hyper_conn_type']} iter {ckpt['iter_num']} lr {lr:.2e}: λ(P⁻¹H) full {full:.3e} | gain scalars only {scal:.3e} | threshold 38/lr {38 / lr:.3e} | raw λ(H) {raw:.3e}")


if __name__ == "__main__":
    main()
