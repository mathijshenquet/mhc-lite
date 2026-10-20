"""Preconditioned sharpness at a checkpoint: extreme eigenvalues of P^-1/2 H P^-1/2 (same spectrum as P^-1 H) with
Adam's preconditioner P = diag(√v̂ + eps), by Lanczos on Hessian-vector products in fp32, the Hessian averaged over
several batches.  Adam with β1 = 0.9 is stable while λ_max(P^-1 H) < 38/lr (Cohen et al., adaptive edge of
stability).  Also the same for the gain-scalar block and the raw (unpreconditioned) Hessian.
usage: python -m analyze.sharpness --out_dir out-finewebedu-large-gdsm-s0 --ckpt ckpt-1000.pt [--steps 30 --batches 4 --block 512]
       python -m analyze.sharpness --selftest      (exact full-Hessian check on a tiny model)"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def lr_at(it, cfg):
    import math
    lr, mn, wu, dec = cfg["learning_rate"], cfg["min_lr"], cfg["warmup_iters"], cfg["lr_decay_iters"]
    if it < wu:
        return lr * (it + 1) / (wu + 1)
    if it > dec:
        return mn
    return mn + 0.5 * (1 + math.cos(math.pi * (it - wu) / (dec - wu))) * (lr - mn)


def lanczos(matvec, dim_like, steps, seed=0):
    """Plain Lanczos (no reorthogonalisation; the extreme Ritz values converge first).  Returns (λ_min, λ_max)."""
    g = torch.Generator(device=dim_like[0].device).manual_seed(seed)
    v = [torch.randn(p.shape, generator=g, device=p.device, dtype=torch.float32) for p in dim_like]
    nrm = torch.sqrt(sum((x ** 2).sum() for x in v)); v = [x / nrm for x in v]
    v_prev = None; beta = 0.0; alphas, betas = [], []
    for _ in range(steps):
        w = matvec(v)
        alpha = sum((wi * vi).sum() for wi, vi in zip(w, v)).item()
        w = [wi - alpha * vi - (beta * vp if v_prev is not None else 0) for wi, vi, vp in zip(w, v, v_prev or v)]
        beta_new = torch.sqrt(sum((wi ** 2).sum() for wi in w)).item()
        alphas.append(alpha); betas.append(beta_new)
        if beta_new < 1e-10:
            break
        v_prev, v, beta = v, [wi / beta_new for wi in w], beta_new
    T = np.diag(alphas) + np.diag(betas[:-1], 1) + np.diag(betas[:-1], -1)
    ev = np.linalg.eigvalsh(T)
    return float(ev[0]), float(ev[-1])


def spectra(model, params, dinv, batches, steps, scalar_mask):
    """λ ranges of P^-1/2 H P^-1/2 (full, scalar block) and of raw H, the Hessian averaged over `batches`."""
    def hvp(vec):
        acc = [torch.zeros_like(p, dtype=torch.float32) for p in params]
        for x, y in batches:
            with sdpa_kernel(SDPBackend.MATH):
                _, loss = model(x, y)
            grads = torch.autograd.grad(loss, params, create_graph=True)
            hv = torch.autograd.grad(grads, params, grad_outputs=vec)
            acc = [a + h.float() / len(batches) for a, h in zip(acc, hv)]
        return acc

    def precond(vec, mask=None):
        u = [vi * di for vi, di in zip(vec, dinv)]
        if mask is not None:
            u = [ui if m else torch.zeros_like(ui) for ui, m in zip(u, mask)]
        w = [h * di for h, di in zip(hvp(u), dinv)]
        if mask is not None:
            w = [wi if m else torch.zeros_like(wi) for wi, m in zip(w, mask)]
        return w

    full = lanczos(precond, params, steps)
    scal = lanczos(lambda v: precond(v, scalar_mask), [p if m else p for p, m in zip(params, scalar_mask)], steps, seed=1) if any(scalar_mask) else (float("nan"),) * 2
    raw = lanczos(hvp, params, steps, seed=2)
    return full, scal, raw


def selftest():
    """Tiny GDSM model: Lanczos vs the exact eigenvalues of P^-1/2 H P^-1/2 from the full Hessian."""
    from model import GPT, GPTConfig
    torch.manual_seed(0)
    m = GPT(GPTConfig(n_layer=1, n_head=2, n_embd=16, block_size=16, bias=False, vocab_size=17, dropout=0.0, hyper_conn_n=4, hyper_conn_type="gdsm")).double()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2, betas=(0.9, 0.95))
    x = torch.randint(0, 17, (4, 16)); y = torch.randint(0, 17, (4, 16))
    for _ in range(20):
        _, l = m(x, y); opt.zero_grad(); l.backward(); opt.step()
    params = [p for p in m.parameters()]
    dinv = []
    for p in params:
        st = opt.state[p]; t = float(st["step"])
        dinv.append((st["exp_avg_sq"] / (1 - 0.95 ** t)).sqrt().add(1e-8).rsqrt())
    names = [n for n, _ in m.named_parameters()]
    mask = [n.endswith("_scale") and p.dim() <= 1 for n, p in zip(names, params)]
    batches = [(x[:2], y[:2]), (x[2:], y[2:])]
    full, scal, raw = spectra(m, params, [d.double() for d in dinv], batches, steps=int(os.environ.get("SELFTEST_STEPS", "200")), scalar_mask=mask)
    flat = torch.cat([p.detach().flatten() for p in params]); shapes = [p.shape for p in params]; sizes = [p.numel() for p in params]

    def loss_flat(f):
        chunks = torch.split(f, sizes); new = {n: c.view(s) for n, c, s in zip(names, chunks, shapes)}
        out = sum(torch.func.functional_call(m, new, (bx, by))[1] for bx, by in batches) / len(batches)
        return out
    with sdpa_kernel(SDPBackend.MATH):
        H = torch.autograd.functional.hessian(loss_flat, flat)
    d = torch.cat([di.flatten() for di in dinv]).double()
    M = d[:, None] * H * d[None, :]
    ev = torch.linalg.eigvalsh(M); evr = torch.linalg.eigvalsh(H)
    sm = torch.cat([torch.full((p.numel(),), float(mk)) for p, mk in zip(params, mask)]).bool()
    evs = torch.linalg.eigvalsh(M[sm][:, sm])
    print(f"selftest ({flat.numel()} params): full exact [{ev[0]:.4e}, {ev[-1]:.4e}] lanczos [{full[0]:.4e}, {full[1]:.4e}]")
    print(f"                 scalars exact [{evs[0]:.4e}, {evs[-1]:.4e}] lanczos [{scal[0]:.4e}, {scal[1]:.4e}]")
    print(f"                 raw H   exact [{evr[0]:.4e}, {evr[-1]:.4e}] lanczos [{raw[0]:.4e}, {raw[1]:.4e}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir"); ap.add_argument("--ckpt", default="ckpt.pt")
    ap.add_argument("--steps", type=int, default=40); ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--block", type=int, default=512); ap.add_argument("--device", default="cuda")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    from analyze.mixer_stats import load, val_batches
    dev = args.device
    model, ckpt = load(args.out_dir, "cpu", args.ckpt)
    model = model.to(dev).float().train(); cfg = ckpt["config"]; opt = ckpt["optimizer"]
    named = list(model.named_parameters())
    order = [(n, p) for n, p in named if p.dim() >= 2] + [(n, p) for n, p in named if p.dim() < 2]
    flat_ids = [i for g in opt["param_groups"] for i in g["params"]]
    b2 = opt["param_groups"][0]["betas"][1]; eps = opt["param_groups"][0]["eps"]
    params = [p for _, p in order]
    dinv = [(opt["state"][i]["exp_avg_sq"].float() / (1 - b2 ** float(opt["state"][i]["step"]))).sqrt().add(eps).rsqrt().to(dev) for i in flat_ids]
    mask = [n.endswith("_scale") and p.dim() <= 1 for n, p in order]
    it = ckpt["iter_num"]; lr = lr_at(it, cfg); kind = ckpt["model_args"]["hyper_conn_type"]
    del ckpt
    batches = list(val_batches(cfg["dataset"], args.block, 1, args.batches, dev, seed=6))
    full, scal, raw = spectra(model, params, dinv, batches, args.steps, mask)
    print(f"{kind} iter {it} lr {lr:.2e} (fp32, Lanczos {args.steps}, {args.batches}×{args.block} tokens): "
          f"λ(P⁻¹H) [{full[0]:.3e}, {full[1]:.3e}] | gain scalars [{scal[0]:.3e}, {scal[1]:.3e}] | threshold 38/lr {38 / lr:.3e} | raw λ(H) [{raw[0]:.3e}, {raw[1]:.3e}]")


if __name__ == "__main__":
    main()
