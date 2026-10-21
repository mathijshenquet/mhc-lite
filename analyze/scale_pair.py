"""Test the scale-invariance explanation of GDSM's indefinite Hessian.
GDSM's residual map is unchanged when, in one fold, the static residual logits and the residual gain are scaled
together (x_f = (static_alpha[:, v:], residual_scale) → c·x_f scales A' by c, and A'/σ_max is scale-invariant).
Then exactly: g_f · x_f = 0 and H x_f = −g_f (gradient restricted to x_f), so the Hessian has a pair ≈ ±‖g_f‖/‖x_f‖
in span(x_f, g_f).  Per fold: ‖g_f‖/‖x_f‖ and cos(g_f, x_f); for the top folds also the identity check
cos(H x_f, −g_f) and the exact 2×2 eigenvalues in span(x̂, ĝ).
usage: python -m analyze.scale_pair --out_dir out-finewebedu-large-gdsm-s0 --ckpt ckpt-1000.pt [--batches 4 --block 512 --top 3]"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyze.mixer_stats import load, val_batches  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True); ap.add_argument("--ckpt", default="ckpt.pt")
    ap.add_argument("--batches", type=int, default=4); ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--top", type=int, default=3); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    model, ck = load(a.out_dir, "cpu", a.ckpt); model = model.to(a.device).float().train(); cfg = ck["config"]; it = ck["iter_num"]; del ck
    folds = [(n, m) for n, m in model.named_modules() if n.endswith(("hc_attn", "hc_mlp"))]
    params = [p for p in model.parameters()]
    idx = {id(p): i for i, p in enumerate(params)}
    batches = list(val_batches(cfg["dataset"], a.block, 1, a.batches, a.device, seed=12))

    def loss_fn():
        tot = 0.0
        for x, y in batches:
            with sdpa_kernel(SDPBackend.MATH):
                _, l = model(x, y)
            tot = tot + l / len(batches)
        return tot

    def xvec(m):
        v = m.num_input_views
        out = [torch.zeros_like(p) for p in params]
        out[idx[id(m.static_alpha)]][:, v:] = m.static_alpha.detach()[:, v:]
        out[idx[id(m.residual_scale)]] = m.residual_scale.detach().clone()
        return out

    loss = loss_fn()
    grads = torch.autograd.grad(loss, params, create_graph=True)
    rows = []
    for n, m in folds:
        x = xvec(m)
        mask = [(xi != 0) for xi in x]
        gf = [g.detach() * mk for g, mk in zip(grads, mask)]
        xn = torch.sqrt(sum((xi ** 2).sum() for xi in x)).item(); gn = torch.sqrt(sum((gi ** 2).sum() for gi in gf)).item()
        cos_gx = (sum((gi * xi).sum() for gi, xi in zip(gf, x)).item() / max(xn * gn, 1e-30))
        rows.append((n, gn / max(xn, 1e-30), cos_gx, xn, gn, x, gf))
    rows.sort(key=lambda r: -r[1])
    print(f"gdsm iter {it}: per fold ‖g_f‖/‖x_f‖ (pair magnitude prediction), cos(g_f, x_f) (should be ≈ 0)")
    for n, r, c, xn, gn, *_ in rows[:8]:
        print(f"  {n[14:]:14s} ratio {r:.3e}  cos {c:+.4f}  ‖x‖ {xn:.3f}  ‖g‖ {gn:.3e}")
    print(f"  max ratio over folds {rows[0][1]:.3e} | median {np.median([r[1] for r in rows]):.3e}")
    for n, r, c, xn, gn, x, gf in rows[:a.top]:
        hx = torch.autograd.grad(grads, params, grad_outputs=x, retain_graph=True)
        hxv = [h.detach() for h in hx]
        cos_id = -sum((h * g).sum() for h, g in zip(hxv, gf)).item() / max(torch.sqrt(sum((h ** 2).sum() for h in hxv)).item() * gn, 1e-30)
        xh = [xi / xn for xi in x]; gh = [gi / gn for gi in gf]
        hg = [h.detach() for h in torch.autograd.grad(grads, params, grad_outputs=gh, retain_graph=True)]
        hxh = [h / xn for h in hxv]
        b = lambda u, w: sum((ui * wi).sum() for ui, wi in zip(u, w)).item()
        M = np.array([[b(xh, hxh), b(xh, hg)], [b(gh, hxh), b(gh, hg)]]); ev = np.linalg.eigvalsh((M + M.T) / 2)
        print(f"  {n[14:]:14s} identity cos(Hx, −g) {cos_id:+.4f} | 2×2 in span(x̂,ĝ): [[{M[0,0]:.2e}, {M[0,1]:.2e}], [{M[1,0]:.2e}, {M[1,1]:.2e}]] → eigenvalues {ev[0]:.3e}, {ev[1]:.3e} (prediction ±{r:.3e})")


if __name__ == "__main__":
    main()
