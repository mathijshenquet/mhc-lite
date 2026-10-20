"""Does signed mixing cancel?  (The "a − b gives noise" hypothesis for GDSM's instability at depth.)
usage: python -m analyze.cancellation --out_dir out-finewebedu-large-gdsm-s0 --ckpt ckpt-1000.pt [--batches 1]

Per fold, on validation tokens, for the residual-stream mixing out_t = Σ_s M[s,t] x_s:
  neg       negative mass of M (Σ negative / Σ |·|)
  fwd       forward cancellation ratio  ||Σ_s M_st x_s|| / Σ_s |M_st| ||x_s||   (1 = no cancellation)
  fwd|M|    the same with all weights made positive (|M|): what the streams' geometry alone gives
  bwd       backward ratio on the gradient through the mixing: ||Σ_t M_st g_t|| / Σ_t |M_st| ||g_t||
  bwd|M|    the same with |M|
  cos       mean pairwise cosine between the 4 input streams
  g_res, g_post   gradient norm of this fold's residual-logit gain and write gain
If cancellation drove the blow-up, GDSM-L would show fwd/bwd well below fwd|M|/bwd|M|, worse than the stable GDSM-M,
and the folds with the most cancellation would carry the largest gain gradients."""

import argparse
import os
import statistics as st
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyze.mixer_stats import load, recorders, val_batches  # noqa: E402


def ratio(weights, vecs):
    """weights (N, s, t), vecs (N, s, d): ||Σ_s w_st v_s|| / Σ_s |w_st| ||v_s||, mean over N and t."""
    num = torch.einsum('nst,nsd->ntd', weights, vecs).norm(dim=-1)
    den = torch.einsum('nst,ns->nt', weights.abs(), vecs.norm(dim=-1)).clamp_min(1e-30)
    return (num / den).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ckpt", default="ckpt.pt")
    ap.add_argument("--batches", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    model, ckpt = load(args.out_dir, args.device, args.ckpt)
    model.train(); cfg = ckpt["config"]; rec = recorders(model)
    for _, m in rec:
        m.record_cancel = True
    rows = {n: [] for n, _ in rec}
    for x, y in val_batches(cfg["dataset"], cfg["block_size"], args.batch_size, args.batches, args.device, seed=5):
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss.backward()
        for n, m in rec:
            s = m.last_mixer.shape[-1]; v = m.last_mix_out.shape[-2] - s
            M = m.last_mixer.float().reshape(-1, s, s)
            xs = m.last_resid_in.float().reshape(M.shape[0], s, -1)
            g = m.last_mix_out.grad.float()[..., v:, :].reshape(M.shape[0], s, -1)
            neg = (M.clamp_max(0).abs().sum() / M.abs().sum()).item()
            xn = xs / xs.norm(dim=-1, keepdim=True).clamp_min(1e-30)
            cos = torch.einsum('nsd,ntd->nst', xn, xn); off = cos[:, ~torch.eye(s, dtype=torch.bool)].mean().item()
            gres = m.residual_scale.grad.norm().item() if getattr(m, "residual_scale", None) is not None and m.residual_scale.grad is not None else float("nan")
            gpost = m.h_post_scale.grad.norm().item() if getattr(m, "h_post_scale", None) is not None and m.h_post_scale.grad is not None else float("nan")
            Mt = M.transpose(-1, -2)   # backward: g_in_s = Σ_t M[s,t] g_t
            rows[n].append((neg, ratio(M, xs), ratio(M.abs(), xs), ratio(Mt, g), ratio(Mt.abs(), g), off, gres, gpost))
    print(f"{cfg['hyper_conn_type']} iter {ckpt['iter_num']}:  fold | neg  fwd  fwd|M|  bwd  bwd|M|  cos | g_res g_post")
    agg = []
    for n, r in rows.items():
        r = [st.mean(c) for c in zip(*r)]; agg.append(r)
        print(f"  {n[14:]:14s} {r[0]:.2f} {r[1]:.3f} {r[2]:.3f}  {r[3]:.3f} {r[4]:.3f}  {r[5]:+.2f} | {r[6]:.2e} {r[7]:.2e}")
    a = [st.mean(c) for c in zip(*agg)]
    print(f"  MEAN           {a[0]:.2f} {a[1]:.3f} {a[2]:.3f}  {a[3]:.3f} {a[4]:.3f}  {a[5]:+.2f}")
    # do the folds that cancel most carry the largest gain gradients?
    import math
    fw = [r[3] / r[4] for r in agg]; gr = [r[6] for r in agg]
    ok = [(x, y) for x, y in zip(fw, gr) if not math.isnan(y) and y > 0]
    if len(ok) > 3 and len(set(round(x, 6) for x, _ in ok)) > 1:
        xs_, ys_ = zip(*ok); mx, my = st.mean(xs_), st.mean([math.log(y) for y in ys_])
        cov = sum((x - mx) * (math.log(y) - my) for x, y in ok); vx = sum((x - mx) ** 2 for x in xs_); vy = sum((math.log(y) - my) ** 2 for y in ys_)
        print(f"  corr(bwd/bwd|M| sign-cancellation, log g_res) over folds: {cov / math.sqrt(vx * vy):+.2f}  (negative = more cancellation ↔ bigger gradient)")


if __name__ == "__main__":
    main()
