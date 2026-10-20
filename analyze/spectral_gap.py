"""Is GDSM's gradient noise tied to tokens whose top two singular values of A' nearly cross?
usage: python -m analyze.spectral_gap --out_dir out-finewebedu-large-gdsm-s0 --ckpt ckpt-1500.pt [--batches 2]
Per fold: distribution of the relative gap (σ₁−σ₂)/σ₁ of the centred residual logits per token, and the per-token
gradient norm w.r.t. those logits, split by gap: small-gap tokens (< 0.05) vs the rest, plus the share of the total
squared logit-gradient carried by the 1 % largest tokens."""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyze.mixer_stats import load, recorders, val_batches  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ckpt", default="ckpt.pt")
    ap.add_argument("--batches", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    model, ckpt = load(args.out_dir, args.device, args.ckpt)
    model.train(); cfg = ckpt["config"]; rec = recorders(model)
    gaps = {n: [] for n, _ in rec}; gnorm = {n: [] for n, _ in rec}
    for x, y in val_batches(cfg["dataset"], cfg["block_size"], args.batch_size, args.batches, args.device, seed=3):
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda" if args.device.startswith("cuda") else "cpu", dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss.backward()
        for n, m in rec:
            sv = m.last_raw_svals.float().reshape(-1, m.last_raw_svals.shape[-1]).cpu()
            gaps[n].append((sv[:, 0] - sv[:, 1]) / sv[:, 0].clamp_min(1e-12))
            g = m.last_logits.grad.float().reshape(sv.shape[0], -1).norm(dim=-1).cpu()
            gnorm[n].append(g)
    print(f"{cfg['hyper_conn_type']} iter {ckpt['iter_num']}: per fold | rel gap median / p5 | frac gap<0.05 | logit-grad mean: small-gap vs rest (ratio) | top-1% tokens' share of grad²")
    tot_small, tot_rest, agg = [], [], []
    for n in gaps:
        gp = torch.cat(gaps[n]); gn = torch.cat(gnorm[n]); small = gp < 0.05
        ms = gn[small].mean().item() if small.any() else float("nan"); mr = gn[~small].mean().item()
        top = gn.pow(2).sort(descending=True).values; share = top[: max(1, len(top) // 100)].sum().item() / top.sum().clamp_min(1e-30).item()
        print(f"  {n[14:]:14s} {gp.median():.3f} / {torch.quantile(gp, 0.05):.3f} | {small.float().mean():.3f} | {ms:.2e} vs {mr:.2e} ({ms / mr if mr > 0 else float('nan'):.2f}) | {share:.2f}")
        tot_small.append(ms); tot_rest.append(mr); agg.append(share)
    import statistics as st
    ok = [(a, b) for a, b in zip(tot_small, tot_rest) if a == a]
    print(f"  mean ratio small-gap/rest {st.mean(a / b for a, b in ok):.2f}; mean top-1% share {st.mean(agg):.2f}")


if __name__ == "__main__":
    main()
