"""How much gradient reaches the residual-mixing logits, per token, and how sensitive is the realisation map
(logits -> mixer) there?  usage: python -m analyze.logit_grad --out_dir out-finewebedu-medium-gdsm-s0 [--ckpt ckpt-500.pt]
Per fold: mean per-token ||dL/dlogits||, mean ||logits||, and the realisation's Jacobian gain ||J v||/||v|| along a
random direction v (finite differences on a sample of tokens).  Works for gdsm (A'/σ), mhc_lite (softmax over n!
permutations) and mhc (Sinkhorn)."""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyze.mixer_stats import load, recorders, val_batches  # noqa: E402


def realise(module, logits):
    """logits -> mixer matrix, for a (N, ...) batch of logit vectors, with the module's own realisation."""
    kind = type(module).__name__
    if kind == "GDSMHyperConnections":
        from hyper_conn.gdsm import gdsm
        n = module.num_residual_streams
        gate = torch.ones(logits.shape[:-2], dtype=logits.dtype, device=logits.device)
        return gdsm(logits, gate)
    if kind == "MHCLite":
        from einops import einsum
        return einsum(torch.softmax(logits, dim=-1), module.perms.to(logits.dtype), '... r, r i j -> ... i j')
    if kind == "ManifoldConstrainedHyperConnections":
        from hyper_conn.mhc import sinkhorn_knopps
        return sinkhorn_knopps(logits, module.sinkhorn_iters)
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ckpt", default="ckpt.pt")
    ap.add_argument("--batches", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--jac_tokens", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    model, ckpt = load(args.out_dir, args.device, args.ckpt)
    model.train(); cfg = ckpt["config"]; rec = recorders(model)
    gnorm = {n: [] for n, _ in rec}; lnorm = {n: [] for n, _ in rec}; jac = {n: [] for n, _ in rec}
    for x, y in val_batches(cfg["dataset"], cfg["block_size"], args.batch_size, args.batches, args.device, seed=4):
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda" if args.device.startswith("cuda") else "cpu", dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss.backward()
        for n, m in rec:
            lg = m.last_logits.detach().float(); g = m.last_logits.grad.float()
            flat_l = lg.reshape(lg.shape[0] * lg.shape[1], -1); flat_g = g.reshape(flat_l.shape)
            gnorm[n].append(flat_g.norm(dim=-1).mean().item()); lnorm[n].append(flat_l.norm(dim=-1).mean().item())
            # Jacobian gain of the realisation along a random direction, finite differences, on a token sample
            idx = torch.randperm(flat_l.shape[0])[: args.jac_tokens]
            sample = lg.reshape(lg.shape[0] * lg.shape[1], *lg.shape[2:])[idx]
            v = torch.randn_like(sample); eps = 1e-3
            with torch.no_grad():
                b0 = realise(m, sample); b1 = realise(m, sample + eps * v / v.flatten(1).norm(dim=-1).reshape(-1, *([1] * (v.dim() - 1))))
                jac[n].append(((b1 - b0).flatten(1).norm(dim=-1) / eps).mean().item())
    import statistics as st
    print(f"{cfg['hyper_conn_type']} iter {ckpt['iter_num']}: per fold | mean ||dL/dlogits|| per token | mean ||logits|| | realisation Jacobian gain")
    for n in gnorm:
        print(f"  {n[14:]:14s} {st.mean(gnorm[n]):.2e} {st.mean(lnorm[n]):8.2f} {st.mean(jac[n]):8.3f}")
    print(f"  MEAN over folds: grad {st.mean(st.mean(v) for v in gnorm.values()):.2e} | logits {st.mean(st.mean(v) for v in lnorm.values()):.2f} | jacobian {st.mean(st.mean(v) for v in jac.values()):.3f}")


if __name__ == "__main__":
    main()
