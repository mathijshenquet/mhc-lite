"""Per-fold parameter magnitudes of the hyper-connections: the learned gain scalars and the Frobenius norms of the
dynamic projections and static logits.  usage: python -m analyze.param_stats --out_dir out-finewebedu-large-gdsm-s0"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyze.mixer_stats import load  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ckpt", default="ckpt.pt")
    args = ap.parse_args()
    model, ckpt = load(args.out_dir, "cpu", args.ckpt)
    rows = []
    for name, m in model.named_modules():
        if not name.endswith(("hc_attn", "hc_mlp")):
            continue
        f = lambda k: float(getattr(m, k).float().norm()) if hasattr(m, k) else float("nan")
        g = lambda k: " ".join(f"{v:.3f}" for v in getattr(m, k).detach().float().flatten().tolist()) if hasattr(m, k) else "-"
        rows.append((name[14:], f("dynamic_alpha_fn"), f("dynamic_beta_fn"), f("dynamic_gate_fn"), f("static_alpha"), f("static_beta"),
                     g("pre_branch_scale"), g("residual_scale"), g("h_post_scale"), g("gate_scale"), g("static_gate")))
    print(f"{ckpt['config']['hyper_conn_type']} iter {ckpt['iter_num']}: per fold ||dyn_alpha|| ||dyn_beta|| ||dyn_gate|| ||static_alpha|| ||static_beta|| | pre_scale res_scale post_scale gate_scale static_gate")
    for r in rows:
        print(f"  {r[0]:14s} {r[1]:8.3f} {r[2]:8.3f} {r[3]:8.3f} {r[4]:8.3f} {r[5]:8.3f} | {r[6]} | {r[7]} | {r[8]} | {r[9]} | {r[10]}")
    import statistics as st
    for i, k in ((1, "dyn_alpha"), (2, "dyn_beta"), (3, "dyn_gate"), (4, "static_alpha")):
        v = [r[i] for r in rows]
        print(f"  mean ||{k}|| {st.mean(v):.3f} max {max(v):.3f}")


if __name__ == "__main__":
    main()
