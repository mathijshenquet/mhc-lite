"""Which of the n residual streams does a trained model actually use?

usage: python -m analyze.stream_usage --out_dir out-finewebedu-small-gdsm-s0 [--ckpt ckpt.pt] [--batches 20]

Per hyper-connection fold (attention / MLP of each block), on validation batches:
  read   mean H_pre weight per stream (how much of each stream feeds the branch), normalised to shares
  write  mean H_post weight per stream (how much of the branch output lands in each stream), shares
  norm   mean RMS norm of each residual stream entering the fold, shares
and for each the participation ratio PR = (Σ w)² / Σ w², the effective number of streams in use
(1 = one stream does everything, n = perfectly even).  Also the mean residual mixer H_res per fold
and the PR of its column mass (which source streams the mix draws on).  Writes <out_dir>/stream_usage.json.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyze.mixer_stats import load, recorders, val_batches  # noqa: E402


def shares(w):
    w = np.asarray(w, dtype=np.float64); s = w.sum()
    return (w / s).tolist() if s > 0 else w.tolist()


def participation(w):
    w = np.asarray(w, dtype=np.float64)
    return float(w.sum() ** 2 / (w ** 2).sum()) if (w ** 2).sum() > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ckpt", default="ckpt.pt")
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    model, ckpt = load(args.out_dir, args.device, args.ckpt)
    cfg = ckpt["config"]
    recorded = recorders(model)
    acc = {name: {"read": [], "write": [], "norm": [], "mixer": []} for name, _ in recorded}
    with torch.no_grad():
        for x, y in val_batches(cfg["dataset"], cfg["block_size"], args.batch_size, args.batches, args.device):
            with torch.autocast(device_type="cuda" if args.device.startswith("cuda") else "cpu", dtype=torch.bfloat16):
                model(x, y)
            for name, m in recorded:
                acc[name]["read"].append(m.last_pre.float().reshape(-1, m.last_pre.shape[-1]).mean(0).cpu())
                acc[name]["write"].append(m.last_post.float().reshape(-1, m.last_post.shape[-1]).mean(0).cpu())
                acc[name]["norm"].append(m.last_stream_norm.reshape(-1, m.last_stream_norm.shape[-1]).mean(0).cpu())
                acc[name]["mixer"].append(m.last_mixer.float().reshape(-1, *m.last_mixer.shape[-2:]).mean(0).cpu())
    layers = {}
    for name, a in acc.items():
        read = torch.stack(a["read"]).mean(0).abs().tolist(); write = torch.stack(a["write"]).mean(0).abs().tolist()
        norm = torch.stack(a["norm"]).mean(0).tolist(); mixer = torch.stack(a["mixer"]).mean(0)
        col_mass = mixer.abs().sum(0).tolist()
        layers[name] = {
            "read": read, "read_share": shares(read), "read_pr": participation(read),
            "write": write, "write_share": shares(write), "write_pr": participation(write),
            "norm": norm, "norm_share": shares(norm), "norm_pr": participation(norm),
            "mixer_mean": mixer.tolist(), "mixer_col_pr": participation(col_mass),
        }
    n = len(next(iter(layers.values()))["read"]) if layers else 0
    summary = {k: float(np.mean([l[k] for l in layers.values()])) for k in ("read_pr", "write_pr", "norm_pr", "mixer_col_pr")} if layers else {}
    report = {"out_dir": args.out_dir, "ckpt": args.ckpt, "hyper_conn_type": cfg["hyper_conn_type"], "iter": ckpt["iter_num"],
              "num_streams": n, "layers": layers, "summary_mean_pr": summary}
    with open(os.path.join(args.out_dir, "stream_usage.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(f"{report['hyper_conn_type']} iter {report['iter']} n={n}  mean PR: read {summary.get('read_pr', 0):.2f} write {summary.get('write_pr', 0):.2f} norm {summary.get('norm_pr', 0):.2f} mixer-cols {summary.get('mixer_col_pr', 0):.2f}")
    print(f"{'fold':26s} {'read shares':>28s} {'PR':>5s} | {'write shares':>28s} {'PR':>5s} | {'norm shares':>28s} {'PR':>5s}")
    for name, l in layers.items():
        fmt = lambda v: " ".join(f"{x:6.2f}" for x in v)
        print(f"{name:26s} {fmt(l['read_share']):>28s} {l['read_pr']:5.2f} | {fmt(l['write_share']):>28s} {l['write_pr']:5.2f} | {fmt(l['norm_share']):>28s} {l['norm_pr']:5.2f}")


if __name__ == "__main__":
    main()
