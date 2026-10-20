"""Distribution of the raw spectral norm σ_max(A') of the centred mixing logits per token and fold (GDSM only):
the normalised direction A'/σ_max has a Jacobian ∝ 1/σ_max, so a small lower tail means gradient blow-ups.
usage: python -m analyze.raw_sigma --out_dir out-finewebedu-large-gdsm-s0 [--ckpt ckpt.pt] [--batches 4]"""

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
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    model, ckpt = load(args.out_dir, args.device, args.ckpt)
    cfg = ckpt["config"]; rec = recorders(model)
    sig = {n: [] for n, _ in rec}; gate = {n: [] for n, _ in rec}
    with torch.no_grad():
        for x, y in val_batches(cfg["dataset"], cfg["block_size"], args.batch_size, args.batches, args.device):
            with torch.autocast(device_type="cuda" if args.device.startswith("cuda") else "cpu", dtype=torch.bfloat16):
                model(x, y)
            for n, m in rec:
                sig[n].append(m.last_raw_sigma.flatten().cpu()); gate[n].append(m.last_raw_gate.flatten().cpu())
    print(f"{cfg['hyper_conn_type']} iter {ckpt['iter_num']}: raw σ_max(A') per token;  fold: median  p1  min  frac<0.1  frac<0.3 | gate median min")
    allsig = []
    for n in sig:
        s = torch.cat(sig[n]); g = torch.cat(gate[n]); allsig.append(s)
        q = torch.quantile(s, torch.tensor([0.5, 0.01]))
        print(f"  {n[14:]:14s} {q[0]:6.3f} {q[1]:6.3f} {s.min():6.3f} {(s < 0.1).float().mean():7.3f} {(s < 0.3).float().mean():7.3f} | {g.median():5.2f} {g.min():5.2f}")
    s = torch.cat(allsig)
    print(f"  ALL            median {s.median():.3f} p1 {torch.quantile(s, 0.01):.3f} min {s.min():.4f} frac<0.1 {(s < 0.1).float().mean():.4f} frac<0.3 {(s < 0.3).float().mean():.4f}")


if __name__ == "__main__":
    main()
