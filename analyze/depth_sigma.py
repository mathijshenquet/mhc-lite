"""σ₂ of the prefix composite mixer as a function of depth (how fast stream identity is averaged away).

usage: python -m analyze.depth_sigma --out_dir out-finewebedu-medium-gdsm-s0 [--batches 4]

For each validation token, multiplies the per-fold residual mixers in order and reports, after each fold,
the mean σ₂ (second singular value) of the prefix product, plus each fold's own mean σ₂.  Writes
<out_dir>/depth_sigma.json.
"""

import argparse
import json
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
    cfg = ckpt["config"]
    recorded = recorders(model)
    prefix, own = [], []
    with torch.no_grad():
        for x, y in val_batches(cfg["dataset"], cfg["block_size"], args.batch_size, args.batches, args.device):
            with torch.autocast(device_type="cuda" if args.device.startswith("cuda") else "cpu", dtype=torch.bfloat16):
                model(x, y)
            mixers = [m.last_mixer.float().reshape(-1, *m.last_mixer.shape[-2:]).cpu() for _, m in recorded]
            product = None; p_row, o_row = [], []
            for m in mixers:
                product = m if product is None else m @ product
                p_row.append(torch.linalg.svdvals(product)[:, 1].mean().item())
                o_row.append(torch.linalg.svdvals(m)[:, 1].mean().item())
            prefix.append(p_row); own.append(o_row)
    prefix = torch.tensor(prefix).mean(0).tolist(); own = torch.tensor(own).mean(0).tolist()
    names = [n for n, _ in recorded]
    json.dump({"hyper_conn_type": cfg["hyper_conn_type"], "folds": names, "prefix_sigma2": prefix, "fold_sigma2": own},
              open(os.path.join(args.out_dir, "depth_sigma.json"), "w"), indent=1)
    print(f"{cfg['hyper_conn_type']}: fold σ₂ (own)   " + " ".join(f"{v:.2f}" for v in own))
    print(f"{cfg['hyper_conn_type']}: prefix σ₂       " + " ".join(f"{v:.2f}" for v in prefix))


if __name__ == "__main__":
    main()
