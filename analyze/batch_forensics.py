"""Spike forensics: which batches give GDSM-L large gradients, and does mHC-lite see the same batches as hard?
usage: python -m analyze.batch_forensics --ckpts /root/ck/gdsm/ckpt-1000.pt /root/ck/lite/ckpt-1000.pt --n 300
Same seeded training batches (1 × block tokens) for every checkpoint.  Per batch: loss, total gradient norm, the
gain scalars' and mixing logits' gradient norms; token statistics (distinct tokens, longest repeated-token run, EOT
count, share of the batch's most frequent token, mean token frequency rank).  Writes a CSV next to the first ckpt,
prints correlations across batches and the top batches with a decoded snippet."""

import argparse
import csv
import os
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyze.mixer_stats import load  # noqa: E402


def batches(dataset, block, n, seed):
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", dataset, "train.bin")
    data = np.memmap(path, dtype=np.uint16, mode="r"); rng = np.random.default_rng(seed)
    for _ in range(n):
        i = int(rng.integers(len(data) - block - 1))
        yield i, torch.from_numpy(data[i:i + block].astype(np.int64))[None], torch.from_numpy(data[i + 1:i + 1 + block].astype(np.int64))[None]


def token_stats(x, freq_rank):
    t = x[0].tolist(); c = Counter(t); run = best = 1
    for a, b in zip(t, t[1:]):
        run = run + 1 if a == b else 1; best = max(best, run)
    return {"distinct": len(c), "max_run": best, "eot": c.get(50256, 0), "top_share": c.most_common(1)[0][1] / len(t),
            "mean_rank": float(np.mean([freq_rank.get(v, 50000) for v in t]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--block", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = {}
    freq_rank = None
    for ck in args.ckpts:
        model, ckpt = load(os.path.dirname(ck), "cpu", os.path.basename(ck)); model = model.to(dev).train()
        cfg = ckpt["config"]; kind = ckpt["model_args"]["hyper_conn_type"]; del ckpt
        if freq_rank is None:
            path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", cfg["dataset"], "train.bin")
            sample = np.memmap(path, dtype=np.uint16, mode="r")[: 20_000_000]
            counts = np.bincount(sample, minlength=50304); order = np.argsort(-counts)
            freq_rank = {int(t): r for r, t in enumerate(order)}
        scal = [p for n, p in model.named_parameters() if n.endswith("_scale") and p.dim() <= 1]
        mix = [p for n, p in model.named_parameters() if n.endswith(("dynamic_alpha_fn", "static_alpha"))]
        for k, (i, x, y) in enumerate(batches(cfg["dataset"], args.block, args.n, args.seed)):
            x, y = x.to(dev), y.to(dev); model.zero_grad(set_to_none=True)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                _, loss = model(x, y)
            loss.backward()
            gn = lambda ps: float(torch.sqrt(sum((p.grad.float() ** 2).sum() for p in ps if p.grad is not None)))
            r = rows.setdefault(k, {"offset": i, **token_stats(x.cpu(), freq_rank)})
            r[f"{kind}_loss"] = loss.item(); r[f"{kind}_grad"] = gn(model.parameters()); r[f"{kind}_grad_scal"] = gn(scal); r[f"{kind}_grad_mix"] = gn(mix)
        del model; torch.cuda.empty_cache() if dev == "cuda" else None
    keys = list(next(iter(rows.values())).keys())
    out = os.path.join(os.path.dirname(args.ckpts[0]), "batch_forensics.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["batch"] + keys); w.writeheader()
        for k, r in rows.items(): w.writerow({"batch": k, **r})
    arr = lambda key: np.array([r.get(key, np.nan) for r in rows.values()], dtype=float)
    kinds = sorted({k.split("_")[0] if not k.startswith("mhc_lite") else "mhc_lite" for k in keys if k.endswith("_grad")})
    print(f"wrote {out} ({len(rows)} batches)")
    for kd in kinds:
        g = arr(f"{kd}_grad"); print(f"{kd:9s} grad norm: median {np.median(g):.3f}  p90 {np.percentile(g, 90):.3f}  max {g.max():.3f}  max/median {g.max() / np.median(g):.1f}")
    if len(kinds) == 2:
        a, b = arr(f"{kinds[0]}_grad"), arr(f"{kinds[1]}_grad")
        print(f"rank correlation of per-batch grad norms {kinds[0]} vs {kinds[1]}: {np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1]:+.2f}")
    g = arr("gdsm_grad") if "gdsm" in kinds else arr(f"{kinds[0]}_grad")
    for s in ["distinct", "max_run", "eot", "top_share", "mean_rank", "gdsm_loss" if "gdsm" in kinds else f"{kinds[0]}_loss"]:
        v = arr(s); print(f"  rank corr(grad, {s}): {np.corrcoef(np.argsort(np.argsort(g)), np.argsort(np.argsort(v)))[0, 1]:+.2f}")
    try:
        import tiktoken; enc = tiktoken.get_encoding("gpt2")
    except Exception:
        enc = None
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "fineweb_edu", "train.bin")
    data = np.memmap(path, dtype=np.uint16, mode="r")
    print("top batches by GDSM grad norm:")
    for k in np.argsort(-g)[:6]:
        r = rows[int(k)]
        snippet = enc.decode(data[r["offset"]: r["offset"] + 60].tolist()).replace("\n", " ")[:110] if enc else ""
        others = " ".join(f"{kd} {r.get(kd + '_grad', float('nan')):.2f}" for kd in kinds)
        print(f"  batch {int(k):3d} {others} | distinct {r['distinct']} run {r['max_run']} eot {r['eot']} | {snippet!r}")


if __name__ == "__main__":
    main()
