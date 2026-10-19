"""Per-layer statistics of the realised residual mixers H_res of a trained checkpoint.

usage: python -m analyze.mixer_stats --out_dir out-finewebedu-small-gdsm-s0 [--batches 50] [--device cuda]

Runs validation batches through the model with the hyper-connection modules recording their
per-token mixers, then reports per layer (attention and MLP folds separately):
  neg_mass      sum of negative entries / sum of |entries|            (0 for Birkhoff variants)
  dev_norm      mean ||H − J/n||₂   (GDSM: the gate; 1 = on the sphere, 0 = full averaging)
  off_identity  mean ||H − I||_F
  fwd_gain / bwd_gain   mean max |row sum| / max |column sum|  (the mHC paper's propagation metric)
and the depth product's fwd/bwd gain and σ₂ (detail-subspace contraction).  Also reports a PAIRED
validation loss: mean cross-entropy over the same seeded validation batches for every checkpoint,
so evaluation sampling noise cancels between variants.  Writes <out_dir>/mixer_stats.json.
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import GPT, GPTConfig  # noqa: E402


def load(out_dir, device):
    ckpt = torch.load(os.path.join(out_dir, "ckpt.pt"), map_location=device, weights_only=False)
    model = GPT(GPTConfig(**ckpt["model_args"]))
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    return model.to(device).eval(), ckpt


def val_batches(dataset, block_size, batch_size, n, device, seed=0):
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", dataset, "val.bin")
    data = np.memmap(path, dtype=np.uint16, mode="r")
    rng = np.random.default_rng(seed)
    for _ in range(n):
        ix = rng.integers(len(data) - block_size - 1, size=batch_size)
        x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
        yield x.to(device), y.to(device)


RECORDING = ("ManifoldConstrainedHyperConnections", "MHCLite", "GDSMHyperConnections")


def recorders(model):
    """Hyper-connection modules that implement the opt-in mixer recorder (plain / hc do not)."""
    found = []
    for name, module in model.named_modules():
        if name.endswith(("hc_attn", "hc_mlp")) and type(module).__name__ in RECORDING:
            module.record_mixers = True
            found.append((name, module))
    return found


def stats(matrix):
    """matrix: (N, n, n) fp32."""
    n = matrix.shape[-1]
    eye = torch.eye(n)
    absolute = matrix.abs()
    return {
        "neg_mass": (matrix.clamp_max(0).abs().sum() / absolute.sum().clamp_min(1e-30)).item(),
        "dev_norm": torch.linalg.matrix_norm(matrix - 1.0 / n, ord=2).mean().item(),
        "off_identity": (matrix - eye).flatten(1).norm(dim=-1).mean().item(),
        "fwd_gain": matrix.sum(-1).abs().amax(-1).mean().item(),
        "bwd_gain": matrix.sum(-2).abs().amax(-1).mean().item(),
        "sigma2": torch.linalg.svdvals(matrix)[:, 1].mean().item() if n > 1 else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batches", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    model, ckpt = load(args.out_dir, args.device)
    cfg = ckpt["config"]
    recorded = recorders(model)
    per_layer = {name: [] for name, _ in recorded}
    composite_fwd, composite_bwd, composite_sigma2, paired_losses = [], [], [], []
    with torch.no_grad():
        for x, y in val_batches(cfg["dataset"], cfg["block_size"], args.batch_size, args.batches, args.device):
            with torch.autocast(device_type="cuda" if args.device.startswith("cuda") else "cpu", dtype=torch.bfloat16):
                _, loss = model(x, y)
            paired_losses.append(loss.item())
            if not recorded:
                continue
            mixers = []
            for name, module in recorded:
                m = module.last_mixer.float().reshape(-1, module.last_mixer.shape[-2], module.last_mixer.shape[-1]).cpu()
                per_layer[name].append(m)
                mixers.append(m)
            product = mixers[0].clone()
            for m in mixers[1:]:
                product = m @ product
            composite_fwd.append(product.sum(-1).abs().amax(-1).mean().item())
            composite_bwd.append(product.sum(-2).abs().amax(-1).mean().item())
            composite_sigma2.append(torch.linalg.svdvals(product)[:, 1].mean().item())
    report = {
        "out_dir": args.out_dir, "hyper_conn_type": cfg["hyper_conn_type"], "iter": ckpt["iter_num"],
        "val_loss": ckpt["best_val_loss"] if isinstance(ckpt["best_val_loss"], float) else float(ckpt["best_val_loss"]),
        "layers": {name: stats(torch.cat(ms)) for name, ms in per_layer.items()},
        "composite": {"fwd_gain": float(np.mean(composite_fwd)), "bwd_gain": float(np.mean(composite_bwd)), "sigma2": float(np.mean(composite_sigma2))} if recorded else None,
        "paired_val_loss": float(np.mean(paired_losses)),
        "paired_val_sem": float(np.std(paired_losses) / np.sqrt(len(paired_losses))),
        "paired_batches": args.batches,
    }
    with open(os.path.join(args.out_dir, "mixer_stats.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(f"{report['hyper_conn_type']}  iter {report['iter']}  val {report['val_loss']:.4f}  paired val {report['paired_val_loss']:.4f} ± {report['paired_val_sem']:.4f} ({args.batches} batches)")
    print(f"{'layer':28s} {'neg':>6s} {'dev':>6s} {'offI':>6s} {'fwd':>6s} {'bwd':>6s} {'sig2':>6s}")
    for name, s in report["layers"].items():
        print(f"{name:28s} {s['neg_mass']:6.3f} {s['dev_norm']:6.3f} {s['off_identity']:6.3f} {s['fwd_gain']:6.3f} {s['bwd_gain']:6.3f} {s['sigma2']:6.3f}")
    c = report["composite"]
    if c:
        print(f"composite: fwd {c['fwd_gain']:.3f} bwd {c['bwd_gain']:.3f} sigma2 {c['sigma2']:.3f}")


if __name__ == "__main__":
    main()
