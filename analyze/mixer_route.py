"""How much of the gradient on a middle MLP's input weights comes through GDSM's per-token mixer (the stream ->
later residual-mixing logits -> B path, which lite's saturated softmax and static mixing do not have)?
On one checkpoint and the same batches: the gradient of every c_fc weight with the full graph, and with that route
cut (forward identical; `cut_mixer_route` detaches the residual-mixing logits from the stream).  Per layer: norm of
both, their cosine, and for the 5 hidden units with the largest weight rows, the per-row gradient norm with / without.
usage: python -m analyze.mixer_route --out_dir <dir> --ckpt ckpt.pt [--batches 2 --block 512]"""

import argparse
import os
import sys

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyze.mixer_stats import load, val_batches  # noqa: E402

LAYERS = (3, 7, 11, 15, 19)


def gradients(model, batches, cut):
    for module in model.modules():
        if hasattr(module, "num_residual_streams"):
            module.cut_mixer_route = cut
    model.zero_grad(set_to_none=True)
    for x, y in batches:
        with sdpa_kernel(SDPBackend.MATH):
            _, loss = model(x, y)
        (loss / len(batches)).backward()
    return {i: model.transformer.h[i].branch_mlp[1].c_fc.weight.grad.detach().clone() for i in LAYERS}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True); parser.add_argument("--ckpt", default="ckpt.pt")
    parser.add_argument("--batches", type=int, default=2); parser.add_argument("--block", type=int, default=512)
    args = parser.parse_args()
    model, checkpoint = load(args.out_dir, "cpu", args.ckpt)
    model = model.float().train(); config, iteration = checkpoint["config"], checkpoint["iter_num"]; del checkpoint
    batches = list(val_batches(config["dataset"], args.block, 1, args.batches, "cpu", seed=11))
    full = gradients(model, batches, cut=False)
    cut = gradients(model, batches, cut=True)
    print(f"{os.path.basename(args.out_dir.rstrip('/'))} iter {iteration}: c_fc gradient, full graph vs mixer route cut")
    for i in LAYERS:
        weight = model.transformer.h[i].branch_mlp[1].c_fc.weight.detach()
        biggest = weight.norm(dim=1).argsort(descending=True)[:5]
        f, c = full[i], cut[i]
        cosine = (f * c).sum() / (f.norm() * c.norm()).clamp_min(1e-30)
        rows = " ".join(f"{int(u)}:{f[u].norm().item():.2e}/{c[u].norm().item():.2e}" for u in biggest)
        print(f"  L{i:02d} |g| full {f.norm().item():.3e} cut {c.norm().item():.3e} cos {cosine.item():+.3f} | biggest rows full/cut {rows}", flush=True)


if __name__ == "__main__":
    main()
