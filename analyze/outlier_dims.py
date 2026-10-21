"""Are the outlier activations of failing bf16 GDSM-L concentrated in a few fixed dimensions ("massive activations")?
Per layer, on a few validation sequences, forward only in fp32:
  stream:  per-channel RMS of the residual streams entering the MLP fold (over tokens and streams); report the top-3
           channel ids and their RMS relative to the median channel RMS
  ln_out:  the same for the MLP's LayerNorm output (what c_fc actually sees)
  hidden:  per-unit max |c_fc pre-activation|; report how many units exceed 20 and the top-3 unit ids
usage: python -m analyze.outlier_dims --runs <out_dir>:<ckpt> [...]"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyze.mixer_stats import load, val_batches  # noqa: E402

LAYERS = (3, 7, 11, 15, 19)


def top(values, k=3):
    ranked = values.argsort(descending=True)[:k]
    median = values.median().clamp_min(1e-12)
    return " ".join(f"{int(i)}:{(values[i] / median).item():.1f}x" for i in ranked)


def analyse(out_dir, ckpt, batches, block):
    model, checkpoint = load(out_dir, "cpu", ckpt)
    model = model.float().eval()
    config, iteration = checkpoint["config"], checkpoint["iter_num"]
    del checkpoint
    sums = {}
    hooks = []
    for index in LAYERS:
        block_module = model.transformer.h[index]
        fold = block_module.hc_mlp
        streams = fold.num_residual_streams
        original = fold.width_connection

        def recording(residuals, original=original, index=index, streams=streams):
            x = residuals.float().unflatten(0, (-1, streams))                 # (b, s, t, d)
            sums.setdefault(index, {}).setdefault("stream", []).append(x.pow(2).mean(dim=(0, 1, 2)))
            return original(residuals)

        fold.width_connection = recording

        def ln_hook(module, inputs, output, index=index):
            sums.setdefault(index, {}).setdefault("ln_out", []).append(output.float().pow(2).mean(dim=(0, 1)))

        def fc_hook(module, inputs, output, index=index):
            sums.setdefault(index, {}).setdefault("hidden", []).append(output.float().abs().amax(dim=(0, 1)))

        hooks.append(block_module.branch_mlp[0].register_forward_hook(ln_hook))
        hooks.append(block_module.branch_mlp[1].c_fc.register_forward_hook(fc_hook))
    with torch.no_grad():
        for x, y in val_batches(config["dataset"], block, 1, batches, "cpu", seed=3):
            model(x, y)
    for hook in hooks:
        hook.remove()
    name = os.path.basename(out_dir.rstrip("/")).replace("out-finewebedu-large-", "")
    print(f"== {name} iter {iteration}")
    for index in LAYERS:
        record = sums[index]
        stream_rms = torch.stack(record["stream"]).mean(0).sqrt()
        ln_rms = torch.stack(record["ln_out"]).mean(0).sqrt()
        hidden_max = torch.stack(record["hidden"]).amax(0)
        print(f"   L{index:02d} stream top {top(stream_rms)} | ln_out top {top(ln_rms)} | hidden >20: {int((hidden_max > 20).sum())}"
              f" top {' '.join(f'{int(i)}:{hidden_max[i].item():.0f}' for i in hidden_max.argsort(descending=True)[:3])}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--block", type=int, default=512)
    args = parser.parse_args()
    for spec in args.runs:
        out_dir, ckpt = spec.rsplit(":", 1)
        analyse(out_dir, ckpt, args.batches, args.block)


if __name__ == "__main__":
    main()
