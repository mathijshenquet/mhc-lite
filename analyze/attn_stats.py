"""Attention-logit and activation statistics per layer, to test whether bf16 GDSM-L drives attention logits up
(softmax saturation) where fp32 / static mixing do not.  Forward only, fp32, on a few validation sequences.
Per layer: max |q·k|/sqrt(d) over causal positions, mean attention entropy (nats), mean ‖q‖ and ‖k‖ per head-vector,
the RMS of the attention block's input (after the read + norm), and the max |MLP pre-activation|.
usage: python -m analyze.attn_stats --runs <out_dir>:<ckpt> [...] [--batches 2 --block 512]"""

import argparse
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyze.mixer_stats import load, val_batches  # noqa: E402


def layer_stats(out_dir, ckpt, batches, block):
    model, checkpoint = load(out_dir, "cpu", ckpt)
    model = model.float().eval()
    config, iteration = checkpoint["config"], checkpoint["iter_num"]
    del checkpoint
    rows = {}
    hooks = []
    for index, block_module in enumerate(model.transformer.h):
        attention = block_module.branch_attn[1]

        def attention_hook(module, inputs, output, index=index):
            x = inputs[0]
            batch, time, channels = x.shape
            heads = module.n_head
            q, k, _ = module.c_attn(x).split(module.n_embd, dim=2)
            q = q.view(batch, time, heads, -1).transpose(1, 2)
            k = k.view(batch, time, heads, -1).transpose(1, 2)
            logits = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
            mask = torch.ones(time, time, dtype=torch.bool).tril()
            logits = logits.masked_fill(~mask, float("-inf"))
            probabilities = logits.softmax(-1)
            entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(-1)
            row = rows.setdefault(index, {"logit_max": 0.0, "entropy": [], "q": [], "k": [], "in_rms": [], "mlp_max": 0.0})
            row["logit_max"] = max(row["logit_max"], logits.masked_fill(~mask, 0).abs().max().item())
            row["entropy"].append(entropy.mean().item())
            row["q"].append(q.norm(dim=-1).mean().item()); row["k"].append(k.norm(dim=-1).mean().item())
            row["in_rms"].append(x.pow(2).mean().sqrt().item())

        def mlp_hook(module, inputs, output, index=index):
            pre = module.c_fc(inputs[0])
            rows.setdefault(index, {"logit_max": 0.0, "entropy": [], "q": [], "k": [], "in_rms": [], "mlp_max": 0.0})
            rows[index]["mlp_max"] = max(rows[index]["mlp_max"], pre.abs().max().item())

        hooks.append(attention.register_forward_hook(attention_hook))
        hooks.append(block_module.branch_mlp[1].register_forward_hook(mlp_hook))
    with torch.no_grad():
        for x, y in val_batches(config["dataset"], block, 1, batches, "cpu", seed=3):
            model(x, y)
    for hook in hooks:
        hook.remove()
    mean = lambda values: sum(values) / len(values)
    return iteration, [(i, r["logit_max"], mean(r["entropy"]), mean(r["q"]), mean(r["k"]), mean(r["in_rms"]), r["mlp_max"])
                       for i, r in sorted(rows.items())]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True, help="<out_dir>:<ckpt file>")
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--block", type=int, default=512)
    args = parser.parse_args()
    for spec in args.runs:
        out_dir, ckpt = spec.rsplit(":", 1)
        iteration, rows = layer_stats(out_dir, ckpt, args.batches, args.block)
        name = os.path.basename(out_dir.rstrip("/")).replace("out-finewebedu-large-", "")
        print(f"== {name} iter {iteration}: layer  logit_max  entropy  |q|  |k|  attn_in_rms  mlp_pre_max")
        for i, logit_max, entropy, qn, kn, rms, mlp in rows:
            if i in (0, 3, 7, 11, 15, 19, 23) or i == len(rows) - 1:
                print(f"   L{i:02d}  {logit_max:8.1f}  {entropy:6.2f}  {qn:6.1f}  {kn:6.1f}  {rms:8.3f}  {mlp:8.1f}", flush=True)


if __name__ == "__main__":
    main()
