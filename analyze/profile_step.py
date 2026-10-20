"""Where does a training step spend its time and memory?  Single GPU, random tokens, no data loader.

usage: python -m analyze.profile_step --size small --hyper_conn_type gdsm --batch_size 16 [--compile] [--act_ckpt]
       [--steps 20] [--trace out.json]

Prints tokens/s (after warm-up), peak allocated memory, and the top CUDA-time operators with the share attributed to
the hyper-connection modules (record_function scopes 'hc_fold') versus attention / MLP / the rest.
"""

import argparse
import importlib.util
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import GPT, GPTConfig  # noqa: E402


def load_size(name):
    spec = importlib.util.spec_from_file_location("cfg", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", f"{name}_model.py"))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return {k: getattr(module, k) for k in ("n_layer", "n_head", "n_embd") if hasattr(module, k)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="small")
    ap.add_argument("--hyper_conn_type", default="gdsm")
    ap.add_argument("--hyper_conn_n", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--block_size", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--act_ckpt", action="store_true")
    ap.add_argument("--trace", default="")
    args = ap.parse_args()
    torch.manual_seed(0)
    cfg = GPTConfig(block_size=args.block_size, vocab_size=50304, bias=False, dropout=0.0, hyper_conn_n=args.hyper_conn_n,
                    hyper_conn_type=args.hyper_conn_type, act_ckpt=args.act_ckpt, **load_size(args.size))
    model = GPT(cfg).cuda().train()
    opt = model.configure_optimizers(0.1, 3e-4, (0.9, 0.95), "cuda")
    step_model = torch.compile(model) if args.compile else model
    x = torch.randint(0, 50304, (args.batch_size, args.block_size), device="cuda"); y = torch.roll(x, -1, 1)

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = step_model(x, y)
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)

    for _ in range(3):
        step()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    for _ in range(args.steps):
        step()
    torch.cuda.synchronize(); dt = time.time() - t0
    toks = args.steps * args.batch_size * args.block_size
    print(f"{args.size} {args.hyper_conn_type} mb={args.batch_size} compile={args.compile} act_ckpt={args.act_ckpt}: "
          f"{toks / dt / 1000:.1f}k tok/s, {dt / args.steps * 1000:.0f} ms/step, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], record_shapes=False) as prof:
        for _ in range(3):
            step()
        torch.cuda.synchronize()
    if args.trace:
        prof.export_chrome_trace(args.trace)
    rows = prof.key_averages()
    total = sum(r.self_device_time_total for r in rows) or 1
    print(f"top operators by self CUDA time (3 steps), share of {total / 1e3 / 3:.0f} ms/step:")
    for r in sorted(rows, key=lambda r: -r.self_device_time_total)[:25]:
        print(f"  {r.self_device_time_total / total * 100:5.1f}%  {r.key[:90]}")


if __name__ == "__main__":
    main()
