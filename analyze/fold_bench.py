"""Micro-benchmark of ONE hyper-connection fold (width + depth connection, no branch): where does its time go?
usage: python -m analyze.fold_bench --hyper_conn_type gdsm --n_embd 512 --tokens 16384 [--compile]
Times forward+backward of (a) the full fold, (b) the fold with the mixer realisation replaced by a fixed identity map
(no sigma / Sinkhorn / permutation mixture), (c) the norm + logit GEMM alone, (d) a plain residual add, on the same
(batch·streams, T, d) input.  Differences give the cost of the realisation, of the read/mix/write einsums and layout."""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import GPT, GPTConfig  # noqa: E402


def timeit(fn, iters=20):
    for _ in range(5):
        fn()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyper_conn_type", default="gdsm")
    ap.add_argument("--n_embd", type=int, default=512)
    ap.add_argument("--tokens", type=int, default=16384, help="batch × sequence tokens (S step: 16 × 1024)")
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()
    cfg = GPTConfig(block_size=1024, vocab_size=50304, n_layer=1, n_head=8, n_embd=args.n_embd, bias=False, dropout=0.0, hyper_conn_n=4, hyper_conn_type=args.hyper_conn_type)
    hc = GPT(cfg).cuda().transformer.h[0].hc_mlp
    b, t = args.tokens // 1024, 1024
    x = torch.randn(b * 4, t, args.n_embd, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    branch_out = torch.randn(b, t, args.n_embd, device="cuda", dtype=torch.bfloat16)

    def full():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            bi, res, kw = hc.width_connection(x)
            y = hc.depth_connection(branch_out, res, **kw)
        y.float().sum().backward()

    def norm_gemm_only():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            from einops import rearrange
            r = rearrange(x, '(b s) ... d -> b ... s d', s=4)
            normed = hc.norm(rearrange(r, 'b ... s d -> b ... (s d)'))
            fused = normed @ torch.cat([hc.dynamic_alpha_fn] + ([hc.dynamic_gate_fn] if hasattr(hc, "dynamic_gate_fn") else []) + [hc.dynamic_beta_fn], dim=-1)
        fused.float().sum().backward()

    def plain_add():
        y = x[: b] + branch_out
        y.float().sum().backward()

    f_full = torch.compile(full) if args.compile else full
    f_ng = torch.compile(norm_gemm_only) if args.compile else norm_gemm_only
    f_pa = torch.compile(plain_add) if args.compile else plain_add
    t_full, t_ng, t_pa = timeit(f_full), timeit(f_ng), timeit(f_pa)
    print(f"{args.hyper_conn_type} d={args.n_embd} tokens={args.tokens} compile={args.compile}: fold fwd+bwd {t_full:.2f} ms | norm+GEMM {t_ng:.2f} ms | plain add {t_pa:.2f} ms | rest (realisation + read/mix/write + layout) {t_full - t_ng:.2f} ms")


if __name__ == "__main__":
    main()
