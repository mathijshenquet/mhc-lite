"""Upstream sHC realisation (hyper_conn.gdsm.shc_paper): unit row and column sums, deviation singular values equal to
|tanh| of the diagonal logits (so spectral norm < 1), H ≈ I at the paper's init, and a tiny GPT with
gdsm_radial="shc" trains a step.  Exits non-zero on failure."""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hyper_conn.gdsm import shc_paper  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402


def main():
    torch.manual_seed(0)
    n = 4
    logits = torch.randn(64, n, n) * 2
    one = torch.ones(())
    h = shc_paper(logits, one, one)
    sums = max((h.sum(-1) - 1).abs().max().item(), (h.sum(-2) - 1).abs().max().item())
    deviation = h - 1.0 / n
    singular = torch.linalg.svdvals(deviation)[..., : n - 1].sort(-1).values
    expected = torch.tanh(torch.diagonal(logits[..., : n - 1, : n - 1], dim1=-2, dim2=-1)).abs().sort(-1).values
    spectrum_error = (singular - expected).abs().max().item()
    init = shc_paper(torch.eye(n)[None] * 4.0, one, one)[0]
    init_error = (init - (1.0 / n + torch.tanh(torch.tensor(4.0)) * (torch.eye(n) - 1.0 / n))).abs().max().item()
    print(f"row/col sum error {sums:.1e} | deviation spectrum vs |tanh(diag)| {spectrum_error:.1e} | "
          f"init vs J + tanh(4)(I - J) {init_error:.1e} | max neg entry {(-h).clamp_min(0).max().item():.3f}")
    config = GPTConfig(n_layer=2, n_head=2, n_embd=32, block_size=16, vocab_size=64, dropout=0.0, bias=False,
                       hyper_conn_type="gdsm", hyper_conn_n=4, gdsm_radial="shc", gdsm_centred_mix=True)
    model = GPT(config)
    tokens = torch.randint(0, 64, (2, 16))
    _, loss = model(tokens, tokens)
    loss.backward()
    grads = [p.grad for n_, p in model.named_parameters() if "shc_gamma" in n_]
    print(f"tiny GPT loss {loss.item():.3f}; shc_gamma grads present {all(g is not None for g in grads)} ({len(grads)})")
    ok = sums < 1e-5 and spectrum_error < 1e-5 and init_error < 1e-5 and torch.isfinite(loss) and len(grads) == 2 * 2 * config.n_layer
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
