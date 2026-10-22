"""dHC checks on a tiny CPU GPT (GDSM-tanh, n = 4): (1) diff_rank = n_embd with basis = write = I gives the same logits
as the full-rank model with the same weights; (2) with diff_rank = k and random writes, the stream differences entering
every fold have rank <= k.  Exits non-zero on failure."""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import GPT, GPTConfig  # noqa: E402


def config(rank, mode="free"):
    return GPTConfig(n_layer=3, n_head=2, n_embd=32, block_size=16, vocab_size=64, dropout=0.0, bias=False,
                     hyper_conn_type="gdsm", hyper_conn_n=4, gdsm_radial="tanh", gdsm_centred_mix=True, gdsm_diff_rank=rank, gdsm_diff_mode=mode)


def main():
    torch.manual_seed(0)
    full = GPT(config(0))
    # move off the zero/identity init so the mixers are non-trivial
    with torch.no_grad():
        for name, parameter in full.named_parameters():
            parameter.add_(0.05 * torch.randn_like(parameter))
    tokens = torch.randint(0, 64, (2, 16))
    reference, _ = full(tokens, tokens)

    identity = GPT(config(32))
    missing = identity.load_state_dict(full.state_dict(), strict=False)
    assert not missing.unexpected_keys, missing.unexpected_keys
    with torch.no_grad():
        identity.diff_basis.copy_(torch.eye(32))
        for module in identity.modules():
            if getattr(module, "diff_rank", 0) > 0:
                module.diff_write.copy_(torch.eye(32))
    same, _ = identity(tokens, tokens)
    error = (same - reference).abs().max().item()
    print(f"(1) identity dHC vs full model: max |logit diff| {error:.2e}")

    low = GPT(config(4))
    low.load_state_dict(full.state_dict(), strict=False)
    with torch.no_grad():
        for module in low.modules():
            if getattr(module, "diff_rank", 0) > 0:
                module.diff_write.normal_()
    ranks = []

    def record(module, args):
        streams = args[0].unflatten(0, (-1, 4)).float()                        # (b, s, t, d)
        diffs = (streams - streams.mean(1, keepdim=True)).reshape(-1, streams.shape[-1])
        singular = torch.linalg.svdvals(diffs)
        ranks.append(int((singular > 1e-4 * singular.max().clamp_min(1e-12)).sum()))

    for module in low.modules():
        if getattr(module, "diff_rank", 0) > 0:
            module.register_forward_pre_hook(record)
    low(tokens, tokens)
    print(f"(2) rank-4 dHC: numerical rank of the stream differences per fold {ranks}")
    residual = GPT(config(8, "residual"))
    residual.load_state_dict(full.state_dict(), strict=False)
    start, _ = residual(tokens, tokens)
    start_error = (start - reference).abs().max().item()
    with torch.no_grad():
        for module in residual.modules():
            if getattr(module, "diff_mode", "") == "residual" and getattr(module, "diff_rank", 0) > 0:
                module.diff_up.normal_()
    moved, _ = residual(tokens, tokens)
    moved_error = (moved - reference).abs().max().item()
    print(f"(3) residual mode: at init vs full model {start_error:.2e}; with diff_up != 0 it differs by {moved_error:.2e}")
    ok = error < 1e-5 and max(ranks) <= 4 and max(ranks) > 0 and start_error < 1e-5 and moved_error > 1e-3
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
