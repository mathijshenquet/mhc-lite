"""Stream-collapse probe: how similar are the n residual streams entering each hyper-connection fold?
Per fold, over the tokens of one batch: mean pairwise cosine of the streams, and the effective number of streams
(participation ratio of their normalised Gram matrix: n = independent, 1 = collapsed into one)."""

import torch


def hc_modules(model):
    return [(n, m) for n, m in model.named_modules() if n.endswith(("hc_attn", "hc_mlp"))]


@torch.no_grad()
def measure(model, x, y, device_type="cuda"):
    mods = hc_modules(model)
    if not mods:
        return None
    for _, m in mods:
        m.record_cancel = True
    was_training = model.training; model.eval()
    try:
        with torch.autocast(device_type, dtype=torch.bfloat16, enabled=device_type == "cuda"):
            model(x, y)
    finally:
        model.train(was_training)
    cos, pr = [], []
    shared_norm, diff_norm, mix_err = [], [], []
    for _, m in mods:
        s = m.num_residual_streams
        r = m.last_resid_in.float(); r = r.reshape(-1, s, r.shape[-1])
        rn = r / r.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        G = rn @ rn.transpose(-1, -2)
        off = G[:, ~torch.eye(s, dtype=torch.bool, device=G.device)]
        cos.append(off.mean().item()); pr.append((s * s / G.pow(2).sum((-1, -2))).mean().item())
        shared = r.mean(dim=-2, keepdim=True)
        shared_norm.append(shared.norm(dim=-1).mean().item()); diff_norm.append((r - shared).norm(dim=-1).mean().item())
        alpha = getattr(m, "last_alpha", None)
        if alpha is not None and m.last_mix_out is not None:
            # the mixing as it ran (autocast) against the same inputs mixed in fp64: the error the precision adds
            exact = torch.einsum("...asbt,...asd->...btd", alpha.double(), m.last_resid_in.double())
            low = m.last_mix_out.detach().double()
            differences = m.last_resid_in.double() - m.last_resid_in.double().mean(dim=-2, keepdim=True)
            # skip folds where the streams are still (nearly) identical (the first fold): no differences to compare to
            if differences.norm() > 1e-4 * m.last_resid_in.double().norm():
                mix_err.append(((low - exact).norm() / differences.norm()).item())
            else:
                mix_err.append(float("nan"))
        m.last_alpha = None
        m.record_cancel = False; m.last_resid_in = None; m.last_mix_out = None
    k = len(cos)
    picks = sorted({min(k - 1, round(fraction * (k - 1))) for fraction in (0.05, 0.25, 0.5, 0.75, 1.0)})
    extra = {}
    for i in picks:
        extra[f"stream/shared_f{i + 1}"] = shared_norm[i]; extra[f"stream/diff_f{i + 1}"] = diff_norm[i]
        if mix_err and mix_err[i] == mix_err[i]:
            extra[f"mixerr/rel_f{i + 1}"] = mix_err[i]
    extra["stream/shared_max"] = max(shared_norm); extra["stream/diff_max"] = max(diff_norm)
    finite = [e for e in mix_err if e == e]
    if finite:
        extra["mixerr/rel_mean"] = sum(finite) / len(finite); extra["mixerr/rel_max"] = max(finite)
    return extra | {"collapse/cos_mean": sum(cos) / k, "collapse/cos_max": max(cos), "collapse/cos_first": cos[0], "collapse/cos_last": cos[-1],
            "collapse/eff_streams_mean": sum(pr) / k, "collapse/eff_streams_min": min(pr)}
