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
    for _, m in mods:
        s = m.num_residual_streams
        r = m.last_resid_in.float(); r = r.reshape(-1, s, r.shape[-1])
        rn = r / r.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        G = rn @ rn.transpose(-1, -2)
        off = G[:, ~torch.eye(s, dtype=torch.bool, device=G.device)]
        cos.append(off.mean().item()); pr.append((s * s / G.pow(2).sum((-1, -2))).mean().item())
        m.record_cancel = False; m.last_resid_in = None; m.last_mix_out = None
    k = len(cos)
    return {"collapse/cos_mean": sum(cos) / k, "collapse/cos_max": max(cos), "collapse/cos_first": cos[0], "collapse/cos_last": cos[-1],
            "collapse/eff_streams_mean": sum(pr) / k, "collapse/eff_streams_min": min(pr)}
