import sys, torch
sys.path.insert(0, ".")
from model import GPT, GPTConfig
from hyper_conn.collapse_probe import measure
x = torch.randint(0, 65, (2, 32), generator=torch.Generator().manual_seed(1))
def build(t="gdsm", **kw):
    torch.manual_seed(0); import random; random.seed(0)
    m = GPT(GPTConfig(n_layer=2, n_head=2, n_embd=64, block_size=32, bias=False, vocab_size=65, dropout=0.0, hyper_conn_n=4, hyper_conn_type=t, **kw))
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "alpha" in n or "scale" in n: p.add_(0.3 * torch.randn_like(p))
    return m.eval()
with torch.no_grad():
    base = build()(x, x)[1].item(); again = build()(x, x)[1].item()
    print("baseline vs baseline:", abs(base - again))
    for kw in [dict(gdsm_centred_mix=True), dict(gdsm_fp32_logits=True)]:
        print(kw, "fp32 loss diff vs baseline:", abs(build(**kw)(x, x)[1].item() - base))
    # bf16 autocast on CPU: centred mixing should not change the loss beyond bf16 noise
    with torch.autocast("cpu", dtype=torch.bfloat16):
        print("bf16 baseline", round(build()(x, x)[1].item(), 4), "bf16 centred", round(build(gdsm_centred_mix=True)(x, x)[1].item(), 4), "fp32", round(base, 4))
for t in ["gdsm", "mhc_lite"]:
    r = measure(build(t), x, x, device_type="cpu")
    print(t, {k: round(v, 4) for k, v in r.items() if k.startswith(("stream/", "mixerr/"))})
