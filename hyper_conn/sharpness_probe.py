"""Live preconditioned-sharpness probe for training: λ_max(P^-1 H) with Adam's current preconditioner
P = diag(√v̂ + eps), by warm-started power iteration on Hessian-vector products of the uncompiled model.
Adam (β1 = 0.9) is stable while λ < 38/lr (Cohen et al., adaptive edge of stability)."""

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


class SharpnessProbe:
    def __init__(self, model, optimizer, iters=5, block=512, batch=1):
        self.model, self.opt, self.iters, self.block, self.batch = model, optimizer, iters, block, batch
        self.params = [p for g in optimizer.param_groups for p in g["params"] if p.requires_grad]
        names = {id(p): n for n, p in model.named_parameters()}
        self.scalar_mask = [names.get(id(p), "").endswith("_scale") and p.dim() <= 1 for p in self.params]
        self.v_full = None; self.v_scal = None

    def _dinv(self):
        out = []
        for p in self.params:
            st = self.opt.state.get(p)
            if not st or "exp_avg_sq" not in st:
                return None
            g = next(g for g in self.opt.param_groups if any(q is p for q in g["params"]))
            b2 = g["betas"][1]; eps = g["eps"]; t = float(st["step"])
            out.append((st["exp_avg_sq"].float() / (1 - b2 ** t)).sqrt().add(eps).rsqrt())
        return out

    def _power(self, hvp, dinv, v, mask):
        if v is None:
            v = [torch.randn_like(p, dtype=torch.float32) for p in self.params]
        if mask is not None:
            v = [vi if m else torch.zeros_like(vi) for vi, m in zip(v, mask)]
        nrm = torch.sqrt(sum((vi ** 2).sum() for vi in v)); v = [vi / nrm for vi in v]
        lam = float("nan")
        for _ in range(self.iters):
            hu = hvp([vi * di for vi, di in zip(v, dinv)])
            w = [h.float() * di for h, di in zip(hu, dinv)]
            if mask is not None:
                w = [wi if m else torch.zeros_like(wi) for wi, m in zip(w, mask)]
            lam = sum((wi * vi).sum() for wi, vi in zip(w, v)).item()
            nrm = torch.sqrt(sum((wi ** 2).sum() for wi in w)).clamp_min(1e-30); v = [wi / nrm for wi in w]
        return lam, v

    def measure(self, x, y, device_type="cuda"):
        dinv = self._dinv()
        if dinv is None:
            return None
        x, y = x[: self.batch, : self.block], y[: self.batch, : self.block]
        was_training = self.model.training; self.model.train()
        with sdpa_kernel(SDPBackend.MATH), torch.autocast(device_type, dtype=torch.bfloat16, enabled=device_type == "cuda"):
            _, loss = self.model(x, y)
        grads = torch.autograd.grad(loss, self.params, create_graph=True, allow_unused=True)
        grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, self.params)]

        def hvp(vec):
            out = torch.autograd.grad(grads, self.params, grad_outputs=vec, retain_graph=True, allow_unused=True)
            return [o if o is not None else torch.zeros_like(p) for o, p in zip(out, self.params)]

        full, self.v_full = self._power(hvp, dinv, self.v_full, None)
        scal, self.v_scal = self._power(hvp, dinv, self.v_scal, self.scalar_mask) if any(self.scalar_mask) else (float("nan"), None)
        del grads
        self.model.zero_grad(set_to_none=True); self.model.train(was_training)
        return {"sharpness/precond": full, "sharpness/precond_scalars": scal}
