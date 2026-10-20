from __future__ import annotations
from typing import Callable

from functools import partial
from random import randrange

import torch
from torch import nn, cat
import torch.nn.functional as F
from torch.nn import Module, Sequential
from torch.utils._pytree import tree_flatten, tree_unflatten

from einops import rearrange, repeat, reduce, einsum
from einops.layers.torch import Rearrange, Reduce

"""
GDSM residual mixing (copied from mhc.py; only the residual-matrix realisation differs).

H_res = J/n + g · A'/σ_max(A'),  A' = double-centred logits, g = clamp(ρ, 0, 1).
Unit row and column sums with signed entries, ||H_res − J/n||₂ = g ≤ 1 (the spectral unit
ball of maps on the complement of 1, i.e. conv O(n−1) ⊃ Birkhoff).  The magnitude g is its
own per-token logit so the optimiser can move radially (a pure clip has no radial gradient
once σ_max > 1).  Exact in one pass; no iterations, no trace in the backward.
"""

"""
ein notation:
b - batch
d - feature dimension
s - residual streams
t - residual streams + num branch inputs
f - number of fractions (division of feature dimension space)
v - number of views for branch input
"""

# helper functions

def exists(v):
    return v is not None

def divisible_by(num, den):
    return (num % den) == 0

def default(v, d):
    return v if exists(v) else d

def identity(t):
    return t

def add(x, y):
    return x + y

# gdsm

_GATE_INIT = 0.999   # clamp's gradient is zero exactly on its boundary

def double_center(matrix):
    return matrix - matrix.mean(-1, keepdim = True) - matrix.mean(-2, keepdim = True) + matrix.mean((-1, -2), keepdim = True)

def _top_eigvec_sym3(m):
    """Top eigenvalue and eigenvector of batched symmetric 3x3 matrices in closed form (Smith 1961 trigonometric
    eigenvalues; eigenvector from the cross product of two rows of M − λI, picking the best-conditioned pair).
    ~40 elementwise ops, no LAPACK/cuSOLVER call and no host sync.  At an exactly degenerate top eigenvalue any unit
    vector of the eigenspace is valid; the cross products vanish there and a fixed fallback vector is used."""
    # scale-normalise first: the trigonometric form loses fp32 precision when the entries are tiny (p³ underflows),
    # and the eigenvector is scale-invariant anyway (the eigenvalue is only used through the vector here)
    scale = m.flatten(-2).abs().amax(dim = -1).clamp_min(1e-30)[..., None, None]
    m = m / scale
    a00, a01, a02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    a11, a12, a22 = m[..., 1, 1], m[..., 1, 2], m[..., 2, 2]
    q = (a00 + a11 + a22) / 3
    b00, b11, b22 = a00 - q, a11 - q, a22 - q
    p2 = (b00 * b00 + b11 * b11 + b22 * b22 + 2 * (a01 * a01 + a02 * a02 + a12 * a12)) / 6
    p = torch.sqrt(p2.clamp_min(1e-30))
    # det(B)/2 with B = (M − qI)/p
    det_b = (b00 * (b11 * b22 - a12 * a12) - a01 * (a01 * b22 - a12 * a02) + a02 * (a01 * a12 - b11 * a02)) / (p * p * p)
    phi = torch.acos((det_b / 2).clamp(-1.0, 1.0)) / 3
    lam = q + 2 * p * torch.cos(phi)                       # largest eigenvalue (of the scaled matrix)
    # rows of M − λI
    r0 = torch.stack([a00 - lam, a01, a02], dim = -1)
    r1 = torch.stack([a01, a11 - lam, a12], dim = -1)
    r2 = torch.stack([a02, a12, a22 - lam], dim = -1)
    c01, c02, c12 = torch.cross(r0, r1, dim = -1), torch.cross(r0, r2, dim = -1), torch.cross(r1, r2, dim = -1)
    n01, n02, n12 = c01.norm(dim = -1), c02.norm(dim = -1), c12.norm(dim = -1)
    best = torch.where((n01 >= n02)[..., None] & (n01 >= n12)[..., None], c01, torch.where((n02 >= n12)[..., None], c02, c12))
    norm = best.norm(dim = -1, keepdim = True)
    fallback = torch.tensor([1.0, 0.0, 0.0], dtype = m.dtype, device = m.device).expand_as(best)
    vec = torch.where(norm > 1e-12, best / norm.clamp_min(1e-30), fallback)
    return lam * scale[..., 0, 0], vec


_ONES_COMPLEMENT_BASIS = {}


def _ones_complement_basis(n, dtype, device):
    """Orthonormal basis Q (n × n−1) of the complement of the all-ones vector (cached per dtype/device)."""
    key = (n, dtype, device)
    if key not in _ONES_COMPLEMENT_BASIS:
        eye = torch.eye(n, dtype = torch.float64)
        ones = torch.ones(n, 1, dtype = torch.float64) / n ** 0.5
        q, _ = torch.linalg.qr(torch.cat([ones, eye[:, : n - 1]], dim = 1))
        _ONES_COMPLEMENT_BASIS[key] = q[:, 1:].to(dtype = dtype, device = device)
    return _ONES_COMPLEMENT_BASIS[key]


import os
_SIGMA_IMPL = os.environ.get("GDSM_SIGMA", "closed")   # "closed" (sync-free closed form, fuses under torch.compile) | "eigh"


def spectral_norm_exact(matrix, use_eigh = None):
    """sigma_max of a double-centred matrix with the exact gradient u v^T; singular vectors detached.
    The centred matrix annihilates 1, so A^T A restricted to the complement of 1 is (n−1)×(n−1); for n = 4 that
    is a 3×3 symmetric eigenproblem solved in closed form (no cuSOLVER call: `torch.linalg.eigh` on CUDA/MPS
    synchronises with the host to check convergence, 2 syncs per fold per step).  `use_eigh` keeps the old path.
    Values agree to 1e-6; gradients agree except at an exactly degenerate top pair (both pick an arbitrary but valid
    subgradient).  Eager on MPS the Jacobi path is ~4x slower than eigh; the win is on CUDA under compile."""
    n = matrix.shape[-1]
    if use_eigh is None:
        use_eigh = _SIGMA_IMPL == "eigh"
    with torch.no_grad():
        if use_eigh or n != 4:
            _, vecs = torch.linalg.eigh(matrix.transpose(-1, -2) @ matrix)
            right = vecs[..., :, -1]
        else:
            q = _ones_complement_basis(n, matrix.dtype, matrix.device)
            gram = matrix.transpose(-1, -2) @ matrix
            small = q.transpose(-1, -2) @ gram @ q
            _, right_small = _top_eigvec_sym3(small)
            right = torch.einsum('ij,...j->...i', q, right_small)
        left = torch.einsum('...ij,...j->...i', matrix, right)
        left = left / left.norm(dim = -1, keepdim = True).clamp_min(1e-12)
    return torch.einsum('...i,...ij,...j->...', left, matrix, right)

def minor_singular_ratios(centered):
    """For double-centred (..., 4, 4) logits A': the two minor singular values of A' divided by σ_max(A'), i.e. the
    singular values of the realised deviation B − J/n on 1⊥ other than the top one (which is 1).  Closed form
    (trigonometric eigenvalues of the scale-normalised 3×3 Gram on the complement of 1); no solver call."""
    n = centered.shape[-1]
    q = _ones_complement_basis(n, centered.dtype, centered.device)
    gram = centered.transpose(-1, -2) @ centered
    m = q.transpose(-1, -2) @ gram @ q
    scale = m.flatten(-2).abs().amax(dim = -1).clamp_min(1e-30)[..., None, None]
    m = m / scale
    a00, a01, a02, a11, a12, a22 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2], m[..., 1, 1], m[..., 1, 2], m[..., 2, 2]
    tr3 = (a00 + a11 + a22) / 3
    b00, b11, b22 = a00 - tr3, a11 - tr3, a22 - tr3
    p2 = (b00 * b00 + b11 * b11 + b22 * b22 + 2 * (a01 * a01 + a02 * a02 + a12 * a12)) / 6
    pp = torch.sqrt(p2.clamp_min(1e-30))
    det_b = (b00 * (b11 * b22 - a12 * a12) - a01 * (a01 * b22 - a12 * a02) + a02 * (a01 * a12 - b11 * a02)) / (pp * pp * pp)
    phi = torch.acos((det_b / 2).clamp(-1.0, 1.0)) / 3
    two_pi_3 = 2.0943951023931953
    lam = torch.stack([tr3 + 2 * pp * torch.cos(phi), tr3 + 2 * pp * torch.cos(phi + two_pi_3), tr3 + 2 * pp * torch.cos(phi + 2 * two_pi_3)], dim = -1).clamp_min(0.0)
    sv = torch.sqrt(lam)
    top = sv.amax(dim = -1, keepdim = True).clamp_min(1e-12)
    ratios = sv / top
    # drop the top one (ratio 1): keep the two smallest
    return ratios.sort(dim = -1).values[..., :2]


def gdsm(logits, gate):
    """logits: (..., n, n) residual logits; gate: (...) raw gate logit.  Returns (..., n, n).

    Computed in fp32 with autocast off: eigh has no bf16 kernel, and under autocast the Gram
    matmul would silently run in bf16."""
    n = logits.shape[-1]
    with torch.autocast(device_type = logits.device.type, enabled = False):
        centered = double_center(logits.float())
        direction = centered / spectral_norm_exact(centered).clamp_min(1e-6)[..., None, None]
        out = 1.0 / n + gate.float().clamp(0.0, 1.0)[..., None, None] * direction
    return out.to(logits.dtype)

# main functions

def get_expand_reduce_stream_functions(
    num_streams,
    add_stream_embed = False,
    dim = None,
    disable = False
):
    if num_streams == 1 or disable:
        return (nn.Identity(), nn.Identity())

    if add_stream_embed:
        assert exists(dim), '`dim` must be passed into get_init_and_expand_reduce_stream_functions for returning an expansion function with stream embeddings added'

        expand_fn = StreamEmbed(num_streams, dim, expand_to_streams = True)
    else:
        expand_fn = Reduce(pattern = 'b ... -> (b s) ...', reduction = 'repeat', s = num_streams)

    reduce_fn = Reduce(pattern = '(b s) ... -> b ...', reduction = 'sum', s = num_streams)

    return expand_fn, reduce_fn

def get_init_and_expand_reduce_stream_functions(
    num_streams,
    num_fracs = 1,
    dim = None,
    add_stream_embed = False,
    disable = None,
    **kwargs
):
    disable = default(disable, num_streams == 1 and num_fracs == 1)

    hyper_conn_klass = GDSMHyperConnections if not disable else Residual

    init_hyper_conn_fn = partial(hyper_conn_klass, num_streams, num_fracs = num_fracs, **kwargs)
    expand_reduce_fns = get_expand_reduce_stream_functions(num_streams, add_stream_embed = add_stream_embed, dim = dim, disable = disable)

    if exists(dim):
        init_hyper_conn_fn = partial(init_hyper_conn_fn, dim = dim)

    return (init_hyper_conn_fn, *expand_reduce_fns)

# norms

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * (self.gamma + 1)

# main classes

# residual base class

class Residual(Module):
    def __init__(
        self,
        *args,
        branch: Module | None = None,
        residual_transform: Module | None = None,
        **kwargs
    ):
        super().__init__()
        self.branch = branch
        self.residual_transform = default(residual_transform, nn.Identity())

    def width_connection(
        self,
        residuals
    ):
        return residuals, residuals, dict()

    def depth_connection(
        self,
        branch_output,
        residuals,

    ):
        return branch_output + self.residual_transform(residuals)

    def decorate_branch(
        self,
        branch: Callable
    ):
        assert not exists(self.branch), 'branch was already wrapped on init'

        def forward_and_add_residual(residual, *args, **kwargs):
            branch_input, add_residual = self.forward(residual)

            branch_output = branch(branch_input, *args, **kwargs)

            residual = add_residual(branch_output)

            return residual

        return forward_and_add_residual

    def forward(
        self,
        residuals,
        *branch_args,
        **branch_kwargs
    ):

        branch_input, residuals, residual_kwargs = self.width_connection(residuals)

        def add_residual_fn(branch_out):
            (branch_out, *rest), tree_spec = tree_flatten(branch_out)

            branch_out = self.depth_connection(branch_out, residuals, **residual_kwargs)

            return tree_unflatten((branch_out, *rest), tree_spec)

        if not exists(self.branch):
            return branch_input, add_residual_fn

        branch_output = self.branch(branch_input, *branch_args, **branch_kwargs)

        return add_residual_fn(branch_output)

# hyper connection residual streams

class GDSMHyperConnections(Module):
    def __init__(
        self,
        num_residual_streams,
        *,
        dim,
        logit_grad_scale = 1.0,
        iso_weight = 0.0,
        branch: Module | None = None,
        layer_index = None,
        channel_first = False,
        dropout = 0.,
        residual_transform: Module | None = None, # to support resnet blocks where dimension in not equal to dimension out - usually a residual conv
        add_branch_out_to_residual = True,  # will disable depth connections (weighted residual sum with beta) if set False
        num_input_views = 1,                # allow for the branch module to receive multiple input views, dimension placed on the very left (before batch)
        depth_residual_fn = add,
        num_fracs = 1,                      # https://arxiv.org/abs/2503.14125
    ):
        """
        mHC layout (Appendix J, Algorithm2 in https://arxiv.org/abs/2409.19606) with the
        residual matrix on the GDSM set instead of the Birkhoff polytope.
        """
        super().__init__()

        self.branch = branch

        # frac-connections paper - num_fracs > 1 will be the `m` in their paper https://arxiv.org/abs/2503.14125

        assert num_fracs >= 1

        self.num_fracs = num_fracs
        self.has_fracs = num_fracs > 1

        self.split_fracs = Rearrange('b ... (f d) -> b ... f d', f = num_fracs)
        self.merge_fracs = Rearrange('b ... f d -> b ... (f d)')

        assert divisible_by(dim, num_fracs), f'feature dimension ({dim}) must be divisible by the `num_fracs` ({num_fracs})'

        dim //= num_fracs # effective dim handled in dimension is feature dimension divided by num fractions

        # they used layernorm in paper, but rmsnorm is fine given what we know now

        assert num_residual_streams > 0, '`num_residual_streams` must be greater than 0'

        self.num_residual_streams = num_residual_streams
        init_residual_index = default(layer_index, randrange(num_residual_streams)) % num_residual_streams # just choose one random residual stream if layer index not given

        # handle the parameter dimensions, which may require (num_residuals x num_fractions) - generalizing hyper + frac connections

        num_residual_streams_fracs = num_residual_streams * num_fracs
        num_input_views_fracs = num_input_views * num_fracs

        self.num_fracs = num_fracs

        # width num residual streams

        assert num_input_views >= 1
        self.num_input_views = num_input_views

        # width connection

        self.norm = RMSNorm(dim * num_residual_streams_fracs)

        init_alpha0 = torch.ones((num_residual_streams_fracs, num_input_views_fracs)) * -1
        init_alpha0[init_residual_index, :] = 1.
        init_alpha1 = torch.eye(num_residual_streams_fracs)   # GDSM: logits ARE the matrix; identity init
        self.static_alpha = nn.Parameter(cat((init_alpha0, init_alpha1), dim = 1))

        # ------
        # XXX we modified the original implementation here 
        # H_res is from nC to n^2, instead of from (n, C) to (n, n) 
        # ------
        self.dynamic_alpha_fn = nn.Parameter(
            torch.zeros(
                dim * num_residual_streams, 
                num_fracs * ( num_residual_streams * num_residual_streams + num_residual_streams * num_input_views )
            ) 
        )

        # GDSM magnitude gate: one logit per token (per frac), its own coordinate
        self.dynamic_gate_fn = nn.Parameter(torch.zeros(dim * num_residual_streams, num_fracs))
        self.static_gate = nn.Parameter(torch.full((num_fracs,), _GATE_INIT))
        self.gate_scale = nn.Parameter(torch.ones(1) * 1e-2)
        self.logit_grad_scale = logit_grad_scale
        self.iso_weight = iso_weight
        self.iso_penalty = None
        self.pre_branch_scale = nn.Parameter(torch.ones(1) * 1e-2)
        self.residual_scale = nn.Parameter(torch.ones(1) * 1e-2)

        # depth connection related (beta)

        self.add_branch_out_to_residual = add_branch_out_to_residual

        if add_branch_out_to_residual:
            beta_init = torch.ones(num_residual_streams_fracs) * -1.
            beta_init[init_residual_index] = 1.
            self.static_beta = nn.Parameter(beta_init)

            # ------ 
            # XXX
            # same here
            # ------
            dynamic_beta_shape = (
                dim * num_residual_streams,
                num_fracs * num_residual_streams
            ) # preserve backwards compat
            self.dynamic_beta_fn = nn.Parameter(torch.zeros(dynamic_beta_shape))

            self.h_post_scale = nn.Parameter(torch.ones(()) * 1e-2)



        # dropouts

        self.dropout = nn.Dropout(dropout)

        # channel first option

        self.channel_first = channel_first

        # maybe residual transform

        self.residual_transform = default(residual_transform, nn.Identity())

        # maybe custom depth connection residual function
        # this is to prepare for gating the addition of the branch outputs to the residual streams
        # needed for memory lanes a la RMT / LMM

        self.depth_residual_fn = depth_residual_fn

    def width_connection(
        self,
        residuals
    ):
        streams = self.num_residual_streams

        maybe_transformed_residuals = self.residual_transform(residuals)

        # width connection

        # handle channel first

        if self.channel_first:
            residuals = rearrange(residuals, 'b d ... -> b ... d')

        # split out fractions

        residuals = self.split_fracs(residuals)

        # split out streams

        residuals = rearrange(residuals, '(b s) ... d -> b ... s d', s = streams)
        if getattr(self, 'record_mixers', False):
            self.last_stream_norm = residuals.detach().float().norm(dim = -1)   # (..., s) per-token stream norms

        # norm
        normed = rearrange(residuals, 'b ... s d -> b ... (s d)', s = streams)
        # normed = F.normalize(normed, dim = -1)
        normed = self.norm(normed)

        # one GEMM for all dynamic logits (alpha, gate, beta) instead of three over the same (s·d)-wide input
        fused = normed @ torch.cat([self.dynamic_alpha_fn, self.dynamic_gate_fn] + ([self.dynamic_beta_fn] if self.add_branch_out_to_residual else []), dim = -1)
        n_alpha = self.dynamic_alpha_fn.shape[-1]; n_gate = self.dynamic_gate_fn.shape[-1]
        wc_weight = fused[..., :n_alpha] # ... f (s v+s)
        gate_logits = fused[..., n_alpha:n_alpha + n_gate]
        dc_weight_fused = fused[..., n_alpha + n_gate:]
        wc_weight = rearrange(wc_weight, '... (s t) -> ... s t', s = streams)

        pre_branch_scale = repeat(self.pre_branch_scale, '1 -> v', v = self.num_input_views * self.num_fracs)
        residual_scale   = repeat(self.residual_scale  , '1 -> s', s = self.num_fracs * streams)
        alpha_scale      = cat((pre_branch_scale, residual_scale))

        dynamic_alpha = wc_weight * alpha_scale

        static_alpha = rearrange(self.static_alpha, '(f s) t -> f s t', s = streams)

        alpha = dynamic_alpha + static_alpha

        alpha = self.split_fracs(alpha) # (batch, seq, fracs1, streams, fracs2, input + residual streams)

        # the alpha is now split and "manifold constrained" with sinkhorn and sigmoid

        # (..., 1, s, 1, v) / (..., 1, s, 1, s)
        alpha_pre, alpha_residual = alpha[..., :self.num_input_views], alpha[..., self.num_input_views:]

        alpha_pre = alpha_pre.sigmoid() 
        if getattr(self, 'record_mixers', False):
            self.last_pre = alpha_pre.detach()[..., :, 0, 0]   # (..., s) read weight per stream (view 0)

        alpha_residual = rearrange(alpha_residual, '... f s g t -> ... f g s t')
        if self.logit_grad_scale != 1.0:
            # damp the gradient reaching the residual logits (forward unchanged): the exact normalisation has a
            # unit-gain Jacobian everywhere, unlike the saturated softmax / contractive Sinkhorn of the Birkhoff
            # variants (measured 0.7-0.9 vs 0.00-0.03), which at depth floods the gain scalars (2026-10-20)
            k = self.logit_grad_scale
            alpha_residual = alpha_residual * k + (alpha_residual * (1.0 - k)).detach()
        # normed is (..., f, s·d); one gate logit per frac from its own slice → (..., f)
        gate = gate_logits.diagonal(dim1 = -2, dim2 = -1) * self.gate_scale + self.static_gate
        gate = gate[..., :, None]                                                     # ... f g(=1 broadcast)
        if self.iso_weight > 0:
            # isometry prior: push the deviation's minor singular values toward 1 (an orthogonal map on 1⊥, like
            # the permutations mHC-lite freezes into at depth), keeping the mixer signed and exact
            with torch.autocast(device_type = alpha_residual.device.type, enabled = False):
                ratios = minor_singular_ratios(double_center(alpha_residual.float()))
            self.iso_penalty = (1.0 - ratios).pow(2).sum(-1).mean()
        if getattr(self, 'record_mixers', False):
            with torch.no_grad():   # raw scale of the centred logits before normalisation: the 1/σ in the Jacobian
                self.last_raw_sigma = spectral_norm_exact(double_center(alpha_residual.float()))[..., 0, 0]
                self.last_raw_gate = gate.detach().float()[..., 0, 0]
                # full singular spectrum of A' per token: the gap σ₁ − σ₂ decides whether the chosen direction is stable
                self.last_raw_svals = torch.linalg.svdvals(double_center(alpha_residual.float())[..., 0, 0, :, :])
            if alpha_residual.requires_grad:
                alpha_residual.retain_grad(); self.last_logits = alpha_residual   # per-token gradient w.r.t. the residual logits
        alpha_residual = gdsm(alpha_residual, gate.expand(*alpha_residual.shape[:-2]))
        alpha_residual = rearrange(alpha_residual, '... f g s t -> ... f s g t')
        if getattr(self, 'record_mixers', False):
            self.last_mixer = alpha_residual.detach()[..., 0, :, 0, :]   # (..., s, t) per token

        alpha = cat((alpha_pre, alpha_residual), dim = -1) # (..., f, s, f, s+v)

        # beta for weights from branch output back to residual streams

        beta = None
        if self.add_branch_out_to_residual:
            dc_weight = dc_weight_fused # ... (s f)
            dc_weight = rearrange(dc_weight, '... (s f) -> ... s f', s = streams)

            dynamic_beta = dc_weight * self.h_post_scale

            static_beta = rearrange(self.static_beta, '... (s f) -> ... s f', s = streams)

            beta = dynamic_beta + static_beta
            beta = beta.sigmoid() * 2 # sigmoid * 2 for "H_post"
            if getattr(self, 'record_mixers', False):
                self.last_post = beta.detach()[..., :, 0]   # (..., s) write weight per stream

        mix_h = einsum(alpha, residuals, '... f1 s f2 t, ... f1 s d -> ... f2 t d')

        if self.num_input_views == 1:
            branch_input, residuals = mix_h[..., 0, :], mix_h[..., 1:, :]
        else:
            branch_input, residuals = mix_h[..., :self.num_input_views, :], mix_h[..., self.num_input_views:, :]
            branch_input = rearrange(branch_input, 'b ... v d -> v b ... d')

        if self.channel_first:
            branch_input = rearrange(branch_input, 'b ... d -> b d ...')

        # maybe merge fractions back

        branch_input = self.merge_fracs(branch_input)

        residuals = rearrange(residuals, 'b ... s d -> (b s) ... d')
        if self.channel_first:
            residuals = rearrange(residuals, 'b ... d -> b d ...')
        residuals = self.merge_fracs(residuals)
        return branch_input, residuals, dict(beta = beta)

    def depth_connection(
        self,
        branch_output,
        residuals,
        *,
        beta
    ):
        assert self.add_branch_out_to_residual

        # maybe split fractions

        branch_output = self.split_fracs(branch_output)

        # 'depth' connection

        if self.channel_first:
            branch_output = rearrange(branch_output, 'b d ... -> b ... d')

        output = einsum(branch_output, beta, 'b ... f1 d, b ... f1 s f2 -> b ... f2 s d')

        output = rearrange(output, 'b ... s d -> (b s) ... d')

        # merge merge back fractions

        output = self.merge_fracs(output)

        # channel first

        if self.channel_first:
            output = rearrange(output, 'b ... d -> b d ...')

        residuals = self.depth_residual_fn(output, residuals)

        return self.dropout(residuals)

    def decorate_branch(
        self,
        branch: Callable
    ):
        assert not exists(self.branch), 'branch was already wrapped on init'

        def forward_and_add_residual(residual, *args, **kwargs):
            branch_input, add_residual = self.forward(residual)

            branch_output = branch(branch_input, *args, **kwargs)

            residual = add_residual(branch_output)

            return residual

        return forward_and_add_residual

    def forward(
        self,
        residuals,
        *branch_args,
        **branch_kwargs
    ):

        branch_input, residuals, residual_kwargs = self.width_connection(residuals)

        def add_residual_fn(branch_out):

            if not self.add_branch_out_to_residual:
                return branch_out

            (branch_out, *rest), tree_spec = tree_flatten(branch_out)

            branch_out = self.depth_connection(branch_out, residuals, **residual_kwargs)

            return tree_unflatten((branch_out, *rest), tree_spec)

        if not exists(self.branch):
            return branch_input, add_residual_fn

        branch_output = self.branch(branch_input, *branch_args, **branch_kwargs)

        return add_residual_fn(branch_output)

GDSMHyperConnections.get_expand_reduce_stream_functions = staticmethod(get_expand_reduce_stream_functions)
GDSMHyperConnections.get_init_and_expand_reduce_stream_functions = staticmethod(get_init_and_expand_reduce_stream_functions)

# stream embed

class StreamEmbed(Module):
    def __init__(
        self,
        num_streams,
        dim,
        channel_first = False,
        expand_to_streams = False
    ):
        super().__init__()
        self.channel_first = channel_first
        self.num_streams = num_streams

        self.expand_to_streams = expand_to_streams
        self.stream_embed = nn.Parameter(torch.zeros(num_streams, dim))

    def forward(self, residuals):

        if self.expand_to_streams:
            residuals = repeat(residuals, 'b ... -> (b s) ...', s = self.num_streams)

        if self.channel_first:
            residuals = rearrange(residuals, '(b s) d ... -> b ... s d', s = self.num_streams)
        else:
            residuals = rearrange(residuals, '(b s) ... d -> b ... s d', s = self.num_streams)

        residuals = residuals + self.stream_embed

        if self.channel_first:
            residuals = rearrange(residuals, 'b ... s d -> (b s) d ...', s = self.num_streams)
        else:
            residuals = rearrange(residuals, 'b ... s d -> (b s) ... d', s = self.num_streams)

        return residuals

# attention pool - taken from Enformer https://www.nature.com/articles/s41592-021-01252-x , in turn taken from somewhere else

class AttentionPoolReduceStream(Module):
    def __init__(
        self,
        num_streams,
        dim,
        channel_first = False
    ):
        super().__init__()
        self.num_streams = num_streams
        self.channel_first = channel_first

        self.to_attn_logits = nn.Linear(dim, dim, bias = False)
        self.to_attn_logits.weight.data.copy_(torch.eye(dim))

    def forward(self, residuals):

        if self.channel_first:
            residuals = rearrange(residuals, '(b s) d ... -> b ... s d', s = self.num_streams)
        else:
            residuals = rearrange(residuals, '(b s) ... d -> b ... s d', s = self.num_streams)

        attn_logits = self.to_attn_logits(residuals)
        attn = attn_logits.softmax(dim = -2)

        residuals = reduce(residuals * attn, 'b ... s d -> b ... d', 'sum')

        if self.channel_first:
            residuals = rearrange(residuals, 'b ... d -> b d ...')

        return residuals
