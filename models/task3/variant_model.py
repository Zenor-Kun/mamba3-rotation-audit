"""
Task 3 / Probe 1: the four theta (rotation-angle) variants.

Surgical rule (do not violate): only the `angles` tensor computed inside
Mamba3.forward is touched. Decay magnitude |lambda| (ADT/DT), B, C, the
trapezoidal discretization (trap), and the selective (data-dependent dt/A)
mechanism are untouched byte-for-byte -- copied straight from the upstream
Mamba3.forward SISO branch. If this file's V1_full path doesn't reproduce
upstream Mamba3 output exactly, the whole ablation is invalid; that's exactly
what consistency_check.py verifies before any training happens.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from mamba_ssm import Mamba3, Mamba2
from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined
from mamba_ssm.modules.mamba3 import heavy_tail_activation

import probe1_config as pcfg

VARIANT_NAMES = pcfg.VARIANTS


class VariantMamba3(Mamba3):
    """Same module as upstream Mamba3 (SISO only), with the rotation angle
    (theta) computation swapped per `variant`. Everything else is an
    unmodified copy of Mamba3.forward's SISO branch."""

    def __init__(self, *args, variant="V1_full", **kwargs):
        assert not kwargs.get("is_mimo", False), "Probe 1 only uses the SISO path (matches Task 2 V1)"
        super().__init__(*args, **kwargs)
        assert variant in VARIANT_NAMES, f"unknown variant {variant}"
        self.variant = variant
        if variant == "V3_theta_frozen_random":
            # one frozen random constant per (head, rotary-angle-slot) -- "every
            # channel gets its own constant", never touched by the optimizer and
            # never a function of the input.
            g = torch.Generator().manual_seed(1234567)  # fixed, independent of training seed
            theta = (torch.rand(self.nheads, self.num_rope_angles, generator=g) * 2 * math.pi) - math.pi
            self.register_buffer("frozen_angle", theta, persistent=True)

    def _theta_switch(self, angles_raw, batch, seqlen):
        """angles_raw: (B, L, num_rope_angles) straight off the in_proj split,
        BEFORE the upstream unsqueeze/expand/cast. Returns the (B, L, nheads,
        num_rope_angles) tensor that upstream would feed to the kernel."""
        if self.variant == "V1_full":
            return angles_raw.unsqueeze(-2).expand(-1, -1, self.nheads, -1).to(torch.float32)
        elif self.variant == "V2_theta_zero":
            out = torch.zeros(batch, seqlen, self.nheads, self.num_rope_angles,
                               device=angles_raw.device, dtype=torch.float32)
            return out
        elif self.variant == "V3_theta_frozen_random":
            # deliberately never reads angles_raw -- input independence is structural,
            # not just "gradient happens to be zero"
            return self.frozen_angle.to(torch.float32).unsqueeze(0).unsqueeze(0).expand(batch, seqlen, -1, -1)
        else:
            raise AssertionError(self.variant)

    def forward(self, u, seq_idx=None, cu_seqlens=None, inference_params=None):
        assert inference_params is None, "Probe 1 training/eval never uses incremental decoding"
        batch, seqlen, dim = u.shape

        zxBCdtAtrap = self.in_proj(u)
        z, x, B, C, dd_dt, dd_A, trap, angles_raw = torch.split(
            zxBCdtAtrap,
            [
                self.d_inner, self.d_inner,
                self.d_state * self.num_bc_heads * self.mimo_rank,
                self.d_state * self.num_bc_heads * self.mimo_rank,
                self.nheads, self.nheads, self.nheads,
                self.num_rope_angles,
            ],
            dim=-1)
        z = rearrange(z, "b l (h p) -> b l h p", p=self.headdim)
        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
        B = rearrange(B, "b l (r g n) -> b l r g n", r=self.mimo_rank, g=self.num_bc_heads)
        C = rearrange(C, "b l (r g n) -> b l r g n", r=self.mimo_rank, g=self.num_bc_heads)
        trap = rearrange(trap, "b l h -> b h l")

        _A = -heavy_tail_activation(dd_A.to(torch.float32))
        _A = torch.clamp(_A, max=-self.A_floor)
        DT = F.softplus(dd_dt + self.dt_bias)
        ADT = _A * DT
        DT = rearrange(DT, "b l n -> b n l")
        ADT = rearrange(ADT, "b l n -> b n l")

        # <<< the only surgical change lives here >>>
        angles = self._theta_switch(angles_raw, batch, seqlen)

        B = self.B_norm(B)
        C = self.C_norm(C)

        y = mamba3_siso_combined(
            Q=C.squeeze(2),
            K=B.squeeze(2),
            V=x,
            ADT=ADT,
            DT=DT,
            Trap=trap,
            Q_bias=self.C_bias.squeeze(1),
            K_bias=self.B_bias.squeeze(1),
            Angles=angles,
            D=self.D,
            Z=z if not self.is_outproj_norm else None,
            chunk_size=self.chunk_size,
            Input_States=None,
            return_final_states=False,
            cu_seqlens=cu_seqlens,
        )
        y = rearrange(y, "b l h p -> b l (h p)")
        if self.is_outproj_norm:
            z = rearrange(z, "b l h p -> b l (h p)")
            y = self.norm(y, z)

        out = self.out_proj(y.to(x.dtype))
        return out


def masked_mean_pool(x, mask=None):
    if mask is None:
        return x.mean(dim=1)
    mask = mask.unsqueeze(-1).to(x.dtype)
    return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)


class VariantBlock(nn.Module):
    """Mamba3-variant residual block, identical wrapper to Task 2's Mamba3Block."""
    def __init__(self, d_model, layer_idx, variant, dropout=0.0):
        super().__init__()
        self.norm = nn.RMSNorm(d_model)
        self.mamba = VariantMamba3(
            d_model=d_model,
            d_state=pcfg.D_STATE,
            expand=pcfg.EXPAND,
            headdim=pcfg.HEADDIM,
            chunk_size=pcfg.CHUNK_SIZE,
            is_mimo=False,
            is_outproj_norm=True,
            layer_idx=layer_idx,
            variant=variant,
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        seq_len_in = x.shape[1]
        out = x + self.drop(self.mamba(self.norm(x)))
        pcfg.task2_cfg.assert_no_stride_or_pool(seq_len_in, out.shape[1], stage_name=f"VariantBlock[{self.mamba.layer_idx}]")
        return out


class ProbeClassifier(nn.Module):
    """V1/V2/V3: Mamba-3-variant stack -> mask-aware mean pool -> linear head."""
    def __init__(self, variant, n_channels=None, d_model=pcfg.D_MODEL, n_layers=pcfg.N_LAYERS, dropout=pcfg.DROPOUT):
        super().__init__()
        n_channels = n_channels or pcfg.task2_cfg.N_DATA_CH
        self.input_proj = nn.Identity() if n_channels == d_model else nn.Linear(n_channels, d_model)
        self.blocks = nn.ModuleList([VariantBlock(d_model, layer_idx=i, variant=variant, dropout=dropout)
                                     for i in range(n_layers)])
        self.final_norm = nn.RMSNorm(d_model)
        self.head_drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x, mask=None):
        seq_len_in = x.shape[-1]
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        pcfg.task2_cfg.assert_no_stride_or_pool(seq_len_in, x.shape[1], stage_name="full_stack")
        x = self.final_norm(x)
        pooled = masked_mean_pool(x, mask)
        return self.head(self.head_drop(pooled)).squeeze(-1)


class Mamba2Block(nn.Module):
    def __init__(self, d_model, layer_idx, dropout=0.0):
        super().__init__()
        self.norm = nn.RMSNorm(d_model)
        self.mamba = Mamba2(
            d_model=d_model,
            d_state=pcfg.V4_D_STATE,
            expand=pcfg.V4_EXPAND,
            headdim=pcfg.V4_HEADDIM,
            chunk_size=pcfg.V4_CHUNK_SIZE,
            layer_idx=layer_idx,
            use_mem_eff_path=False,  # causal_conv1d isn't installed in this env; falls back to plain conv1d
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        seq_len_in = x.shape[1]
        out = x + self.drop(self.mamba(self.norm(x)))
        pcfg.task2_cfg.assert_no_stride_or_pool(seq_len_in, out.shape[1], stage_name=f"Mamba2Block[{self.mamba.layer_idx}]")
        return out


class V4RealOnlyClassifier(nn.Module):
    """V4: real-valued Mamba-2 stack, widened to match V1's trainable param count
    within +/-5%. No complex numbers, no rotation, anywhere."""
    def __init__(self, n_channels=None, d_model=pcfg.D_MODEL, n_layers=pcfg.N_LAYERS, dropout=pcfg.DROPOUT):
        super().__init__()
        n_channels = n_channels or pcfg.task2_cfg.N_DATA_CH
        self.input_proj = nn.Identity() if n_channels == d_model else nn.Linear(n_channels, d_model)
        self.blocks = nn.ModuleList([Mamba2Block(d_model, layer_idx=i, dropout=dropout) for i in range(n_layers)])
        self.final_norm = nn.RMSNorm(d_model)
        self.head_drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x, mask=None):
        seq_len_in = x.shape[-1]
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        pcfg.task2_cfg.assert_no_stride_or_pool(seq_len_in, x.shape[1], stage_name="full_stack_v4")
        x = self.final_norm(x)
        pooled = masked_mean_pool(x, mask)
        return self.head(self.head_drop(pooled)).squeeze(-1)


def build_classifier(variant):
    if variant == "V4_real_only":
        return V4RealOnlyClassifier()
    return ProbeClassifier(variant=variant)


def trainable_param_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
