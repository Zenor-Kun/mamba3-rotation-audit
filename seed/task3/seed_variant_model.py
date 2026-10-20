"""
SEED Mamba Stage 2: the four theta-switch variants, all sharing V1's EXACT
config (front-end, d_model=128, d_state=128, headdim=32 [the hardware-forced
value], mimo_rank=4, is_mimo=True, 2 layers, bf16) -- only the rotation angle
(theta) computation differs, so the ablation is fair.

V1/V2/V3 use SeedVariantMamba3: a byte-for-byte copy of upstream Mamba3.forward
(the MIMO branch we actually run), with ONLY the single `angles = ...` line
(upstream line 202) replaced by a theta-switch. Everything else -- decay
|lambda| (ADT/DT), B/C, trapezoid, the MIMO kernel call and its fuse logic --
is copied unchanged. If V1_full doesn't reproduce stock Mamba3 exactly, the
ablation is invalid (checked in seed_variant_consistency.py).

  V1_full: theta data-dependent + learnable (identical to stock Mamba3).
  V2_theta_zero: rotation angle forced to 0 -> pure real decay (complex-channel
                 count unchanged; the state just never rotates).
  V3_theta_frozen_random: random constant angle per (head, rope-slot), NOT
                 learnable, input-INDEPENDENT (structural, not just zero-grad).
  V4_real_only: real-valued Mamba-2 stack, widened to match V1's param count
                +-5% (no complex numbers / rotation anywhere).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mamba_ssm import Mamba3, Mamba2
from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo import mamba3_mimo as mamba3_mimo_combined
from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined
from mamba_ssm.modules.mamba3 import heavy_tail_activation

import seed_mamba_config as cfg
from seed_mamba_model import STFTFrontend

VARIANTS = ["V1_full", "V2_theta_zero", "V3_theta_frozen_random", "V4_real_only"]


class SeedVariantMamba3(Mamba3):
    def __init__(self, *args, variant="V1_full", **kwargs):
        assert kwargs.get("is_mimo", False), "SEED Stage 2 runs the MIMO path (matches V1)"
        super().__init__(*args, **kwargs)
        assert variant in ("V1_full", "V2_theta_zero", "V3_theta_frozen_random")
        self.variant = variant
        if variant == "V3_theta_frozen_random":
            g = torch.Generator().manual_seed(1234567)  # fixed, independent of training seed
            theta = (torch.rand(self.nheads, self.num_rope_angles, generator=g) * 2 * math.pi) - math.pi
            self.register_buffer("frozen_angle", theta, persistent=True)

    def _theta_switch(self, angles_raw, batch, seqlen):
        if self.variant == "V1_full":
            return angles_raw.unsqueeze(-2).expand(-1, -1, self.nheads, -1).to(torch.float32)
        if self.variant == "V2_theta_zero":
            return torch.zeros(batch, seqlen, self.nheads, self.num_rope_angles,
                               device=angles_raw.device, dtype=torch.float32)
        if self.variant == "V3_theta_frozen_random":
            return self.frozen_angle.to(torch.float32)[None, None].expand(batch, seqlen, -1, -1)
        raise AssertionError(self.variant)

    def forward(self, u, seq_idx=None, cu_seqlens=None, inference_params=None):
        assert inference_params is None, "Stage 2 never uses incremental decoding"
        batch, seqlen, dim = u.shape

        zxBCdtAtrap = self.in_proj(u)
        z, x, B, C, dd_dt, dd_A, trap, angles = torch.split(
            zxBCdtAtrap,
            [self.d_inner, self.d_inner,
             self.d_state * self.num_bc_heads * self.mimo_rank,
             self.d_state * self.num_bc_heads * self.mimo_rank,
             self.nheads, self.nheads, self.nheads, self.num_rope_angles],
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

        # <<< the ONLY surgical change: theta (rotation angle) >>>
        angles = self._theta_switch(angles, batch, seqlen)

        B = self.B_norm(B)
        C = self.C_norm(C)

        y = mamba3_mimo_combined(
            Q=C, K=B, V=x, ADT=ADT, DT=DT, Trap=trap,
            Q_bias=self.C_bias, K_bias=self.B_bias,
            MIMO_V=self.mimo_x, MIMO_Z=self.mimo_z,
            MIMO_Out=self.mimo_o if (self.fuse_pregate_headwise_norm or not self.is_outproj_norm) else None,
            Angles=angles, D=self.D,
            Z=z if (self.fuse_pregate_headwise_norm or not self.is_outproj_norm) else None,
            chunk_size=self.chunk_size, rotary_dim_divisor=self.rotary_dim_divisor,
            dtype=x.dtype, return_state=False, cu_seqlens=cu_seqlens,
            fuse_pregate_headwise_rms_norm=self.fuse_pregate_headwise_norm,
            outproj_norm_weight=self.norm.weight if self.fuse_pregate_headwise_norm else None,
            outproj_norm_eps=self.norm.eps if self.fuse_pregate_headwise_norm else 1e-5,
        )
        if self.is_outproj_norm and not self.fuse_pregate_headwise_norm:
            z = torch.einsum("blhp,hrp->blrhp", z.float(), self.mimo_z)
            z = rearrange(z, "b l r h p -> b l r (h p)")
            y = rearrange(y, "b l r h p -> b l r (h p)").float()
            y = self.norm(y, z)
            y = rearrange(y, "b l r (h p) -> b l r h p", p=self.headdim)
            y = torch.einsum("blrhp,hrp->blhp", y, self.mimo_o)
        y = rearrange(y, "b l h p -> b l (h p)")
        return self.out_proj(y.to(x.dtype))


class VariantBlock(nn.Module):
    def __init__(self, d_model, layer_idx, variant, dropout=0.0):
        super().__init__()
        self.norm = nn.RMSNorm(d_model)
        self.mamba = SeedVariantMamba3(
            d_model=d_model, d_state=cfg.D_STATE, expand=cfg.EXPAND, headdim=cfg.HEADDIM,
            chunk_size=cfg.CHUNK_SIZE, is_mimo=cfg.IS_MIMO, mimo_rank=cfg.MIMO_RANK,
            is_outproj_norm=True, layer_idx=layer_idx, variant=variant)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        seq_in = x.shape[1]
        out = x + self.drop(self.mamba(self.norm(x)))
        cfg.assert_no_stride_or_pool(seq_in, out.shape[1], stage_name=f"VariantBlock[{self.mamba.layer_idx}]")
        return out


class Mamba2Block(nn.Module):
    def __init__(self, d_model, d_state, expand, headdim, layer_idx, dropout=0.0, chunk_size=64):
        super().__init__()
        self.norm = nn.RMSNorm(d_model)
        # Mamba2 (real, no MIMO tilelang kernel) has no 3090 smem limit, so it uses
        # a standard chunk_size=64 -- NOT the MIMO-forced chunk_size=8, and headdim
        # must be a power of 2 (headdim=40 triggers an illegal-memory-access in the
        # Mamba2 triton kernel).
        self.mamba = Mamba2(d_model=d_model, d_state=d_state, expand=expand, headdim=headdim,
                            chunk_size=chunk_size, layer_idx=layer_idx, use_mem_eff_path=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        seq_in = x.shape[1]
        out = x + self.drop(self.mamba(self.norm(x)))
        cfg.assert_no_stride_or_pool(seq_in, out.shape[1], stage_name=f"Mamba2Block[{self.mamba.layer_idx}]")
        return out


class SeedVariantClassifier(nn.Module):
    """V1/V2/V3 (complex, theta-switched) and V4 (real Mamba2) share the front-end,
    channel folding, pooling and head. V4 widens d_model to param-match V1."""
    def __init__(self, variant, d_model=cfg.D_MODEL, v4_d_state=cfg.D_STATE, v4_expand=cfg.EXPAND,
                 v4_headdim=64, n_layers=cfg.N_LAYERS, dropout=cfg.DROPOUT):
        super().__init__()
        self.variant = variant
        self.frontend = STFTFrontend(drop_gamma=False)
        self.input_proj = nn.Linear(cfg.N_FREQ_X2, d_model)
        if variant == "V4_real_only":
            self.blocks = nn.ModuleList([Mamba2Block(d_model, v4_d_state, v4_expand, v4_headdim, layer_idx=i,
                                                     dropout=dropout, chunk_size=V4_CHUNK_SIZE)
                                         for i in range(n_layers)])
        else:
            self.blocks = nn.ModuleList([VariantBlock(d_model, layer_idx=i, variant=variant, dropout=dropout)
                                         for i in range(n_layers)])
        self.final_norm = nn.RMSNorm(d_model)
        self.head_drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, cfg.N_CLASSES)

    def forward(self, x):
        feat = self.frontend(x)                    # (b, 62, frames, 52)
        b, c, tf, f = feat.shape
        h = self.input_proj(feat.reshape(b * c, tf, f))
        seq_in = h.shape[1]
        for blk in self.blocks:
            h = blk(h)
        cfg.assert_no_stride_or_pool(seq_in, h.shape[1], stage_name="seed_variant_full_stack")
        h = self.final_norm(h)
        pooled = h.mean(dim=1).reshape(b, c, -1).mean(dim=1)
        return self.head(self.head_drop(pooled))


def trainable_param_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# V4 width param-matches V1 (504,227) within +-5%: d_model=192, d_state=64,
# headdim=64 -> 511,079 params (1.36% off). headdim must be a power of 2 for the
# Mamba2 triton kernel (headdim=40 -> illegal memory access), which ruled out the
# tighter 0.49% match; this is the closest robust real-valued config.
V4_D_MODEL = 192
V4_D_STATE = 64
V4_EXPAND = 2
V4_HEADDIM = 64
V4_CHUNK_SIZE = 64


def build_variant(variant):
    if variant == "V4_real_only":
        return SeedVariantClassifier("V4_real_only", d_model=V4_D_MODEL, v4_d_state=V4_D_STATE,
                                     v4_expand=V4_EXPAND, v4_headdim=V4_HEADDIM)
    return SeedVariantClassifier(variant)
