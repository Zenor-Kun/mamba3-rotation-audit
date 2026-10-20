"""
Task 2 V1 Full baseline: official Mamba-3 block (SISO), d_model=128, N_LAYERS layers,
mask-aware mean pooling -> linear binary classifier. This exists only to prove the
pipeline is alive end to end -- no ablation, no phase analysis here.

Locked rule: no stride / pooling in front of the SSM. Enforced by asserting the
sequence length is unchanged across every block (config.assert_no_stride_or_pool).
"""
import torch
import torch.nn as nn
from mamba_ssm import Mamba3

import config as cfg


class Mamba3Block(nn.Module):
    def __init__(self, d_model, layer_idx, dropout=0.0):
        super().__init__()
        self.norm = nn.RMSNorm(d_model)
        self.mamba = Mamba3(
            d_model=d_model,
            d_state=cfg.D_STATE,
            expand=cfg.EXPAND,
            headdim=cfg.HEADDIM,
            chunk_size=cfg.CHUNK_SIZE,
            is_mimo=cfg.IS_MIMO,
            is_outproj_norm=True,
            layer_idx=layer_idx,
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        seq_len_in = x.shape[1]
        out = x + self.drop(self.mamba(self.norm(x)))
        cfg.assert_no_stride_or_pool(seq_len_in, out.shape[1], stage_name=f"Mamba3Block[{self.mamba.layer_idx}]")
        return out


def masked_mean_pool(x, mask=None):
    """x: (B, L, D). mask: (B, L) of 1/0, or None (all ones). Returns (B, D)."""
    if mask is None:
        return x.mean(dim=1)
    mask = mask.unsqueeze(-1).to(x.dtype)
    summed = (x * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1e-6)
    return summed / denom


class MambaV1Classifier(nn.Module):
    def __init__(self, n_channels=cfg.N_DATA_CH, d_model=cfg.D_MODEL, n_layers=cfg.N_LAYERS, dropout=cfg.DROPOUT):
        super().__init__()
        self.input_proj = nn.Identity() if n_channels == d_model else nn.Linear(n_channels, d_model)
        self.blocks = nn.ModuleList([Mamba3Block(d_model, layer_idx=i, dropout=dropout) for i in range(n_layers)])
        self.final_norm = nn.RMSNorm(d_model)
        self.head_drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x, mask=None):
        """x: (B, n_channels, L) raw window tensor -> (B,) logits."""
        seq_len_in = x.shape[-1]
        x = x.transpose(1, 2)  # (B, L, n_channels)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        cfg.assert_no_stride_or_pool(seq_len_in, x.shape[1], stage_name="full_stack")
        x = self.final_norm(x)
        pooled = masked_mean_pool(x, mask)
        return self.head(self.head_drop(pooled)).squeeze(-1)
