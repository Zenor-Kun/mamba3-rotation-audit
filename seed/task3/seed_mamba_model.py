"""
SEED Stage 1 model: complex-STFT front-end -> channel-folded MIMO Mamba-3 ->
mean pool -> 3-class head. V1_full = stock mamba_ssm.Mamba3 (full theta
rotation); no theta-switch variants here.

Front-end keeps both structural axes intact (no giant flatten FC):
  raw segment (batch, 62, 800) -> per channel torch.stft (n_fft=128, hop=6,
  hann, center=False) -> complex (batch, 62, 65, 113) -> keep the 26 in-band
  bins -> stack (real, imag) -> (batch, 62, 113, 52). Channels fold into the
  batch dim; a single small nn.Linear(52 -> d_model) is shared across every
  (channel, frame). Frames (113) are the Mamba sequence axis.

band_mask (optional): zero the gamma freq bins (real+imag) BEFORE the shared
projection -> the "drop gamma" condition, identical architecture.
"""
import numpy as np
import torch
import torch.nn as nn
from mamba_ssm import Mamba3

import seed_mamba_config as cfg


class STFTFrontend(nn.Module):
    def __init__(self, drop_gamma=False):
        super().__init__()
        self.register_buffer("kept_bins", torch.tensor(cfg.KEPT_BIN_IDX, dtype=torch.long))
        # per-kept-bin keep mask (1.0 keep, 0.0 zero) for the drop-gamma condition
        keep = np.ones(cfg.N_FREQ_BINS, dtype=np.float32)
        if drop_gamma:
            keep[cfg.GAMMA_KEPT_IDX] = 0.0
        self.register_buffer("bin_keep", torch.tensor(keep))  # (N_FREQ_BINS,) float32
        self.drop_gamma = drop_gamma

    def forward(self, x):
        # x: (batch, 62, 800). STFT computed in fp32 (autocast disabled -- complex
        # STFT is not a bf16 op); returns fp32 features, cast to bf16 later by autocast.
        b, c, t = x.shape
        with torch.autocast(device_type="cuda", enabled=False):
            xr = x.reshape(b * c, t).float()
            window = torch.hann_window(cfg.STFT_N_FFT, device=x.device)
            spec = torch.stft(xr, n_fft=cfg.STFT_N_FFT, hop_length=cfg.STFT_HOP,
                              win_length=cfg.STFT_N_FFT, window=window,
                              center=False, return_complex=True)      # (b*c, 65, 113)
            spec = spec[:, self.kept_bins, :cfg.N_FRAMES]             # (b*c, 26, 112) drop last frame
            keep = self.bin_keep.float()[None, :, None]
            re = spec.real * keep
            im = spec.imag * keep
            feat = torch.cat([re, im], dim=1).transpose(1, 2).contiguous()  # (b*c, frames, 52)
        return feat.reshape(b, c, cfg.N_FRAMES, cfg.N_FREQ_X2)       # (b, 62, frames, 52) fp32


class SeedMambaBlock(nn.Module):
    def __init__(self, d_model, layer_idx, dropout=0.0):
        super().__init__()
        self.norm = nn.RMSNorm(d_model)
        self.mamba = Mamba3(
            d_model=d_model, d_state=cfg.D_STATE, expand=cfg.EXPAND, headdim=cfg.HEADDIM,
            chunk_size=cfg.CHUNK_SIZE, is_mimo=cfg.IS_MIMO, mimo_rank=cfg.MIMO_RANK,
            is_outproj_norm=True, layer_idx=layer_idx,
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        seq_in = x.shape[1]
        out = x + self.drop(self.mamba(self.norm(x)))
        cfg.assert_no_stride_or_pool(seq_in, out.shape[1], stage_name=f"SeedMambaBlock[{self.mamba.layer_idx}]")
        return out


class SeedMambaClassifier(nn.Module):
    def __init__(self, drop_gamma=False, d_model=cfg.D_MODEL, n_layers=cfg.N_LAYERS, dropout=cfg.DROPOUT):
        super().__init__()
        self.frontend = STFTFrontend(drop_gamma=drop_gamma)
        self.input_proj = nn.Linear(cfg.N_FREQ_X2, d_model)   # 52 -> d_model, shared across channels AND frames
        self.blocks = nn.ModuleList([SeedMambaBlock(d_model, layer_idx=i, dropout=dropout) for i in range(n_layers)])
        self.final_norm = nn.RMSNorm(d_model)
        self.head_drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, cfg.N_CLASSES)

    def forward(self, x):
        # x: (batch, 62, 800) raw (already z-scored per channel by the caller)
        feat = self.frontend(x)                    # (b, 62, frames, 52)
        b, c, tf, f = feat.shape
        h = feat.reshape(b * c, tf, f)
        h = self.input_proj(h)                     # (b*c, frames, d_model)
        seq_in = h.shape[1]
        for blk in self.blocks:
            h = blk(h)
        cfg.assert_no_stride_or_pool(seq_in, h.shape[1], stage_name="seed_mamba_full_stack")
        h = self.final_norm(h)
        pooled_frames = h.mean(dim=1)                       # (b*c, d_model)
        pooled = pooled_frames.reshape(b, c, -1).mean(dim=1)  # (b, d_model) mean over channels
        return self.head(self.head_drop(pooled))            # (b, 3)


def trainable_param_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(drop_gamma=False):
    return SeedMambaClassifier(drop_gamma=drop_gamma)
