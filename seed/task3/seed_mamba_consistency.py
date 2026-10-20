"""
Consistency / health checks for the SEED Stage 1 Mamba pipeline. Run before the
LOSO sweep. Verifies:
  1. STFT front-end output shape (b,62,113,52), seq_len>=100, kept-bin count,
     effective frame rate + theta->Hz constant.
  2. drop_gamma actually zeros the gamma freq bins and nothing else.
  3. MIMO Mamba-3 is instantiated with is_mimo=True / mimo_rank=4 and a bf16
     forward produces finite (b,3) logits.
  4. no stride/pool inside the SSM (asserts in forward don't trip).
  5. a few optimizer steps on a tiny batch drive CE loss DOWN (model can learn).
"""
import numpy as np
import torch
import torch.nn as nn

import seed_mamba_config as cfg
from seed_mamba_model import build_model, STFTFrontend, trainable_param_count

DTYPE = torch.bfloat16
dev = "cuda:0"


def check_frontend():
    print(f"SEQ_LEN(frames)={cfg.SEQ_LEN}  (>=100: {cfg.SEQ_LEN>=100})")
    print(f"kept freq bins={cfg.N_FREQ_BINS} (feature dim={cfg.N_FREQ_X2})")
    print(f"effective frame rate={cfg.EFFECTIVE_FRAME_RATE_HZ:.4f} Hz; theta->Hz f=phi*{cfg.THETA_TO_HZ_CONST:.4f}")
    from collections import Counter
    print("bins per band:", dict(Counter(cfg.KEPT_BIN_BAND)))
    x = torch.randn(3, cfg.N_CH, cfg.WIN_LEN)
    full = STFTFrontend(drop_gamma=False)(x)
    dg = STFTFrontend(drop_gamma=True)(x)
    assert full.shape == (3, cfg.N_CH, cfg.N_FRAMES, cfg.N_FREQ_X2), full.shape
    # drop_gamma must zero exactly the gamma bins (real & imag halves)
    g = cfg.GAMMA_KEPT_IDX; ng = cfg.NONGAMMA_KEPT_IDX
    re_dg, im_dg = dg[..., :cfg.N_FREQ_BINS], dg[..., cfg.N_FREQ_BINS:]
    assert torch.all(re_dg[..., g] == 0) and torch.all(im_dg[..., g] == 0), "gamma not zeroed"
    assert torch.allclose(dg[..., ng], full[..., ng]) and torch.allclose(im_dg[..., ng], full[..., cfg.N_FREQ_BINS:][..., ng]), "non-gamma changed"
    print(f"frontend shape {tuple(full.shape)} OK; drop_gamma zeros gamma bins {g.tolist()} only -> PASS")


def check_mimo_and_forward():
    m = build_model(drop_gamma=False).to(dev)
    mb = m.blocks[0].mamba
    print(f"Mamba3 is_mimo={mb.is_mimo} mimo_rank={getattr(mb,'mimo_rank',None)} d_state={mb.d_state} headdim={mb.headdim} chunk_size={cfg.CHUNK_SIZE}")
    assert mb.is_mimo is True and getattr(mb, "mimo_rank", None) == cfg.MIMO_RANK
    x = torch.randn(2, cfg.N_CH, cfg.WIN_LEN, device=dev)
    with torch.autocast(device_type="cuda", dtype=DTYPE):
        y = m(x)
    assert y.shape == (2, cfg.N_CLASSES), y.shape
    assert torch.isfinite(y).all(), "non-finite logits"
    print(f"bf16(autocast) forward -> logits {tuple(y.shape)} dtype={y.dtype} finite; params={trainable_param_count(m):,} -> PASS")


def check_learns():
    torch.manual_seed(0)
    m = build_model(drop_gamma=False).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    x = torch.randn(8, cfg.N_CH, cfg.WIN_LEN, device=dev)
    y = torch.randint(0, cfg.N_CLASSES, (8,), device=dev)
    losses = []
    for _ in range(20):
        opt.zero_grad()
        with torch.autocast(device_type="cuda", dtype=DTYPE):
            out = m(x)
        l = loss_fn(out.float(), y); l.backward(); opt.step()
        losses.append(l.item())
    print(f"overfit tiny batch: loss {losses[0]:.3f} -> {losses[-1]:.3f}")
    assert losses[-1] < losses[0] - 0.1, "loss did not decrease -> model not learning"
    print("model can drive CE loss down -> PASS")


if __name__ == "__main__":
    check_frontend()
    check_mimo_and_forward()
    check_learns()
    print("\nALL CONSISTENCY CHECKS PASSED")
