"""
Stage 2 consistency checks (run before training):
  1. V1_full's SeedVariantMamba3 reproduces stock mamba_ssm.Mamba3 EXACTLY
     (same weights -> same output). If not, the theta-switch ablation is invalid.
  2. V2/V3 theta-switch actually changes the angle path (outputs differ from V1)
     and each variant's forward is finite; V3's angle is input-independent.
  3. V4 is param-matched to V1 within +-5% and trains (CE loss drops).
  4. every variant: bf16 autocast forward -> finite (b,3) logits.
"""
import numpy as np
import torch
import torch.nn as nn

import seed_mamba_config as cfg
from seed_variant_model import (SeedVariantMamba3, SeedVariantClassifier, build_variant,
                                trainable_param_count, V4_D_MODEL)
from mamba_ssm import Mamba3

DTYPE = torch.bfloat16
dev = "cuda:0"


def check_v1_equals_stock():
    torch.manual_seed(0)
    kw = dict(d_model=cfg.D_MODEL, d_state=cfg.D_STATE, expand=cfg.EXPAND, headdim=cfg.HEADDIM,
              chunk_size=cfg.CHUNK_SIZE, is_mimo=True, mimo_rank=cfg.MIMO_RANK,
              is_outproj_norm=True, layer_idx=0)
    stock = Mamba3(**kw).to(dev)
    var = SeedVariantMamba3(variant="V1_full", **kw).to(dev)
    var.load_state_dict(stock.state_dict())  # identical weights
    x = torch.randn(4, cfg.SEQ_LEN, cfg.D_MODEL, device=dev)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=DTYPE):
        ys = stock(x); yv = var(x)
    diff = (ys.float() - yv.float()).abs().max().item()
    print(f"V1_full vs stock Mamba3: max_abs_diff={diff:.3e} -> {'PASS' if diff==0 else 'FAIL'}")
    assert diff == 0, "V1_full must reproduce stock Mamba3 exactly"


def check_variants_forward_and_differ():
    x = torch.randn(3, cfg.N_CH, cfg.WIN_LEN, device=dev)
    outs = {}
    params = {}
    for v in ["V1_full", "V2_theta_zero", "V3_theta_frozen_random", "V4_real_only"]:
        torch.manual_seed(0)
        m = build_variant(v).to(dev)
        params[v] = trainable_param_count(m)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=DTYPE):
            y = m(x)
        assert y.shape == (3, cfg.N_CLASSES) and torch.isfinite(y).all(), f"{v} bad output"
        outs[v] = y.float().cpu()
        print(f"{v:24s} params={params[v]:,} forward finite {tuple(y.shape)}")
    # V2/V3 must differ from V1 given identical seed init (theta path changed)
    for v in ["V2_theta_zero", "V3_theta_frozen_random"]:
        d = (outs["V1_full"] - outs[v]).abs().max().item()
        print(f"  {v} differs from V1_full by max {d:.3e} -> {'PASS' if d>0 else 'FAIL(theta not switched)'}")
        assert d > 0
    r = abs(params["V4_real_only"] - params["V1_full"]) / params["V1_full"]
    print(f"  V4 param match: {params['V4_real_only']:,} vs V1 {params['V1_full']:,} = {r*100:.2f}% (<=5%: {r<=0.05})")
    assert r <= 0.05


def check_v3_input_independent():
    m = build_variant("V3_theta_frozen_random").to(dev).eval()
    mb = m.blocks[0].mamba
    a1 = mb._theta_switch(torch.randn(2, cfg.SEQ_LEN, mb.num_rope_angles, device=dev), 2, cfg.SEQ_LEN)
    a2 = mb._theta_switch(torch.randn(2, cfg.SEQ_LEN, mb.num_rope_angles, device=dev), 2, cfg.SEQ_LEN)
    same = torch.equal(a1, a2)
    print(f"V3 angle input-independent (two random inputs -> same angle): {same} -> {'PASS' if same else 'FAIL'}")
    assert same


def check_each_learns():
    for v in ["V2_theta_zero", "V4_real_only"]:
        torch.manual_seed(0)
        m = build_variant(v).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        x = torch.randn(8, cfg.N_CH, cfg.WIN_LEN, device=dev)
        y = torch.randint(0, cfg.N_CLASSES, (8,), device=dev)
        losses = []
        for _ in range(15):
            opt.zero_grad()
            with torch.autocast(device_type="cuda", dtype=DTYPE):
                out = m(x)
            l = loss_fn(out.float(), y); l.backward(); opt.step(); losses.append(l.item())
        print(f"{v}: loss {losses[0]:.3f}->{losses[-1]:.3f} -> {'PASS' if losses[-1]<losses[0]-0.1 else 'FAIL'}")
        assert losses[-1] < losses[0] - 0.1


if __name__ == "__main__":
    check_v1_equals_stock()
    check_variants_forward_and_differ()
    check_v3_input_independent()
    check_each_learns()
    print("\nALL STAGE-2 CONSISTENCY CHECKS PASSED")
