#!/usr/bin/env python3
"""
eval/run_solo_pfc_baseline.py — Reproduce original EDGE solo PFC on our val set.

Goal: sanity-check that our PFC pipeline is unbiased — if we can get the
published EDGE solo PFC (~1.5) using their released `checkpoint.pt`, the high
duet PFC numbers are model behaviour, not measurement artefact.

Pipeline (mirrors eval/run_val_pfc.py but in SOLO mode):
  1. Load `checkpoint.pt` with duet=False (cond_feature_dim=4800).
  2. For each val follower slice, read its jukebox features (4800-d, 150 frames).
  3. Sample motion via diffusion.ddim_sample with music-only conditioning.
  4. FK to joint trajectories, save as `{full_pose: [150, 24, 3]}` pkl.
  5. Run the exact same PFC formula used everywhere else (eval/eval_pfc.py).

Reference numbers
-----------------
  EDGE paper (solo, AIST++ test):     PFC = 1.5363  (with CCL)
  Our val GT follower (real human):   PFC = 1.2607
  Our duet best (lead_only ep1700):   PFC = 2.350

Usage
-----
    python eval/run_solo_pfc_baseline.py \
        --checkpoint checkpoint.pt \
        --out_dir    eval/solo_pfc_baseline
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.quaternion import ax_from_6v
from eval.run_val_pfc import calc_pfc, slice_index


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint",      default="checkpoint.pt",
                   help="EDGE pretrained solo checkpoint")
    p.add_argument("--data_dir",        default="data")
    p.add_argument("--split",           default="val", choices=["val", "test"])
    p.add_argument("--feature_type",    default="jukebox")
    p.add_argument("--guidance_weight", type=float, default=2.0,
                   help="Solo CFG weight (EDGE default is 2)")
    p.add_argument("--out_dir",         default="eval/solo_pfc_baseline")
    p.add_argument("--seed",            type=int, default=1234)
    opt = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device       = {device}")
    print(f"[setup] checkpoint   = {opt.checkpoint}")
    print(f"[setup] feature_type = {opt.feature_type}")
    print(f"[setup] split        = {opt.split}")
    print(f"[setup] guidance     = {opt.guidance_weight} (solo CFG)")

    # ── Load EDGE in SOLO mode ─────────────────────────────────────────
    from EDGE import EDGE
    model = EDGE(opt.feature_type, opt.checkpoint, duet=False,
                 guidance_weight_music=opt.guidance_weight)
    model.eval()
    diffusion  = model.diffusion
    normalizer = model.normalizer
    smpl       = diffusion.smpl
    repr_dim   = model.repr_dim   # 151
    horizon    = model.horizon    # 150
    print(f"[setup] solo cond dim = 4800 (jukebox music features only)")

    # ── Enumerate val follower slices ──────────────────────────────────
    pairs = json.load(open(os.path.join(
        opt.data_dir, "splits", f"duet_pairs_{opt.split}.json")))
    split_dir = os.path.join(opt.data_dir, opt.split)
    feat_dir  = os.path.join(split_dir, f"{opt.feature_type}_feats")

    slices = []
    for pair in pairs:
        for npy in glob.glob(os.path.join(feat_dir,
                                          f"{pair['follower']}_slice*.npy")):
            slices.append({
                "name":     os.path.splitext(os.path.basename(npy))[0],
                "feat_npy": npy,
            })
    print(f"[setup] {len(slices)} slices to generate "
          f"(same {len(pairs)} {opt.split} pairs as duet eval)")

    # ── Generate solo motion for each slice ────────────────────────────
    out_dir = Path(opt.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gen_dir = out_dir / "gen"
    gen_dir.mkdir(exist_ok=True)

    print("[gen] running EDGE solo inference (DDIM sampling, music cond)")
    for idx, item in enumerate(tqdm(slices)):
        out_pkl = gen_dir / f"{item['name']}.pkl"
        if out_pkl.exists():
            continue

        music = torch.from_numpy(np.load(item["feat_npy"])).unsqueeze(0).to(device)
        # Solo mode: cond is ONLY music (4800-d), no lead concat.
        cond = music

        torch.manual_seed(opt.seed + idx)
        with torch.no_grad():
            sample = diffusion.ddim_sample((1, horizon, repr_dim), cond)

        # Unnormalize → split into (contacts, pose) → FK to joints.
        sample_unn = normalizer.unnormalize(sample.cpu())
        _, pose = torch.split(sample_unn, (4, repr_dim - 4), dim=2)
        pos = pose[:, :, :3].to(device)
        q   = ax_from_6v(pose[:, :, 3:].reshape(1, horizon, 24, 6).to(device))
        joints = smpl.forward(q, pos)[0].detach().cpu().numpy()  # [150, 24, 3]

        pickle.dump({"full_pose": joints}, open(out_pkl, "wb"))

    # ── Compute PFC ────────────────────────────────────────────────────
    print("\n[pfc] computing PFC on generated motions...")
    pfc = calc_pfc(str(gen_dir))
    print(f"\n{'=' * 60}")
    print(f"  EDGE solo PFC on {opt.split} (this work)   : {pfc:.4f}")
    print(f"  EDGE paper reported (AIST++ test, with CCL): 1.5363")
    print(f"  GT follower PFC (real human duet, val)     : 1.2607")
    print(f"  Our duet best (lead_only ep1700, val)      : 2.3505")
    print(f"{'=' * 60}")

    summary = {
        "checkpoint":            opt.checkpoint,
        "split":                 opt.split,
        "guidance_weight":       opt.guidance_weight,
        "n_slices":              len(slices),
        "edge_solo_pfc_repro":   float(pfc),
        "reference": {
            "edge_paper_solo_pfc":   1.5363,
            "gt_follower_pfc":       1.2607,
            "duet_lead_best_pfc":    2.3505,
        },
    }
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=2)
    print(f"\n[done] summary written to {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
