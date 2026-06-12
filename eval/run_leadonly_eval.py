#!/usr/bin/env python3
"""
eval/run_leadonly_eval.py

对单个 checkpoint 比较三种条件下的生成质量，重点回答：
"只用主舞（lead-only）生成伴舞，相比全条件掉多少？"

三组条件（通过 compositional CFG 的 guidance 权重切换）：
  full        w_music=2, w_lead=2   全条件（音乐 + 主舞）
  lead_only   w_music=0, w_lead=2   只用主舞   ← 重点
  music_only  w_music=2, w_lead=0   只用音乐（退回原版 EDGE，做对照）

每组在同一批 val 对齐切片上：
  - PFC（物理脚部接触，越低越好；附 GT 参照）
  - LMA（生成伴舞 vs 真实主舞 的相似度，越高越像在跟随；附真人对天花板）

为公平对比，同一切片在三组里使用相同的随机种子。
模型只加载一次，三组之间仅切换 diffusion.guidance_weight{,_lead}。

用法（需要 GPU，走 slurm）：
    python eval/run_leadonly_eval.py \
        --checkpoint runs/train/exp9/weights/train-2000.pt \
        --data_dir data --out_dir eval/leadonly_2000
"""

import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.dance_dataset import preprocess_motion_to_tensor
from eval.run_val_pfc import calc_pfc, motion_to_joints, sample_to_joints, slice_index
from eval.lma_similarity import compute_similarity
from vis import SMPLSkeleton
from EDGE import EDGE


# (名称, w_music, w_lead)
CONFIGS = [
    ("full",       2.0, 2.0),
    ("lead_only",  0.0, 2.0),
    ("music_only", 2.0, 0.0),
]


def build_aligned(pairs, motion_dir, feat_dir):
    aligned = []
    for pair in pairs:
        lead_map = {slice_index(p): p for p in
                    glob.glob(os.path.join(motion_dir, f"{pair['lead']}_slice*.pkl"))}
        foll_map = {slice_index(p): p for p in
                    glob.glob(os.path.join(motion_dir, f"{pair['follower']}_slice*.pkl"))}
        feat_map = {slice_index(p): p for p in
                    glob.glob(os.path.join(feat_dir, f"{pair['follower']}_slice*.npy"))}
        for si in sorted(set(lead_map) & set(foll_map) & set(feat_map)):
            aligned.append({
                "lead_pkl":     lead_map[si],
                "follower_pkl": foll_map[si],
                "feat_npy":     feat_map[si],
                "name":         f"{pair['follower']}_slice{si}",
            })
    return aligned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   default="runs/train/exp9/weights/train-2000.pt")
    parser.add_argument("--data_dir",     default="data")
    parser.add_argument("--out_dir",      default="eval/leadonly_2000")
    parser.add_argument("--feature_type", default="jukebox")
    parser.add_argument("--seed",         type=int, default=1234)
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {opt.checkpoint}")
    os.makedirs(opt.out_dir, exist_ok=True)

    # ── 数据 ────────────────────────────────────────────────────────────────
    pairs      = json.load(open(os.path.join(opt.data_dir, "splits", "duet_pairs_val.json")))
    val_dir    = os.path.join(opt.data_dir, "val")
    motion_dir = os.path.join(val_dir, "motions_sliced")
    feat_dir   = os.path.join(val_dir, f"{opt.feature_type}_feats")
    aligned    = build_aligned(pairs, motion_dir, feat_dir)
    print(f"Val pairs: {len(pairs)}  |  对齐切片: {len(aligned)}")

    # ── 预计算主舞 / GT 伴舞 关节（CPU FK，用于 LMA 与 GT PFC）────────────────
    smpl_cpu = SMPLSkeleton(device=None)
    gt_dir   = os.path.join(opt.out_dir, "ground_truth")
    os.makedirs(gt_dir, exist_ok=True)

    lead_joints  = {}   # name -> [T,24,3]
    gt_joints    = {}   # name -> [T,24,3]
    print("\n[Step 1] 计算主舞 + GT 伴舞 FK ...")
    for item in tqdm(aligned, desc="FK"):
        name = item["name"]
        ld = pickle.load(open(item["lead_pkl"], "rb"))
        lead_joints[name] = motion_to_joints(ld["pos"], ld["q"], smpl_cpu, "cpu")

        gt_pkl = os.path.join(gt_dir, f"{name}.pkl")
        if os.path.exists(gt_pkl):
            gt_joints[name] = pickle.load(open(gt_pkl, "rb"))["full_pose"]
        else:
            fd = pickle.load(open(item["follower_pkl"], "rb"))
            gj = motion_to_joints(fd["pos"], fd["q"], smpl_cpu, "cpu")
            pickle.dump({"full_pose": gj}, open(gt_pkl, "wb"))
            gt_joints[name] = gj

    gt_pfc      = calc_pfc(gt_dir)
    real_lma    = float(np.mean([
        compute_similarity(lead_joints[i["name"]], gt_joints[i["name"]])["total_score"]
        for i in aligned
    ]))
    print(f"GT PFC = {gt_pfc:.4f}   真人对 LMA(主舞 vs 真实伴舞) = {real_lma:.4f}")

    # ── 加载模型（一次）─────────────────────────────────────────────────────
    print("\n[Step 2] 加载模型 ...")
    model      = EDGE(opt.feature_type, opt.checkpoint, duet=True)
    model.eval()
    diffusion  = model.diffusion
    normalizer = model.normalizer
    smpl_gpu   = diffusion.smpl
    repr_dim   = model.repr_dim
    horizon    = model.horizon

    # ── 三组条件 ──────────────────────────────────────────────────────────────
    results = []
    for cfgname, wm, wl in CONFIGS:
        diffusion.guidance_weight      = wm
        diffusion.guidance_weight_lead = wl
        gen_dir = os.path.join(opt.out_dir, cfgname)
        os.makedirs(gen_dir, exist_ok=True)
        print(f"\n[{cfgname}] w_music={wm}  w_lead={wl}")

        lma_scores = []
        for idx, item in enumerate(tqdm(aligned, desc=f"  采样 {cfgname}")):
            name = item["name"]
            ld   = pickle.load(open(item["lead_pkl"], "rb"))
            lead_t = preprocess_motion_to_tensor(
                ld["pos"], ld["q"], normalizer,
            ).unsqueeze(0).to(device)
            music = torch.from_numpy(np.load(item["feat_npy"])).unsqueeze(0).to(device)
            cond  = torch.cat([lead_t, music.to(lead_t.dtype)], dim=-1)

            torch.manual_seed(opt.seed + idx)   # 三组同切片同噪声
            with torch.no_grad():
                sample = diffusion.ddim_sample((1, horizon, repr_dim), cond)

            gj = sample_to_joints(sample, normalizer, repr_dim, horizon, smpl_gpu, device)
            pickle.dump({"full_pose": gj}, open(os.path.join(gen_dir, f"{name}.pkl"), "wb"))
            lma_scores.append(compute_similarity(lead_joints[name], gj)["total_score"])

        pfc = calc_pfc(gen_dir)
        lma = float(np.mean(lma_scores))
        results.append({"cfg": cfgname, "wm": wm, "wl": wl, "pfc": pfc, "lma": lma})
        print(f"  → PFC={pfc:.4f}  LMA={lma:.4f}")

    # ── 汇总 ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print(f"  Checkpoint: {opt.checkpoint}")
    print(f"  对齐切片: {len(aligned)}   GT PFC={gt_pfc:.4f}   真人对 LMA={real_lma:.4f}")
    print("-" * 64)
    print(f"{'条件':<12}{'w_music':>8}{'w_lead':>8}{'PFC':>10}{'vs GT':>10}{'LMA':>10}")
    print("-" * 64)
    for r in results:
        print(f"{r['cfg']:<12}{r['wm']:>8.1f}{r['wl']:>8.1f}"
              f"{r['pfc']:>10.4f}{r['pfc']-gt_pfc:>+10.4f}{r['lma']:>10.4f}")
    print("-" * 64)
    print(f"{'GT(真人)':<12}{'':>8}{'':>8}{gt_pfc:>10.4f}{0.0:>+10.4f}{real_lma:>10.4f}")
    print("=" * 64)

    full = next((r for r in results if r["cfg"] == "full"), None)
    lead = next((r for r in results if r["cfg"] == "lead_only"), None)
    if full and lead:
        print(f"\nlead-only vs full：LMA {lead['lma']:.4f} vs {full['lma']:.4f} "
              f"(Δ={lead['lma']-full['lma']:+.4f})   "
              f"PFC {lead['pfc']:.4f} vs {full['pfc']:.4f} "
              f"(Δ={lead['pfc']-full['pfc']:+.4f})")

    json.dump(results, open(os.path.join(opt.out_dir, "summary.json"), "w"), indent=2)
    print(f"\n结果已写入 {opt.out_dir}/summary.json")


if __name__ == "__main__":
    main()
