#!/usr/bin/env python3
"""
eval/run_full_sweep.py

对 weights_dir 下所有 checkpoint 做完整的 PFC + LMA 评估，输出可直接做报表
的 csv + json。

复用策略
--------
  - 已经在 ``--pkl_dir`` 下存在 ``epoch_XXXX/<name>.pkl`` 的 epoch 直接读取
    ``full_pose`` 算 PFC / LMA（跳过推理）
  - 缺失的 epoch 走完整推理（加载 checkpoint → DDIM 采样 → FK → 存 pkl）
  - 主舞 / GT 关节只算一次，放到 ``--pkl_dir/ground_truth``
  - 每个 epoch 跑完就 dump 一次 summary.json / sweep.csv，半路挂掉也不丢

注意
----
  - 已存在的 .pkl 不会被覆盖；当前的 pfc_val/epoch_0050..1000 是 5/22 用
    默认 guidance (2.0, 2.0) 跑的，无固定随机种子。新生成的部分会用
    ``--seed`` 保证可复现。混用对 epoch-级聚合统计无影响（93 切片均值）
  - 想强制重跑某些 epoch，删掉对应 ``epoch_XXXX/`` 目录即可

用法（GPU + slurm）：
    python eval/run_full_sweep.py \
        --weights_dir runs/train/exp9/weights \
        --pkl_dir     eval/pfc_val \
        --out_dir     eval/full_sweep
"""

import argparse
import csv
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
from eval.lma_similarity import compute_similarity
from eval.run_val_pfc import (
    calc_pfc, motion_to_joints, sample_to_joints, slice_index,
)
from vis import SMPLSkeleton


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

def build_aligned(pairs, motion_dir, feat_dir):
    """根据 duet pairs JSON + 切片目录构造对齐 slice 列表。"""
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


def epoch_of(ckpt_path):
    return int(os.path.basename(ckpt_path).replace("train-", "").replace(".pt", ""))


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights_dir",    default="runs/train/exp9/weights")
    parser.add_argument("--pkl_dir",        default="eval/pfc_val",
                        help="生成 motion .pkl 的根目录（已有的子目录直接复用）")
    parser.add_argument("--data_dir",       default="data")
    parser.add_argument("--out_dir",        default="eval/full_sweep")
    parser.add_argument("--guidance_music", type=float, default=2.0)
    parser.add_argument("--guidance_lead",  type=float, default=2.0)
    parser.add_argument("--feature_type",   default="jukebox")
    parser.add_argument("--seed",           type=int, default=1234,
                        help="新生成时的种子。已存在的 .pkl 不重跑")
    parser.add_argument("--epoch_min",      type=int, default=0)
    parser.add_argument("--epoch_max",      type=int, default=10**9)
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(opt.out_dir, exist_ok=True)
    os.makedirs(opt.pkl_dir, exist_ok=True)

    # ── 收集 checkpoint ─────────────────────────────────────────────────────
    ckpts = sorted(
        glob.glob(os.path.join(opt.weights_dir, "train-*.pt")),
        key=epoch_of,
    )
    ckpts = [c for c in ckpts if opt.epoch_min <= epoch_of(c) <= opt.epoch_max]
    if not ckpts:
        print(f"在 {opt.weights_dir} 没找到 train-*.pt")
        return
    print(f"将评估 {len(ckpts)} 个 checkpoint：epoch "
          f"{epoch_of(ckpts[0])} ~ {epoch_of(ckpts[-1])}")

    # ── 数据：val pairs + 对齐切片 ──────────────────────────────────────────
    pairs      = json.load(open(os.path.join(opt.data_dir, "splits", "duet_pairs_val.json")))
    val_dir    = os.path.join(opt.data_dir, "val")
    motion_dir = os.path.join(val_dir, "motions_sliced")
    feat_dir   = os.path.join(val_dir, f"{opt.feature_type}_feats")
    aligned    = build_aligned(pairs, motion_dir, feat_dir)
    print(f"Val pairs: {len(pairs)}  |  对齐切片: {len(aligned)}")

    # ── 主舞 + GT 伴舞关节（CPU FK，一次）───────────────────────────────────
    smpl_cpu = SMPLSkeleton(device=None)
    gt_dir   = os.path.join(opt.pkl_dir, "ground_truth")
    os.makedirs(gt_dir, exist_ok=True)

    print("\n[GT] 主舞 + GT 伴舞 FK ...")
    lead_joints = {}
    gt_joints   = {}
    for item in tqdm(aligned, desc="GT FK"):
        name = item["name"]
        ld   = pickle.load(open(item["lead_pkl"], "rb"))
        lead_joints[name] = motion_to_joints(ld["pos"], ld["q"], smpl_cpu, "cpu")

        gt_pkl = os.path.join(gt_dir, f"{name}.pkl")
        if os.path.exists(gt_pkl):
            gt_joints[name] = pickle.load(open(gt_pkl, "rb"))["full_pose"]
        else:
            fd = pickle.load(open(item["follower_pkl"], "rb"))
            gj = motion_to_joints(fd["pos"], fd["q"], smpl_cpu, "cpu")
            pickle.dump({"full_pose": gj}, open(gt_pkl, "wb"))
            gt_joints[name] = gj

    gt_pfc   = calc_pfc(gt_dir)
    real_lma = float(np.mean([
        compute_similarity(lead_joints[i["name"]], gt_joints[i["name"]])["total_score"]
        for i in aligned
    ]))
    print(f"  GT PFC = {gt_pfc:.4f}   真人对 LMA(主舞 vs 真实伴舞) = {real_lma:.4f}")

    # ── 逐 epoch 评估 ───────────────────────────────────────────────────────
    from EDGE import EDGE
    results = []

    for ckpt in ckpts:
        epoch   = epoch_of(ckpt)
        gen_dir = os.path.join(opt.pkl_dir, f"epoch_{epoch:04d}")
        os.makedirs(gen_dir, exist_ok=True)

        # 检查缺哪些切片
        missing = [it for it in aligned
                   if not os.path.exists(os.path.join(gen_dir, f"{it['name']}.pkl"))]

        if missing:
            print(f"\n[Epoch {epoch}] 加载模型，补 {len(missing)}/{len(aligned)} 个切片 ...")
            model      = EDGE(opt.feature_type, ckpt, duet=True,
                              guidance_weight_music=opt.guidance_music,
                              guidance_weight_lead=opt.guidance_lead)
            model.eval()
            diffusion  = model.diffusion
            normalizer = model.normalizer
            smpl_gpu   = diffusion.smpl
            repr_dim   = model.repr_dim
            horizon    = model.horizon

            for idx, item in enumerate(tqdm(missing, desc=f"  推理 epoch {epoch}")):
                ld    = pickle.load(open(item["lead_pkl"], "rb"))
                lead  = preprocess_motion_to_tensor(
                    ld["pos"], ld["q"], normalizer,
                ).unsqueeze(0).to(device)
                music = torch.from_numpy(np.load(item["feat_npy"])).unsqueeze(0).to(device)
                cond  = torch.cat([lead, music.to(lead.dtype)], dim=-1)

                torch.manual_seed(opt.seed + idx)
                with torch.no_grad():
                    sample = diffusion.ddim_sample((1, horizon, repr_dim), cond)
                gj = sample_to_joints(sample, normalizer, repr_dim, horizon,
                                      smpl_gpu, device)
                pickle.dump({"full_pose": gj},
                            open(os.path.join(gen_dir, f"{item['name']}.pkl"), "wb"))

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            print(f"\n[Epoch {epoch}] 全部 .pkl 已存在，跳过推理")

        # ── 算指标 ───────────────────────────────────────────────────────────
        pfc = calc_pfc(gen_dir)

        lma_scores = []
        for item in aligned:
            gj = pickle.load(open(
                os.path.join(gen_dir, f"{item['name']}.pkl"), "rb"
            ))["full_pose"]
            lma_scores.append(
                compute_similarity(lead_joints[item["name"]], gj)["total_score"]
            )
        lma     = float(np.mean(lma_scores))
        lma_std = float(np.std(lma_scores))

        results.append({
            "epoch":       epoch,
            "pfc":         pfc,
            "pfc_vs_gt":   pfc - gt_pfc,
            "lma":         lma,
            "lma_std":     lma_std,
            "lma_vs_real": lma - real_lma,
        })
        print(f"  Epoch {epoch:4d}  PFC={pfc:.4f} (Δ={pfc-gt_pfc:+.4f})  "
              f"LMA={lma:.4f}±{lma_std:.3f} (Δ={lma-real_lma:+.4f})")

        # 每个 epoch 都 dump 一次，半路挂掉也不丢数据
        summary = {
            "weights_dir":    opt.weights_dir,
            "n_aligned":      len(aligned),
            "guidance_music": opt.guidance_music,
            "guidance_lead":  opt.guidance_lead,
            "gt_pfc":         gt_pfc,
            "real_lma":       real_lma,
            "results":        results,
        }
        json.dump(summary,
                  open(os.path.join(opt.out_dir, "summary.json"), "w"), indent=2)
        with open(os.path.join(opt.out_dir, "sweep.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "epoch", "pfc", "pfc_vs_gt", "lma", "lma_std", "lma_vs_real",
            ])
            writer.writeheader()
            writer.writerows(results)

    # ── 汇总表 ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"  Weights: {opt.weights_dir}   切片: {len(aligned)}")
    print(f"  GT PFC = {gt_pfc:.4f}   真人对 LMA = {real_lma:.4f}")
    print("-" * 72)
    print(f"{'Epoch':>6}{'PFC':>10}{'vs GT':>10}{'LMA':>10}{'vs Real':>10}")
    print("-" * 72)
    best_pfc = min(results, key=lambda r: r["pfc"])
    best_lma = max(results, key=lambda r: r["lma"])
    for r in results:
        marks = ""
        if r["epoch"] == best_pfc["epoch"]:
            marks += " ◀PFC"
        if r["epoch"] == best_lma["epoch"]:
            marks += " ◀LMA"
        print(f"{r['epoch']:>6}{r['pfc']:>10.4f}{r['pfc_vs_gt']:>+10.4f}"
              f"{r['lma']:>10.4f}{r['lma_vs_real']:>+10.4f}{marks}")
    print("=" * 72)
    print(f"\n输出：")
    print(f"  {opt.out_dir}/summary.json")
    print(f"  {opt.out_dir}/sweep.csv")


if __name__ == "__main__":
    main()
