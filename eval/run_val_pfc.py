#!/usr/bin/env python3
"""
eval/run_val_pfc.py

对所有保存的 checkpoint 批量评估 PFC，输出对比表，找到最优 epoch。

流程：
  1. 对 val set 真实伴舞做一次 FK，保存 ground_truth/ pkl（只跑一次）
  2. 对每个 checkpoint，生成伴舞动作并计算 PFC
  3. 打印对比表（含 val_loss、LMA、PFC）

用法（需要 GPU）：
    python eval/run_val_pfc.py \\
        --weights_dir runs/train/exp7/weights \\
        --out_dir eval/pfc_val
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
from dataset.quaternion import ax_from_6v
from vis import SMPLSkeleton
from pytorch3d.transforms import (
    RotateAxisAngle, axis_angle_to_quaternion,
    quaternion_multiply, quaternion_to_axis_angle,
)


# ── PFC 计算（直接嵌入，避免打印干扰）─────────────────────────────────────

def calc_pfc(pkl_dir):
    """返回 PFC 均值（×10000），越低越好。"""
    import random
    DT = 1 / 30
    up_dir   = 2
    flat_dirs = [i for i in range(3) if i != up_dir]
    foot_idx  = [7, 10, 8, 11]

    files = glob.glob(os.path.join(pkl_dir, "*.pkl"))
    if len(files) > 1000:
        files = random.sample(files, 1000)
    scores = []
    for pkl in files:
        info     = pickle.load(open(pkl, "rb"))
        joint3d  = info["full_pose"]                                  # [T, 24, 3]
        root_v   = (joint3d[1:, 0, :] - joint3d[:-1, 0, :]) / DT
        root_a   = (root_v[1:] - root_v[:-1]) / DT
        root_a[:, up_dir] = np.maximum(root_a[:, up_dir], 0)
        root_a   = np.linalg.norm(root_a, axis=-1)
        if root_a.max() == 0:
            continue
        root_a /= root_a.max()

        feet   = joint3d[:, foot_idx]
        foot_v = np.linalg.norm(
            feet[2:, :, flat_dirs] - feet[1:-1, :, flat_dirs], axis=-1
        )
        fm     = np.zeros((len(foot_v), 2))
        fm[:, 0] = np.minimum(foot_v[:, 0], foot_v[:, 1])
        fm[:, 1] = np.minimum(foot_v[:, 2], foot_v[:, 3])
        scores.append((fm[:, 0] * fm[:, 1] * root_a).mean())

    return float(np.mean(scores) * 10000) if scores else float("nan")


# ── 运动数据处理 ────────────────────────────────────────────────────────────

def motion_to_joints(pos_np, q_np, smpl, device):
    """原始60fps运动 → Z-up关节位置 [T, 24, 3]（降采样到30fps）"""
    pos_np  = pos_np[::2]
    q_np    = q_np[::2]
    T       = pos_np.shape[0]
    pos     = torch.Tensor(pos_np)
    local_q = torch.Tensor(q_np).reshape(T, -1, 3)

    root_q      = local_q[:, :1, :]
    root_q_quat = axis_angle_to_quaternion(root_q)
    rotation    = torch.Tensor([0.7071068, 0.7071068, 0, 0])
    root_q_quat = quaternion_multiply(rotation, root_q_quat)
    local_q[:, :1, :] = quaternion_to_axis_angle(root_q_quat)
    pos = RotateAxisAngle(90, axis="X", degrees=True).transform_points(pos)

    joints = smpl.forward(local_q.unsqueeze(0).to(device),
                          pos.unsqueeze(0).to(device))
    return joints[0].detach().cpu().numpy()


def sample_to_joints(sample, normalizer, repr_dim, horizon, smpl, device):
    """模型输出 [1, T, 151] → Z-up关节位置 [T, 24, 3]"""
    sample  = normalizer.unnormalize(sample.cpu())
    _, pose = torch.split(sample, (4, repr_dim - 4), dim=2)
    pos     = pose[:, :, :3].to(device)
    q       = ax_from_6v(pose[:, :, 3:].reshape(1, horizon, 24, 6).to(device))
    joints  = smpl.forward(q, pos)
    return joints[0].detach().cpu().numpy()


def slice_index(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return int(stem.split("_slice")[-1])


# ── 主程序 ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_dir",    default="runs/train/exp7/weights",
                        help="包含 train-*.pt 的目录")
    parser.add_argument("--data_dir",       default="data")
    parser.add_argument("--out_dir",        default="eval/pfc_val")
    parser.add_argument("--guidance_music", type=float, default=2.0)
    parser.add_argument("--guidance_lead",  type=float, default=2.0)
    parser.add_argument("--feature_type",   default="jukebox")
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    gt_dir = os.path.join(opt.out_dir, "ground_truth")
    os.makedirs(gt_dir, exist_ok=True)

    # ── 找出所有 checkpoint，按 epoch 排序 ────────────────────────────────
    ckpt_paths = sorted(
        glob.glob(os.path.join(opt.weights_dir, "train-*.pt")),
        key=lambda p: int(os.path.basename(p).replace("train-", "").replace(".pt", ""))
    )
    if not ckpt_paths:
        print(f"No checkpoints found in {opt.weights_dir}")
        return
    print(f"Found {len(ckpt_paths)} checkpoints: "
          f"epoch {os.path.basename(ckpt_paths[0])} ~ {os.path.basename(ckpt_paths[-1])}")

    # ── 加载 val pairs ────────────────────────────────────────────────────
    pairs      = json.load(open(os.path.join(opt.data_dir, "splits", "duet_pairs_val.json")))
    val_dir    = os.path.join(opt.data_dir, "val")
    motion_dir = os.path.join(val_dir, "motions_sliced")
    feat_dir   = os.path.join(val_dir, f"{opt.feature_type}_feats")
    print(f"Val pairs: {len(pairs)}")

    # pair → 对齐切片列表（只算一次）
    aligned = []
    for pair in pairs:
        lead_map     = {slice_index(p): p for p in
                        glob.glob(os.path.join(motion_dir, f"{pair['lead']}_slice*.pkl"))}
        follower_map = {slice_index(p): p for p in
                        glob.glob(os.path.join(motion_dir, f"{pair['follower']}_slice*.pkl"))}
        feat_map     = {slice_index(p): p for p in
                        glob.glob(os.path.join(feat_dir, f"{pair['follower']}_slice*.npy"))}
        common = sorted(set(lead_map) & set(follower_map) & set(feat_map))
        for si in common:
            aligned.append({
                "lead_pkl":      lead_map[si],
                "follower_pkl":  follower_map[si],
                "feat_npy":      feat_map[si],
                "name":          f"{pair['follower']}_slice{si}",
            })
    print(f"Total aligned slices: {len(aligned)}")

    # ── Step 1：ground truth FK（只跑一次）───────────────────────────────
    # 用第一个 checkpoint 的 smpl（结构固定，与 epoch 无关）
    from EDGE import EDGE
    print("\n[Step 1] Computing ground truth FK...")
    _tmp_model = EDGE(opt.feature_type, ckpt_paths[0], duet=True)
    smpl_cpu   = SMPLSkeleton(device=None)   # CPU smpl 用于 GT（避免 GPU 内存占用）
    gt_pfc_ready = True

    for item in tqdm(aligned, desc="GT FK"):
        gt_pkl_path = os.path.join(gt_dir, f"{item['name']}.pkl")
        if os.path.exists(gt_pkl_path):
            continue                          # 已经算过，跳过
        gt_data   = pickle.load(open(item["follower_pkl"], "rb"))
        gt_joints = motion_to_joints(gt_data["pos"], gt_data["q"], smpl_cpu, "cpu")
        pickle.dump({"full_pose": gt_joints}, open(gt_pkl_path, "wb"))

    gt_pfc = calc_pfc(gt_dir)
    print(f"Ground Truth PFC = {gt_pfc:.4f}")
    del _tmp_model

    # ── Step 2：对每个 checkpoint 推理并计算 PFC ─────────────────────────
    results = []

    for ckpt_path in ckpt_paths:
        epoch = int(os.path.basename(ckpt_path).replace("train-", "").replace(".pt", ""))
        gen_dir = os.path.join(opt.out_dir, f"epoch_{epoch:04d}")
        os.makedirs(gen_dir, exist_ok=True)

        print(f"\n[Epoch {epoch}] Loading model...")
        model      = EDGE(opt.feature_type, ckpt_path, duet=True,
                          guidance_weight_music=opt.guidance_music,
                          guidance_weight_lead=opt.guidance_lead)
        model.eval()
        diffusion  = model.diffusion
        normalizer = model.normalizer
        smpl       = diffusion.smpl
        repr_dim   = model.repr_dim
        horizon    = model.horizon

        for item in tqdm(aligned, desc=f"  Inference epoch {epoch}"):
            gen_pkl_path = os.path.join(gen_dir, f"{item['name']}.pkl")
            if os.path.exists(gen_pkl_path):
                continue

            lead_data   = pickle.load(open(item["lead_pkl"], "rb"))
            lead_tensor = preprocess_motion_to_tensor(
                lead_data["pos"], lead_data["q"], normalizer
            ).unsqueeze(0).to(device)

            music_feat = torch.from_numpy(
                np.load(item["feat_npy"])
            ).unsqueeze(0).to(device)

            cond = torch.cat([lead_tensor, music_feat], dim=-1)

            with torch.no_grad():
                sample = diffusion.ddim_sample((1, horizon, repr_dim), cond)

            gen_joints = sample_to_joints(sample, normalizer, repr_dim, horizon, smpl, device)
            pickle.dump({"full_pose": gen_joints}, open(gen_pkl_path, "wb"))

        gen_pfc = calc_pfc(gen_dir)
        results.append({"epoch": epoch, "gen_pfc": gen_pfc})
        print(f"  Epoch {epoch:4d}  PFC = {gen_pfc:.4f}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── 汇总打印 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 45)
    print(f"{'Epoch':>8}  {'Gen PFC':>10}  {'vs GT':>10}")
    print("-" * 45)
    best = min(results, key=lambda r: r["gen_pfc"])
    for r in results:
        marker = " ◀ best" if r["epoch"] == best["epoch"] else ""
        print(f"{r['epoch']:>8}  {r['gen_pfc']:>10.4f}  "
              f"{r['gen_pfc'] - gt_pfc:>+10.4f}{marker}")
    print("-" * 45)
    print(f"{'GT':>8}  {gt_pfc:>10.4f}")
    print("=" * 45)
    print(f"\n最优 checkpoint: train-{best['epoch']}.pt  (PFC={best['gen_pfc']:.4f})")


if __name__ == "__main__":
    main()
