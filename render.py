#!/usr/bin/env python3
"""
render.py — 独立渲染脚本

直接加载训练好的 checkpoint，对 val 集生成双人舞视频。
支持长序列：自动把所有对齐 slice 拼成一段长视频（用滑动窗口 long_ddim_sample）。

用法示例：
    # 渲染 val 集前 3 对，每对完整长度（所有 slice 拼接）
    python render.py --checkpoint runs/train/exp7/weights/train-950.pt --n_pairs 3

    # 只取每对的第 1 个 slice（快速预览）
    python render.py --checkpoint runs/train/exp7/weights/train-950.pt --n_pairs 3 --single_slice

Slurm 提交：
    sbatch run_render.sh
"""

import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pytorch3d.transforms import (axis_angle_to_quaternion,
                                  quaternion_to_axis_angle)

from dataset.dance_dataset import preprocess_motion_to_tensor
from dataset.quaternion import ax_from_6v, quat_slerp
from vis import SMPLSkeleton, skeleton_render


def slice_index(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return int(stem.split("_slice")[-1])


def load_lead_chunks(lead_map, common_slices, normalizer, device):
    """加载并预处理主舞各切片 → [N, T, 151] tensor（已归一化）"""
    chunks = []
    for si in common_slices:
        d = pickle.load(open(lead_map[si], "rb"))
        t = preprocess_motion_to_tensor(d["pos"], d["q"], normalizer)
        chunks.append(t)
    return torch.stack(chunks).to(device)          # [N, 150, 151]


def load_music_chunks(feat_map, common_slices, device):
    """加载各切片的 jukebox 特征 → [N, T, 4800] tensor"""
    chunks = [torch.from_numpy(np.load(feat_map[si])) for si in common_slices]
    return torch.stack(chunks).to(device)          # [N, 150, 4800]


def _stitch_and_fk(chunks, normalizer, repr_dim, horizon, smpl, device,
                   apply_drift_correction=True):
    """Stitch normalized motion chunks [N, T, 151] in PARAMETER space using
    raised-cosine cross-fade (pos) + slerp (joint rotations), then FK to
    world-space joint positions.

    Why parameter-space stitching (vs the old joint-space concat-and-drop)?
      - The old approach took chunk 0 in full and the LAST 2.5s of every
        subsequent chunk. Each boundary then exposed the model's internal
        transition from "inherited" frames (0..half-1, sync'd to the previous
        chunk via long_ddim_sample's `x[1:, :half] = x[:-1, half:]`) to
        "novel" frames (half..T-1, freshly denoised by this chunk). That
        transition tends to show up as a perceptible kink every 2.5s and a
        looping feel every 5s.
      - Blending the FULL overlap region instead, in axis-angle/pos space,
        lets adjacent chunks contribute together and removes the boundary.

    Args:
        chunks:                   [N, T, 151] normalized tensor.
        apply_drift_correction:   For ground-truth lead chunks (sliced from
            the same source motion), corresponding overlap frames are
            already identical and no correction is needed. For model-
            generated follower chunks, set True to snap chunk i's first
            frame onto chunk (i-1)'s frame at the start of the overlap.

    Returns: numpy array [output_T, 24, 3] in world coordinates.
    """
    N = chunks.shape[0]
    m = normalizer.unnormalize(chunks.cpu()).to(device)
    _, pose = torch.split(m, (4, repr_dim - 4), dim=2)
    pos = pose[:, :, :3].clone()                                     # [N, T, 3]
    q   = ax_from_6v(pose[:, :, 3:].reshape(N, horizon, 24, 6))      # [N, T, 24, 3]

    if N == 1:
        return smpl.forward(q, pos)[0].detach().cpu().numpy()

    s = horizon
    assert s % 2 == 0, "horizon must be even for half-overlap stitching"
    half = s // 2

    # (1) Root-position drift correction (follower only).
    if apply_drift_correction:
        for i in range(1, N):
            offset = pos[i - 1, half] - pos[i, 0]                    # [3]
            pos[i] = pos[i] + offset

    # (2) Raised-cosine ramp: 0 → 1 with zero derivative at both ends.
    ramp = 0.5 * (1.0 - torch.cos(
        torch.linspace(0, float(np.pi), half)
    )).to(device)

    fade_out = torch.ones((1, s, 1), device=device)
    fade_in  = torch.ones((1, s, 1), device=device)
    fade_out[:, half:, :] = (1.0 - ramp)[None, :, None]
    fade_in[:,  :half, :] = ramp[None, :, None]

    pos[:-1] = pos[:-1] * fade_out
    pos[1:]  = pos[1:]  * fade_in

    full_pos = torch.zeros((s + half * (N - 1), 3), device=device)
    idx = 0
    for ps in pos:
        full_pos[idx:idx + s] += ps
        idx += half

    # (3) Joint rotations: slerp on quaternions with the same raised-cosine weight.
    slerp_weight = ramp[None, :, None]
    left, right = q[:-1, half:], q[1:, :half]
    left  = axis_angle_to_quaternion(left)
    right = axis_angle_to_quaternion(right)
    merged = quat_slerp(left, right, slerp_weight)                   # [N-1, half, 24, 4]
    merged = quaternion_to_axis_angle(merged)                        # [N-1, half, 24, 3]

    full_q = torch.zeros((s + half * (N - 1), 24, 3), device=device)
    full_q[:half] += q[0, :half]
    idx = half
    for qs in merged:
        full_q[idx:idx + half] += qs
        idx += half
    full_q[idx:idx + half] += q[-1, half:]

    return smpl.forward(
        full_q.unsqueeze(0), full_pos.unsqueeze(0)
    )[0].detach().cpu().numpy()


def sample_to_joints(sample, normalizer, repr_dim, horizon, smpl, device):
    """模型输出 [N, T, 151] → 平滑拼接的关节轨迹（与音频对齐）。"""
    return _stitch_and_fk(sample, normalizer, repr_dim, horizon, smpl, device,
                          apply_drift_correction=True)


def lead_to_joints(lead_chunks, normalizer, repr_dim, horizon, smpl, device):
    """主舞归一化 tensor [N, T, 151] → 拼接关节轨迹。主舞各 chunk 来自同源
    GT 动作，重叠区帧本身就完全一致，无需 drift 校正。"""
    return _stitch_and_fk(lead_chunks, normalizer, repr_dim, horizon, smpl, device,
                          apply_drift_correction=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",     default="runs/train/exp7/weights/train-950.pt")
    parser.add_argument("--data_dir",       default="data")
    parser.add_argument("--out",            default="renders/duet_val_950")
    parser.add_argument("--split",          default="val", choices=["val", "test"],
                        help="渲染哪个 split 的 pair。val=ch01, test=ch02。"
                             "用对应的 duet_pairs_{split}.json 和 data/{split}/ 目录")
    parser.add_argument("--n_pairs",        type=int, default=3,
                        help="渲染前 N 对 pair")
    parser.add_argument("--single_slice",   action="store_true",
                        help="只渲染每对第 1 个 slice（快速预览）")
    parser.add_argument("--guidance_music", type=float, default=2.0)
    parser.add_argument("--guidance_lead",  type=float, default=2.0)
    parser.add_argument("--feature_type",   default="jukebox")
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {opt.checkpoint}")

    # ── 加载模型 ───────────────────────────────────────────────────────────
    from EDGE import EDGE
    model = EDGE(
        opt.feature_type,
        opt.checkpoint,
        duet=True,
        guidance_weight_music=opt.guidance_music,
        guidance_weight_lead=opt.guidance_lead,
    )
    model.eval()
    diffusion  = model.diffusion
    normalizer = model.normalizer
    smpl       = diffusion.smpl
    repr_dim   = model.repr_dim   # 151
    horizon    = model.horizon    # 150

    # ── 加载指定 split 的 pairs ────────────────────────────────────────────
    pairs = json.load(open(
        os.path.join(opt.data_dir, "splits", f"duet_pairs_{opt.split}.json")
    ))
    pairs = pairs[:opt.n_pairs]
    print(f"Split: {opt.split}  (n_pairs={len(pairs)})")

    split_dir  = os.path.join(opt.data_dir, opt.split)
    motion_dir = os.path.join(split_dir, "motions_sliced")
    feat_dir   = os.path.join(split_dir, f"{opt.feature_type}_feats")
    wav_dir    = os.path.join(split_dir, "wavs_sliced")

    os.makedirs(opt.out, exist_ok=True)
    epoch_tag  = os.path.basename(opt.checkpoint).replace(".pt", "")

    for pair_idx, pair in enumerate(pairs):
        lead_name     = pair["lead"]
        follower_name = pair["follower"]

        lead_map  = {slice_index(p): p for p in
                     glob.glob(os.path.join(motion_dir, f"{lead_name}_slice*.pkl"))}
        foll_map  = {slice_index(p): p for p in
                     glob.glob(os.path.join(motion_dir, f"{follower_name}_slice*.pkl"))}
        feat_map  = {slice_index(p): p for p in
                     glob.glob(os.path.join(feat_dir, f"{follower_name}_slice*.npy"))}
        wav_map   = {slice_index(p): p for p in
                     glob.glob(os.path.join(wav_dir, f"{follower_name}_slice*.wav"))}

        common = sorted(set(lead_map) & set(foll_map) & set(feat_map) & set(wav_map))
        if not common:
            print(f"[Pair {pair_idx}] No aligned slices, skipping.")
            continue

        # 关键修正：data/*/motions_sliced/ 里的切片是训练用 stride=0.5s 的 dense 切片，
        # 但 diffusion.long_ddim_sample 内部 stitch 假设 stride=2.5s（half=75 frames）。
        # 若直接把全部 dense 切片喂进去，会把 ~12s 真实音频拉伸成 37.5s 输出，
        # 切片之间错位 2s/5s/10s... 导致渲染视频每 5s 出现循环和顿挫。
        # 等价于 EDGE 原版 `slice_audio(stride=2.5, length=5)` 在推理时重新切。
        common = [i for i in common if i % 5 == 0]
        if not common:
            print(f"[Pair {pair_idx}] No 2.5s-stride slices, skipping.")
            continue

        if opt.single_slice:
            common = [common[0]]

        n_slices = len(common)
        # 每个 2.5s-stride chunk 长 5s，相邻覆盖 2.5s，总时长 = 5 + 2.5*(N-1) = 2.5*N + 2.5
        duration = 2.5 * n_slices + 2.5 if n_slices > 1 else 5
        print(f"\n[Pair {pair_idx}] {lead_name} → {follower_name}")
        print(f"  {n_slices} chunks (2.5s stride) = {duration}s output")

        # ── 构建 cond [N, T, 4951] ────────────────────────────────────────
        lead_chunks  = load_lead_chunks(lead_map, common, normalizer, device)
        music_chunks = load_music_chunks(feat_map, common, device)
        cond = torch.cat([lead_chunks, music_chunks], dim=-1)  # [N, 150, 4951]

        # ── 推理 ──────────────────────────────────────────────────────────
        print(f"  Sampling ({n_slices} chunks via {'ddim' if n_slices==1 else 'long_ddim'})...")
        with torch.no_grad():
            shape = (n_slices, horizon, repr_dim)
            if n_slices == 1:
                sample = diffusion.ddim_sample(shape, cond)
            else:
                # long_ddim_sample 要求 batch > 1，自动做滑动窗口拼接
                sample = diffusion.long_ddim_sample(shape, cond)

        # ── FK → 关节轨迹 ─────────────────────────────────────────────────
        follower_joints = sample_to_joints(sample, normalizer, repr_dim, horizon, smpl, device)
        lead_joints     = lead_to_joints(lead_chunks, normalizer, repr_dim, horizon, smpl, device)

        # ── 拼接音频（多个 slice wav 首尾相接）───────────────────────────
        wav_paths = [wav_map[si] for si in common]

        print(f"  Rendering → {opt.out}/")
        skeleton_render(
            follower_joints,
            epoch=epoch_tag,
            out=opt.out,
            name=wav_paths if n_slices > 1 else wav_paths[0],
            sound=True,
            stitch=(n_slices > 1),
            poses_lead=lead_joints,
        )

    print(f"\nDone. Videos saved to: {opt.out}/")


if __name__ == "__main__":
    main()
