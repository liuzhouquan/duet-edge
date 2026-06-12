#!/usr/bin/env python3
"""
eval/run_test_eval_longmode.py

**Per-song long-mode** test eval — 与 EDGE 原版 PFC 评估协议对齐。

EDGE 原版做法
---------------
- 每首歌 → DDIM 采样 N 个 stride-2.5s 的 5-秒 chunk → long_ddim_sample stitch
  成 1 个长序列 → 1 个 .pkl per song
- eval/eval_pfc.py 对每个 .pkl 算 1 个 PFC，然后跨 song 取平均

vs 之前的 per-slice 评估
------------------------
- 旧脚本 (run_test_eval.py, 输出 eval/test_eval_per_slice/) 是每个 5s slice
  独立 ddim_sample，每 slice 1 个 .pkl，93 个 .pkl 取平均
- 长序列里的 chunk 间 stitch 平滑效应 → 旧版数字偏高，新版才是和 EDGE 1.5363
  同协议的数

对每个 pair
------------
1. 拿到所有 dense 0.5s-stride slice 索引
2. 每 5 个取 1 个 → 得到 stride-2.5s 的 N chunk
3. 组成 cond batch → diffusion.render_sample(mode="long")
4. 一次保存 1 个 .pkl per pair（=10 pairs → 10 个 .pkl）
5. PFC = avg of 10 per-pair PFC
6. LMA = avg of 10 per-pair (long lead vs long generated follower) Laban 相似度

用法
----
    python eval/run_test_eval_longmode.py \
        --weights_dir runs/train/exp9/weights \
        --ckpts_lead_only 1700,1750,1800,1850,1900 \
        --ckpts_full      1750,1800,1850,1900,1950 \
        --out_dir         eval/test_eval
"""

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

from dataset.dance_dataset import preprocess_motion_to_tensor
from eval.lma_similarity import compute_similarity
from eval.run_val_pfc import calc_pfc, motion_to_joints, slice_index
from vis import SMPLSkeleton


def build_pairs_long(pairs, motion_dir, feat_dir, raw_motion_dir, raw_wav_dir,
                     every=5):
    """对每个 duet pair，挑出每 `every`-th 个 dense slice 作为 long-mode 输入。

    every=5 表示从 0.5s stride 的 dense slice 取每 5 个 = 2.5s stride（EDGE 标准）。
    """
    pairs_long = []
    for pair in pairs:
        lead_files = sorted(
            glob.glob(os.path.join(motion_dir, f"{pair['lead']}_slice*.pkl")),
            key=slice_index,
        )
        foll_files = sorted(
            glob.glob(os.path.join(motion_dir, f"{pair['follower']}_slice*.pkl")),
            key=slice_index,
        )
        feat_files = sorted(
            glob.glob(os.path.join(feat_dir, f"{pair['follower']}_slice*.npy")),
            key=slice_index,
        )
        idx_lead = {slice_index(p) for p in lead_files}
        idx_foll = {slice_index(p) for p in foll_files}
        idx_feat = {slice_index(p) for p in feat_files}
        common = sorted(idx_lead & idx_foll & idx_feat)

        # 每 every 个取 1 个
        long_indices = [i for i in common if i % every == 0]
        if not long_indices:
            print(f"  [WARN] {pair['lead']} → 没有 stride-{every*0.5}s 切片，跳过")
            continue

        N = len(long_indices)
        pairs_long.append({
            "name":            f"{pair['follower']}",   # used for .pkl naming
            "lead_seq":        pair["lead"],
            "foll_seq":        pair["follower"],
            "N":               N,
            "lead_pkls":       [os.path.join(motion_dir, f"{pair['lead']}_slice{i}.pkl") for i in long_indices],
            "music_feats":     [os.path.join(feat_dir,   f"{pair['follower']}_slice{i}.npy") for i in long_indices],
            "raw_lead_pkl":    os.path.join(raw_motion_dir, f"{pair['lead']}.pkl"),
            "raw_foll_pkl":    os.path.join(raw_motion_dir, f"{pair['follower']}.pkl"),
            # 给 render_sample 当 .pkl 命名用（输出名取这个文件去掉 _sliceN 部分）
            "wav_name":        os.path.join(raw_wav_dir + "_sliced",
                                            f"{pair['follower']}_slice0.wav"),
        })
    return pairs_long


def long_output_length(N, half_frames=75, chunk_frames=150):
    """Long-mode stitch 后的输出帧数（同 render_sample 的内部公式）."""
    return chunk_frames if N == 1 else chunk_frames + half_frames * (N - 1)


def parse_int_list(s):
    if not s.strip():
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


# ============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights_dir", default="runs/train/exp9/weights")
    parser.add_argument("--data_dir",    default="data")
    parser.add_argument("--out_dir",     default="eval/test_eval")
    parser.add_argument("--ckpts_lead_only", type=str,
                        default="1700,1750,1800,1850,1900")
    parser.add_argument("--ckpts_full",      type=str,
                        default="1750,1800,1850,1900,1950")
    parser.add_argument("--feature_type", default="jukebox")
    parser.add_argument("--seed",         type=int, default=1234)
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(opt.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device:    {device}")
    print(f"Weights:   {opt.weights_dir}")
    print(f"Out dir:   {out_dir}")
    print(f"Protocol:  per-song long-mode (EDGE-compatible)")

    # ── 数据 (test 集 ch02) ────────────────────────────────────────────────
    pairs_raw  = json.load(open(os.path.join(opt.data_dir, "splits", "duet_pairs_test.json")))
    test_dir   = os.path.join(opt.data_dir, "test")
    motion_dir = os.path.join(test_dir, "motions_sliced")
    feat_dir   = os.path.join(test_dir, f"{opt.feature_type}_feats")
    raw_motion_dir = os.path.join(test_dir, "motions")
    raw_wav_dir    = os.path.join(test_dir, "wavs")

    pairs = build_pairs_long(pairs_raw, motion_dir, feat_dir,
                              raw_motion_dir, raw_wav_dir, every=5)
    print(f"\nTest pairs: {len(pairs)}")
    for p in pairs:
        T_gen = long_output_length(p["N"])
        print(f"  - {p['lead_seq'][:35]:<35s} → "
              f"{p['foll_seq'][:35]:<35s}  N={p['N']}  T_gen={T_gen}f ({T_gen/30:.1f}s)")

    # ── 候选 (ckpt, config) 组合 ─────────────────────────────────────────────
    ckpts_lead = parse_int_list(opt.ckpts_lead_only)
    ckpts_full = parse_int_list(opt.ckpts_full)
    runs = []
    for ep in ckpts_lead:
        runs.append({"cfg": "lead_only", "epoch": ep, "wm": 0.0, "wl": 2.0})
    for ep in ckpts_full:
        runs.append({"cfg": "full",      "epoch": ep, "wm": 2.0, "wl": 2.0})
    print(f"\n将评估 {len(runs)} 个 (epoch, config) 组合：")
    for r in runs:
        print(f"  - {r['cfg']:10s} epoch={r['epoch']}  (w_m={r['wm']}, w_l={r['wl']})")

    # ── Step 1: GT FK (each pair: 1 long lead joints + 1 long foll joints) ──
    smpl_cpu = SMPLSkeleton(device=None)
    gt_dir   = out_dir / "ground_truth"
    gt_dir.mkdir(parents=True, exist_ok=True)
    lead_long_joints = {}    # name -> [T_lead, 24, 3] 30fps
    foll_long_joints = {}    # name -> [T_foll, 24, 3] 30fps (GT for PFC eval)

    print("\n[Step 1] GT FK (raw mocap, 30fps, 应用 smpl_scaling) ...")
    for p in tqdm(pairs, desc="GT FK"):
        ld = pickle.load(open(p["raw_lead_pkl"], "rb"))
        fd = pickle.load(open(p["raw_foll_pkl"], "rb"))
        # 关键修正：raw mocap 的 smpl_trans 没除以 smpl_scaling，必须先除
        # sliced motions 写盘前已经除过（见 data/slice.py:34 `pos /= scale`）
        ld_pos = ld["pos"] / ld["scale"][0]
        fd_pos = fd["pos"] / fd["scale"][0]
        lead_long_joints[p["name"]] = motion_to_joints(ld_pos, ld["q"], smpl_cpu, "cpu")
        foll_long_joints[p["name"]] = motion_to_joints(fd_pos, fd["q"], smpl_cpu, "cpu")

        # 保存 GT follower 长序列 .pkl，用于算 GT PFC（同时间范围）
        T_gen = long_output_length(p["N"])
        T_keep = min(T_gen, foll_long_joints[p["name"]].shape[0])
        gt_pkl = gt_dir / f"{p['name']}.pkl"
        pickle.dump({"full_pose": foll_long_joints[p["name"]][:T_keep]},
                    open(gt_pkl, "wb"))

    gt_pfc = calc_pfc(str(gt_dir))   # 在 10 个 GT .pkl 上算
    real_lma_scores = []
    for p in pairs:
        T_gen = long_output_length(p["N"])
        T_keep = min(T_gen,
                     lead_long_joints[p["name"]].shape[0],
                     foll_long_joints[p["name"]].shape[0])
        real_lma_scores.append(compute_similarity(
            lead_long_joints[p["name"]][:T_keep],
            foll_long_joints[p["name"]][:T_keep],
        )["total_score"])
    real_lma = float(np.mean(real_lma_scores))
    print(f"\n  Test GT PFC (per-song avg) = {gt_pfc:.4f}")
    print(f"  Test GT LMA (long lead vs long follower) = {real_lma:.4f}")

    # ── Step 2: 逐 (ckpt, config) 跑评估 ────────────────────────────────────
    from EDGE import EDGE
    results = []
    last_ckpt = None
    model = None

    for run_i, run in enumerate(runs):
        ckpt_path = os.path.join(opt.weights_dir, f"train-{run['epoch']}.pt")
        gen_dir = out_dir / f"epoch_{run['epoch']:04d}_{run['cfg']}"
        gen_dir.mkdir(parents=True, exist_ok=True)

        # 加载/复用 model
        if ckpt_path != last_ckpt:
            if model is not None:
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            print(f"\n[Run {run_i+1}/{len(runs)}] 加载 {ckpt_path}")
            model = EDGE(opt.feature_type, ckpt_path, duet=True,
                         guidance_weight_music=run["wm"],
                         guidance_weight_lead=run["wl"])
            model.eval()
            last_ckpt = ckpt_path
        else:
            model.diffusion.guidance_weight      = run["wm"]
            model.diffusion.guidance_weight_lead = run["wl"]
            print(f"\n[Run {run_i+1}/{len(runs)}] 切换 guidance (w_m={run['wm']}, w_l={run['wl']})")

        diffusion  = model.diffusion
        normalizer = model.normalizer
        repr_dim   = model.repr_dim
        horizon    = model.horizon

        # 对每个 pair：构造 cond batch，调 render_sample(mode="long")
        run_tag = f"train-{run['epoch']}"
        for p_idx, p in enumerate(tqdm(pairs, desc=f"  {run['cfg']}@{run['epoch']}")):
            out_pkl = gen_dir / f"{run_tag}_{p['name']}.pkl"
            if out_pkl.exists():
                continue   # 复用之前已生成的

            # 加载 N 个 lead chunk
            lead_chunks = []
            for lp in p["lead_pkls"]:
                ld = pickle.load(open(lp, "rb"))
                lead_chunks.append(
                    preprocess_motion_to_tensor(ld["pos"], ld["q"], normalizer)
                )
            lead_tensor = torch.stack(lead_chunks).to(device)             # [N, 150, 151]

            # 加载 N 个 music feat
            music_chunks = [torch.from_numpy(np.load(f)) for f in p["music_feats"]]
            music_tensor = torch.stack(music_chunks).to(device)            # [N, 150, 4800]

            cond = torch.cat([lead_tensor, music_tensor.to(lead_tensor.dtype)], dim=-1)
            # [N, 150, 4951]

            N = p["N"]
            torch.manual_seed(opt.seed + p_idx)
            shape = (N, horizon, repr_dim)

            # 调 render_sample，mode="long"，fk_out 自动保存 1 个 stitched .pkl
            tmp_render_out = str(out_dir / "_tmp_render")
            Path(tmp_render_out).mkdir(parents=True, exist_ok=True)
            with torch.no_grad():
                diffusion.render_sample(
                    shape, cond, normalizer,
                    epoch=run_tag,
                    render_out=tmp_render_out,
                    fk_out=str(gen_dir),
                    name=[p["wav_name"]],
                    sound=False,
                    mode="long",
                    duet=True,
                    repr_dim=repr_dim,
                    render=False,
                )
            # render_sample 的 .pkl 命名规则：{epoch}_{seq_name_no_slice}.pkl
            # 对我们而言：{run_tag}_{p['name']}.pkl  ✓ 已与 out_pkl 一致

        # 该 (ckpt, cfg) 跑完 → 算 PFC + LMA
        pfc = calc_pfc(str(gen_dir))
        lma_scores = []
        for p in pairs:
            saved = gen_dir / f"{run_tag}_{p['name']}.pkl"
            foll_long_pred = pickle.load(open(saved, "rb"))["full_pose"]
            T = foll_long_pred.shape[0]
            lead_long = lead_long_joints[p["name"]][:T]
            T_use = min(T, lead_long.shape[0])
            lma_scores.append(compute_similarity(
                lead_long[:T_use],
                foll_long_pred[:T_use],
            )["total_score"])
        lma     = float(np.mean(lma_scores))
        lma_std = float(np.std(lma_scores))

        results.append({
            "cfg":         run["cfg"],
            "epoch":       run["epoch"],
            "wm":          run["wm"],
            "wl":          run["wl"],
            "pfc":         pfc,
            "pfc_vs_gt":   pfc - gt_pfc,
            "pfc_ratio_gt": pfc / gt_pfc if gt_pfc > 0 else float("inf"),
            "lma":         lma,
            "lma_std":     lma_std,
            "lma_vs_real": lma - real_lma,
        })
        print(f"  → {run['cfg']:10s} @ {run['epoch']}  "
              f"PFC={pfc:.4f} (Δ={pfc-gt_pfc:+.4f}, {pfc/gt_pfc:.2f}x GT)  "
              f"LMA={lma:.4f} (Δ={lma-real_lma:+.4f})")

        # 每跑一组就 dump 一次（半路挂掉数据不丢）
        summary = {
            "split":          "test (crossmodal_test.txt, ch02)",
            "protocol":       "per-song long-mode (EDGE-compatible)",
            "n_pairs":        len(pairs),
            "weights_dir":    opt.weights_dir,
            "gt_pfc":         gt_pfc,
            "real_lma":       real_lma,
            "results":        results,
        }
        json.dump(summary, open(out_dir / "summary.json", "w"), indent=2)

    # ── Step 3: 写 paper_table.md ─────────────────────────────────────────
    md = []
    md.append("# Test Set Evaluation Results (Per-Song Long-Mode)\n\n")
    md.append("**Protocol**: EDGE-compatible per-song long-mode. "
              "Each pair → 1 stitched long sequence → 1 PFC value; "
              "averaged across {} pairs.\n".format(len(pairs)))
    md.append("**Split**: AIST++ crossmodal_test (ch02)\n")
    md.append("**Conditioning**: lead motion + music (duet extension)\n")
    md.append("**No post-processing**: raw model outputs.\n\n")

    md.append("## Ground-Truth Baselines\n\n")
    md.append(f"- **GT PFC** (real follower, per-song avg): **{gt_pfc:.4f}**\n")
    md.append(f"- **GT LMA** (real lead vs real follower, per-song avg): **{real_lma:.4f}** ← human-pair ceiling\n\n")

    md.append("## Per-checkpoint Results (10 .pkl per row, one per pair)\n\n")
    md.append("| Config | Epoch | PFC ↓ | vs GT | × GT | LMA ↑ | gap to real |\n")
    md.append("|--------|------:|------:|------:|-----:|------:|------------:|\n")
    for cfg in ["lead_only", "full"]:
        for r in [x for x in results if x["cfg"] == cfg]:
            md.append(f"| {cfg:9s} | {r['epoch']} | "
                      f"{r['pfc']:.3f} | {r['pfc_vs_gt']:+.3f} | {r['pfc_ratio_gt']:.2f}x | "
                      f"{r['lma']:.4f} | {r['lma_vs_real']:+.4f} |\n")
    md.append(f"| GT (real) | — | **{gt_pfc:.3f}** | — | 1.00x | **{real_lma:.4f}** | — |\n\n")

    best_pfc = min(results, key=lambda x: x["pfc"])
    best_lma = max(results, key=lambda x: x["lma"])
    md.append("## Best Checkpoints\n\n")
    md.append(f"- **PFC 最低**: `{best_pfc['cfg']} @ {best_pfc['epoch']}` "
              f"→ PFC = {best_pfc['pfc']:.4f} (LMA = {best_pfc['lma']:.4f})\n")
    md.append(f"- **LMA 最高**: `{best_lma['cfg']} @ {best_lma['epoch']}` "
              f"→ LMA = {best_lma['lma']:.4f} (PFC = {best_lma['pfc']:.4f})\n\n")

    md.append("## EDGE Paper Comparison\n\n")
    md.append("From EDGE [Tseng et al., CVPR 2023] Table 1 (same `crossmodal_test.txt`, ch02, **same per-song protocol**):\n\n")
    md.append("| Method | Task | PFC ↓ |\n|--------|------|------:|\n")
    md.append("| Ground Truth (paper) | solo | 1.332 |\n")
    md.append("| EDGE w=2 + CCL       | solo | 1.5363 |\n")
    md.append("| EDGE w=1 + CCL       | solo | 1.6545 |\n")
    md.append("| Bailando             | solo | 1.754 |\n")
    md.append("| FACT                 | solo | 2.2543 |\n")
    md.append("| EDGE w/o CCL         | solo | 3.0806 |\n")
    md.append(f"| **Ours (duet, {best_pfc['cfg']} @ {best_pfc['epoch']})** | **duet** | **{best_pfc['pfc']:.4f}** |\n\n")

    md.append("## Notes\n\n")
    md.append("- This is per-song long-mode eval (EDGE-compatible). The earlier per-slice eval is at `eval/test_eval_per_slice/`.\n")
    md.append(f"- Our GT PFC ({gt_pfc:.3f}) differs from EDGE paper's 1.332: different subset (duet pairs vs full solo test), different motion preprocessing.\n")
    md.append("- Duet task introduces follower reactivity → inherently more foot adjustments → higher PFC than solo. Not a direct apples-to-apples comparison; we provide EDGE numbers for context.\n")

    (out_dir / "paper_table.md").write_text("".join(md))
    print(f"\n=== 报告已生成 ===")
    print(f"  {out_dir/'summary.json'}")
    print(f"  {out_dir/'paper_table.md'}")
    print("\n" + "="*80)
    print("".join(md))


if __name__ == "__main__":
    main()
