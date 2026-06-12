#!/usr/bin/env python3
"""
eval/diagnose_contact.py

诊断模型预测的 contact 标签的可靠性，验证 CCL（Contact Consistency Loss）
在 duet 模式下是否真正起作用。

核心假设
-------
EDGE 原版 (solo): PFC = 1.5363 (with CCL) vs 3.0806 (w/o CCL)
你的 duet best:   PFC = 2.35     ← 接近 "w/o CCL" 范围

假设：duet 模式下模型预测的 contact 几乎全部 < 0.95，导致 CCL 的
``static_idx = model_contact > 0.95`` 几乎全是 False，foot_loss 永远是 0，
CCL 等于失效。

输出
----
1. 文本统计：每只脚的 contact 预测均值、>0.5 比例、>0.95 比例(CCL 阈值)
2. histogram 图：4 个脚的 contact 值分布
3. 时序图（前几个 slice）：predicted contact vs GT contact 对比
4. CCL 命中率：4 个脚 × 所有帧里多少帧 contact > 0.95（CCL 真正生效的帧）
5. 模型脚速度 vs GT 脚速度对比（看模型生成的脚有没有真的"踩稳"）

用法
----
    python eval/diagnose_contact.py \
        --checkpoint runs/train/exp9/weights/train-1700.pt \
        --guidance_music 0.0 --guidance_lead 2.0 \
        --out_dir eval/contact_diag
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
from dataset.quaternion import ax_from_6v
from vis import SMPLSkeleton
from eval.run_val_pfc import slice_index, motion_to_joints


def gt_contact_from_motion(joints_30fps, vel_threshold=0.01):
    """从 FK 后的关节位置算 GT contact (foot velocity < threshold).

    用的是 dataset/dance_dataset.py 训练时算 contact 的同一公式。

    joints_30fps:    [T, 24, 3]
    返回:            [T, 4]  bool/float
    """
    foot_idx = [7, 8, 10, 11]   # 与训练代码一致
    feet = joints_30fps[:, foot_idx]                          # [T, 4, 3]
    feetv = np.zeros((feet.shape[0], 4))
    feetv[:-1] = np.linalg.norm(feet[1:] - feet[:-1], axis=-1)
    return (feetv < vel_threshold).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint",     required=True,
                        help="要诊断的 checkpoint 路径")
    parser.add_argument("--data_dir",       default="data")
    parser.add_argument("--out_dir",        default="eval/contact_diag")
    parser.add_argument("--guidance_music", type=float, default=0.0)
    parser.add_argument("--guidance_lead",  type=float, default=2.0)
    parser.add_argument("--feature_type",   default="jukebox")
    parser.add_argument("--seed",           type=int, default=1234)
    parser.add_argument("--max_slices",     type=int, default=0,
                        help="仅取前 N 个 slice（0 = 全部 93 个）。Quick 诊断时设 10")
    parser.add_argument("--no_plot",        action="store_true",
                        help="跳过 matplotlib 绘图（如服务器无显示）")
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {opt.checkpoint}")
    print(f"Config: w_music={opt.guidance_music}, w_lead={opt.guidance_lead}")
    out_dir = Path(opt.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 数据 + 对齐切片 ──────────────────────────────────────────────────────
    pairs      = json.load(open(os.path.join(opt.data_dir, "splits", "duet_pairs_val.json")))
    val_dir    = os.path.join(opt.data_dir, "val")
    motion_dir = os.path.join(val_dir, "motions_sliced")
    feat_dir   = os.path.join(val_dir, f"{opt.feature_type}_feats")

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
    if opt.max_slices:
        aligned = aligned[:opt.max_slices]
    print(f"将诊断 {len(aligned)} 个 slice")

    # ── 加载模型 ────────────────────────────────────────────────────────────
    from EDGE import EDGE
    model = EDGE(opt.feature_type, opt.checkpoint, duet=True,
                 guidance_weight_music=opt.guidance_music,
                 guidance_weight_lead=opt.guidance_lead)
    model.eval()
    diffusion  = model.diffusion
    normalizer = model.normalizer
    smpl_gpu   = diffusion.smpl
    smpl_cpu   = SMPLSkeleton(device=None)
    repr_dim   = model.repr_dim
    horizon    = model.horizon

    # ── 推理 + 提取 contact 维度 ─────────────────────────────────────────────
    pred_contacts_all = []        # [N, T, 4]  模型预测的 contact (从 151 维输出的前 4 维)
    pred_foot_v_all   = []        # [N, T-1, 4] 模型生成动作的脚速度
    gt_contacts_all   = []        # [N, T, 4]  GT motion 算出的 contact
    gt_foot_v_all     = []        # [N, T-1, 4] GT 动作的脚速度
    names = []

    for idx, item in enumerate(tqdm(aligned, desc="推理 + 提取")):
        # 准备 lead + music cond
        ld = pickle.load(open(item["lead_pkl"], "rb"))
        lead_t = preprocess_motion_to_tensor(ld["pos"], ld["q"], normalizer).unsqueeze(0).to(device)
        music = torch.from_numpy(np.load(item["feat_npy"])).unsqueeze(0).to(device)
        cond = torch.cat([lead_t, music.to(lead_t.dtype)], dim=-1)

        torch.manual_seed(opt.seed + idx)
        with torch.no_grad():
            sample = diffusion.ddim_sample((1, horizon, repr_dim), cond)  # [1, 150, 151]

        # unnormalize 还原到原始尺度
        sample_unn = normalizer.unnormalize(sample.cpu())                  # [1, 150, 151]
        contacts_pred = sample_unn[0, :, :4].numpy()                       # [150, 4]
        pose = sample_unn[0, :, 4:]                                        # [150, 147]
        pos  = pose[:, :3].to(device).unsqueeze(0)                         # [1, 150, 3]
        q    = ax_from_6v(pose[:, 3:].reshape(1, horizon, 24, 6).to(device))
        gen_joints = smpl_gpu.forward(q, pos)[0].detach().cpu().numpy()    # [150, 24, 3]

        # GT 动作
        fd = pickle.load(open(item["follower_pkl"], "rb"))
        gt_joints = motion_to_joints(fd["pos"], fd["q"], smpl_cpu, "cpu")  # [150, 24, 3]

        # 计算 4 个脚的速度（生成 vs GT）
        foot_idx = [7, 8, 10, 11]
        gen_feet = gen_joints[:, foot_idx]
        gen_foot_v = np.linalg.norm(gen_feet[1:] - gen_feet[:-1], axis=-1)  # [149, 4]
        gt_feet = gt_joints[:, foot_idx]
        gt_foot_v  = np.linalg.norm(gt_feet[1:] - gt_feet[:-1], axis=-1)    # [149, 4]

        # GT contact
        gt_contacts = gt_contact_from_motion(gt_joints)                     # [150, 4]

        pred_contacts_all.append(contacts_pred)
        pred_foot_v_all.append(gen_foot_v)
        gt_contacts_all.append(gt_contacts)
        gt_foot_v_all.append(gt_foot_v)
        names.append(item["name"])

    pred_contacts = np.concatenate(pred_contacts_all, axis=0)   # [N*T, 4]
    pred_foot_v   = np.concatenate(pred_foot_v_all, axis=0)     # [N*(T-1), 4]
    gt_contacts   = np.concatenate(gt_contacts_all, axis=0)
    gt_foot_v     = np.concatenate(gt_foot_v_all, axis=0)

    # ── 关键统计 ────────────────────────────────────────────────────────────
    foot_names = ["L_ankle(j7)", "R_ankle(j8)", "L_toe(j10)", "R_toe(j11)"]

    R = []
    R.append("=" * 70)
    R.append(f"Contact Diagnostic Report")
    R.append(f"Checkpoint: {opt.checkpoint}")
    R.append(f"Config:     w_music={opt.guidance_music}  w_lead={opt.guidance_lead}")
    R.append(f"Slices:     {len(aligned)}  (val set)")
    R.append("=" * 70)
    R.append("")

    R.append("## 1. 模型预测的 contact 值分布")
    R.append(f"{'Foot':<14} {'mean':>8} {'>0.5':>10} {'>0.95':>10}  {'(CCL 阈值是 >0.95)':<20}")
    R.append("-" * 70)
    ccl_hit_rate = []
    for k in range(4):
        c = pred_contacts[:, k]
        m   = c.mean()
        p50 = (c > 0.5).mean() * 100
        p95 = (c > 0.95).mean() * 100
        ccl_hit_rate.append(p95)
        R.append(f"{foot_names[k]:<14} {m:>8.3f} {p50:>9.1f}% {p95:>9.1f}%")
    avg_ccl_hit = np.mean(ccl_hit_rate)
    R.append("")

    R.append("## 2. CCL 命中率诊断（关键！）")
    R.append(f"  全部 4 只脚平均：{avg_ccl_hit:.1f}% 的帧 contact > 0.95")
    if avg_ccl_hit < 5:
        R.append("  ⚠️ CCL 几乎从不触发 → CCL 实质失效！")
        R.append("     这解释了为什么 PFC 接近 EDGE 'w/o CCL' (3.08)")
    elif avg_ccl_hit < 20:
        R.append("  ⚠️ CCL 触发率很低")
    else:
        R.append("  ✓ CCL 在合理范围内触发")
    R.append("")

    R.append("## 3. GT contact 命中率参照")
    R.append(f"{'Foot':<14} {'mean':>8} {'>0.5':>10}")
    R.append("-" * 70)
    for k in range(4):
        c = gt_contacts[:, k]
        m   = c.mean()
        p50 = (c > 0.5).mean() * 100
        R.append(f"{foot_names[k]:<14} {m:>8.3f} {p50:>9.1f}%")
    R.append("")

    R.append("## 4. 模型 vs GT 脚速度对比（越接近 GT 越物理合理）")
    R.append(f"{'Foot':<14} {'pred mean':>12} {'GT mean':>12} {'比率':>10}")
    R.append("-" * 70)
    for k in range(4):
        pv = pred_foot_v[:, k].mean()
        gv = gt_foot_v[:, k].mean()
        ratio = pv / gv if gv > 0 else float("inf")
        R.append(f"{foot_names[k]:<14} {pv:>12.4f} {gv:>12.4f} {ratio:>9.2f}x")
    R.append("")
    R.append("  ratio > 1.5x → 模型脚比真人移得快（潜在滑步）")
    R.append("")

    R.append("## 5. 预测 contact vs GT contact 一致性")
    # 用 0.5 阈值二值化
    pc_bin = (pred_contacts > 0.5).astype(int)
    gc_bin = gt_contacts.astype(int)
    R.append(f"{'Foot':<14} {'Precision':>10} {'Recall':>10} {'F1':>8}")
    R.append("-" * 70)
    for k in range(4):
        tp = ((pc_bin[:, k] == 1) & (gc_bin[:, k] == 1)).sum()
        fp = ((pc_bin[:, k] == 1) & (gc_bin[:, k] == 0)).sum()
        fn = ((pc_bin[:, k] == 0) & (gc_bin[:, k] == 1)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2*prec*rec / (prec + rec) if (prec + rec) > 0 else 0.0
        R.append(f"{foot_names[k]:<14} {prec:>10.3f} {rec:>10.3f} {f1:>8.3f}")
    R.append("")

    report_txt = "\n".join(R)
    print(report_txt)
    (out_dir / "report.txt").write_text(report_txt)
    print(f"\n报告写入: {out_dir / 'report.txt'}")

    # 把原始数组也存下来，方便后续调研
    np.savez(out_dir / "raw_data.npz",
             pred_contacts=pred_contacts,
             gt_contacts=gt_contacts,
             pred_foot_v=pred_foot_v,
             gt_foot_v=gt_foot_v,
             names=np.array(names))
    print(f"原始数据写入: {out_dir / 'raw_data.npz'}")

    # ── 图：4 个脚的 contact 值分布 ──────────────────────────────────────────
    if not opt.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 2, figsize=(10, 6))
            for k, ax in enumerate(axes.flat):
                ax.hist(pred_contacts[:, k], bins=50, alpha=0.6, label="Pred", color="C0")
                ax.hist(gt_contacts[:, k]*pred_contacts[:, k].max() + 0.001,
                        bins=50, alpha=0.4, label="GT(scaled)", color="C1")
                ax.axvline(0.95, color="red", linestyle="--", alpha=0.5, label="CCL thresh")
                ax.set_title(f"{foot_names[k]}\n>0.95: {ccl_hit_rate[k]:.1f}%")
                ax.set_xlabel("contact value"); ax.legend(fontsize=8)
            ckpt_name = Path(opt.checkpoint).stem
            fig.suptitle(f"Contact prediction distribution — {ckpt_name}\n"
                         f"(w_m={opt.guidance_music}, w_l={opt.guidance_lead})")
            fig.tight_layout()
            png = out_dir / "contact_histogram.png"
            fig.savefig(png, dpi=120, bbox_inches="tight")
            print(f"直方图: {png}")

            # 时序对比图：前 3 个 slice 的 contact 时序
            n_show = min(3, len(pred_contacts_all))
            fig, axes = plt.subplots(n_show, 4, figsize=(16, 2.5*n_show), sharey=True)
            if n_show == 1:
                axes = axes.reshape(1, -1)
            for i in range(n_show):
                for k in range(4):
                    ax = axes[i, k]
                    ax.plot(pred_contacts_all[i][:, k], label="Pred", lw=1.2)
                    ax.plot(gt_contacts_all[i][:, k]  , label="GT", lw=0.8, alpha=0.7)
                    ax.axhline(0.95, color="red", linestyle="--", alpha=0.3)
                    ax.set_ylim(-0.1, 1.1)
                    if i == 0:
                        ax.set_title(foot_names[k])
                    if k == 0:
                        ax.set_ylabel(names[i][:20], fontsize=8)
                    if i == n_show - 1:
                        ax.set_xlabel("frame")
                    ax.legend(fontsize=7)
            fig.suptitle("Contact prediction over time (first 3 slices)")
            fig.tight_layout()
            png2 = out_dir / "contact_timeseries.png"
            fig.savefig(png2, dpi=120, bbox_inches="tight")
            print(f"时序图: {png2}")
        except Exception as e:
            print(f"绘图失败（不影响数据）: {e}")


if __name__ == "__main__":
    main()
