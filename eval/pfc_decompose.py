#!/usr/bin/env python3
"""
eval/pfc_decompose.py

把 PFC 公式拆成三个独立因子，分别比较 pred vs GT，
定位 "PFC 高出来的部分到底来自脚速度乘积、还是 root_a 形状、还是两者协同"。

PFC 定义（与 eval/eval_pfc.py 完全一致）：

  fL[i]  = min(v_Lank[i], v_Ltoe[i])        # 左侧最小脚水平速度 (S-2,)
  fR[i]  = min(v_Rank[i], v_Rtoe[i])        # 右侧 (S-2,)
  ra[i]  = ||root_a[i]||₂  with up-dir clamped at 0, then ÷ max(ra)  ∈[0,1]
  pf[i]  = fL[i] · fR[i] · ra[i]
  PFC    = mean_over_slices(mean_over_frames(pf)) × 10000

分解策略：对每个 slice 独立计算 3 个 means
  A = mean(fL · fR)        — 纯脚速度乘积部分（无 ra 调制）
  B = mean(ra)             — 归一化后的 root_a 平均（衡量尖刺度，越平 = 越大）
  C = mean(fL · fR · ra)   — 真实 PFC（÷10000）

再额外算 raw root_a 统计 (未归一化前)，用来分离 "root_a 模式" 和 "max-norm 副作用"。

用法：
    python eval/pfc_decompose.py \
        --pred_dir eval/pfc_val_leadonly_per_slice/epoch_1700 \
        --gt_dir   eval/pfc_val_leadonly_per_slice/ground_truth \
        --label    "lead_w2_ep1700"
"""

import argparse
import glob
import os
import pickle

import numpy as np


DT = 1 / 30
UP_DIR = 2
FLAT_DIRS = [i for i in range(3) if i != UP_DIR]
FOOT_IDX = [7, 10, 8, 11]   # 与 eval_pfc.py 一致：L_ank, L_toe, R_ank, R_toe


def decompose_slice(joint3d: np.ndarray):
    """单个 slice 的因子分解。joint3d: [T, 24, 3], Z-up.

    返回 dict，键值为该 slice 各因子的标量。
    """
    # ── root acceleration（与 eval_pfc.py 公式严格一致）─────────────────
    root_v = (joint3d[1:, 0, :] - joint3d[:-1, 0, :]) / DT
    root_a = (root_v[1:] - root_v[:-1]) / DT
    root_a[:, UP_DIR] = np.maximum(root_a[:, UP_DIR], 0)
    root_a_mag = np.linalg.norm(root_a, axis=-1)            # (S-2,)

    if root_a_mag.max() == 0:
        return None

    ra_max  = float(root_a_mag.max())
    ra_mean = float(root_a_mag.mean())
    ra_norm = root_a_mag / ra_max                            # (S-2,) ∈[0,1]

    # ── foot velocities (horizontal only) ─────────────────────────────
    feet = joint3d[:, FOOT_IDX]                              # [T, 4, 3]
    foot_v = np.linalg.norm(
        feet[2:, :, FLAT_DIRS] - feet[1:-1, :, FLAT_DIRS], axis=-1
    )                                                        # (S-2, 4)
    fL = np.minimum(foot_v[:, 0], foot_v[:, 1])              # left  min
    fR = np.minimum(foot_v[:, 2], foot_v[:, 3])              # right min
    prod = fL * fR                                           # (S-2,)

    return {
        # 真实 PFC slice 值（÷10000）
        "pfc_raw":          float((prod * ra_norm).mean()),
        # 去 ra 调制
        "feet_prod":        float(prod.mean()),
        # 单独看 ra_norm
        "ra_norm_mean":     float(ra_norm.mean()),
        # 脚速度本身
        "fL_mean":          float(fL.mean()),
        "fR_mean":          float(fR.mean()),
        # raw root_a 模式（未归一化）
        "ra_raw_max":       ra_max,
        "ra_raw_mean":      ra_mean,
        "ra_peak_ratio":    ra_max / ra_mean if ra_mean > 0 else float("inf"),
    }


def aggregate(pkl_dir):
    files = sorted(glob.glob(os.path.join(pkl_dir, "*.pkl")))
    rows = []
    for f in files:
        info = pickle.load(open(f, "rb"))
        d = decompose_slice(info["full_pose"])
        if d is None:
            continue
        d["name"] = os.path.basename(f)
        rows.append(d)
    return rows


def mean_over_slices(rows, key):
    return float(np.mean([r[key] for r in rows]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir", required=True)
    p.add_argument("--gt_dir",   required=True)
    p.add_argument("--label",    default="pred",
                   help="给 pred 配置打个标签，方便报告")
    opt = p.parse_args()

    print(f"读 pred: {opt.pred_dir}")
    pred = aggregate(opt.pred_dir)
    print(f"读 gt:   {opt.gt_dir}")
    gt   = aggregate(opt.gt_dir)
    print(f"pred slices = {len(pred)}, gt slices = {len(gt)}")

    # ── 全集均值 ───────────────────────────────────────────────────────
    def show(key):
        gp = mean_over_slices(gt,   key)
        pp = mean_over_slices(pred, key)
        ratio = pp / gp if gp > 0 else float("inf")
        return gp, pp, ratio

    keys = [
        ("PFC (×10000) — 真实",       "pfc_raw",        1e4),
        ("feet_prod = mean(fL·fR)",   "feet_prod",      1.0),
        ("ra_norm_mean",              "ra_norm_mean",   1.0),
        ("fL_mean (左脚 min vel)",    "fL_mean",        1.0),
        ("fR_mean (右脚 min vel)",    "fR_mean",        1.0),
        ("ra_raw_mean (未归一化)",    "ra_raw_mean",    1.0),
        ("ra_raw_max  (尖刺峰值)",    "ra_raw_max",     1.0),
        ("ra_peak_ratio = max/mean",  "ra_peak_ratio",  1.0),
    ]

    print()
    print("=" * 92)
    print(f"{'Factor':<32} {'GT':>14} {'Pred('+opt.label+')':>22} {'Pred/GT':>10}")
    print("-" * 92)
    for name, key, scale in keys:
        gp, pp, ratio = show(key)
        gp_s, pp_s = gp * scale, pp * scale
        print(f"{name:<32} {gp_s:>14.5f} {pp_s:>22.5f} {ratio:>10.3f}x")
    print("=" * 92)

    # ── 反事实归因 ──────────────────────────────────────────────────────
    # 如果把脚速度乘积"等比缩放"到 GT 水平，PFC 会变成多少？
    # 近似假设：在每个 slice 内 fL·fR 与 ra_norm 相对独立。
    gt_feet  = mean_over_slices(gt,   "feet_prod")
    pred_feet= mean_over_slices(pred, "feet_prod")
    gt_pfc   = mean_over_slices(gt,   "pfc_raw")   * 1e4
    pred_pfc = mean_over_slices(pred, "pfc_raw")   * 1e4

    # 因子比
    r_feet = pred_feet / gt_feet if gt_feet > 0 else float("inf")
    r_ra   = mean_over_slices(pred, "ra_norm_mean") / mean_over_slices(gt, "ra_norm_mean")
    r_pfc  = pred_pfc / gt_pfc if gt_pfc > 0 else float("inf")
    # 在独立假设下应有 r_pfc ≈ r_feet × r_ra；偏差 = 协方差贡献
    r_pred_indep = r_feet * r_ra
    cov_gap = r_pfc / r_pred_indep if r_pred_indep > 0 else float("inf")

    print()
    print(f"GT PFC          = {gt_pfc:.4f}")
    print(f"Pred PFC        = {pred_pfc:.4f}")
    print(f"Pred/GT 总比率   = {r_pfc:.3f}x")
    print(f"  分解（独立假设下乘积应等于总比率）:")
    print(f"    脚乘积比率   r_feet = {r_feet:.3f}x")
    print(f"    ra_norm 比率 r_ra   = {r_ra:.3f}x")
    print(f"    乘积         r_feet·r_ra = {r_pred_indep:.3f}x")
    print(f"    协方差残量    r_pfc / r_pred_indep = {cov_gap:.3f}x")
    print()
    print("解读：")
    print("  ▸ 哪个因子比率最大，就是 PFC 偏高的主因。")
    print("  ▸ 协方差残量 >1 意味着 'pred 中脚乘积大的帧 跟 ra 大的帧 更同步'，")
    print("    也就是模型在 root 猛加速时脚没踩稳（CCL 没把 contact 帧打到 v≈0）。")
    print("  ▸ ra_peak_ratio 越小，说明轨迹越平滑、尖刺越少 → ra_norm 平均拉得越高 → 推 PFC 上升。")


if __name__ == "__main__":
    main()
