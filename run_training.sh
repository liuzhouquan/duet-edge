#!/bin/bash
#SBATCH --job-name=edge-train
#SBATCH --time=04:00:00
#SBATCH --open-mode=append
#SBATCH --output=logs/train-output.log
#SBATCH --error=logs/train-error.log
#SBATCH --gres=gpu:1

cd /data/zliu753/EDGE

# /home is ephemeral on this cluster; init_env.sh remaps HOME → /data/zliu753
# and sources conda, so all config files persist across sessions.
source /data/zliu753/init_env.sh
conda activate edge

mkdir -p logs

export WANDB_MODE=disabled

# ── 模式说明 ──────────────────────────────────────────────────────────────────
#
# 【单人模式】原始 EDGE 逻辑：输入=音乐特征(4800维)，输出=单人舞动作
#   条件维度: cond_feature_dim=4800
#   数据集:   AISTPPDataset（data/train/ 和 data/test/）
#   权重:     可从官方预训练 checkpoint.pt 继续训练
#
# 【双人模式】扩展逻辑：输入=主舞动作(151维)+音乐特征(4800维)，输出=伴舞动作
#   条件维度: cond_feature_dim=4951
#   数据集:   DuetDataset（同一 music_id 下不同舞者配对）
#   权重:     与单人模式不兼容，需要单独训练
#             可用 --checkpoint 指定原始权重做 fine-tune（cond_projection 层会重新初始化）
#
# 注意：两种模式的 checkpoint 不能互换加载，因为 cond_projection 层维度不同。
# ─────────────────────────────────────────────────────────────────────────────

# ── 断点续训说明 ──────────────────────────────────────────────────────────────
#
# 训练过程中会保存两类文件（均在 runs/train/exp/weights/ 下）：
#   latest.pt       — 每 10 epoch 覆盖一次，job 被中止时最多损失 10 个 epoch
#   train-100.pt    — 每 100 epoch 保存一次永久存档，用于回滚或分析
#
# 断点续训：只需在 --checkpoint 指向最近的 latest.pt 或 train-N.pt，
#           训练会自动从上次保存的 epoch+1 继续，无需手动指定 epoch。
#
# ─────────────────────────────────────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════════════════════
# 【当前激活】双人模式 baseline 验证训练
#   目的：用轻量 baseline 特征（35维）验证 duet fine-tune 流程是否跑通
#   前提：run_prepare_data.sh 已完成（data/train/baseline_feats/ 存在）
#   完成后换成 jukebox 正式训练
# ══════════════════════════════════════════════════════════════════════════════
accelerate launch train.py \
  --batch_size 64 \
  --epochs 200 \
  --feature_type baseline \
  --learning_rate 0.0002 \
  --duet \
  --drop_prob_music 0.15 \
  --drop_prob_lead 0.15 \
  --save_latest_interval 10 \
  --save_interval 50 \
  --checkpoint checkpoint.pt

# ── 双人 baseline 断点续训 ──────────────────────────────────────────────────
# accelerate launch train.py \
#   --batch_size 64 \
#   --epochs 200 \
#   --feature_type baseline \
#   --learning_rate 0.0002 \
#   --duet \
#   --drop_prob_music 0.15 \
#   --drop_prob_lead 0.15 \
#   --save_latest_interval 10 \
#   --save_interval 50 \
#   --checkpoint runs/train/exp/weights/latest.pt

# ── 双人模式正式训练（jukebox，验证通过后用）─────────────────────────────
# accelerate launch train.py \
#   --batch_size 128 \
#   --epochs 2000 \
#   --feature_type jukebox \
#   --learning_rate 0.0002 \
#   --duet \
#   --drop_prob_music 0.15 \
#   --drop_prob_lead 0.15 \
#   --save_latest_interval 10 \
#   --save_interval 100 \
#   --checkpoint checkpoint.pt

# ── 双人模式正式训练断点续训 ────────────────────────────────────────────────
# accelerate launch train.py \
#   --batch_size 128 \
#   --epochs 2000 \
#   --feature_type jukebox \
#   --learning_rate 0.0002 \
#   --duet \
#   --drop_prob_music 0.15 \
#   --drop_prob_lead 0.15 \
#   --save_latest_interval 10 \
#   --save_interval 100 \
#   --checkpoint runs/train/exp/weights/latest.pt

# ── 单人模式训练（兼容官方预训练权重，不用 --duet）────────────────────────
# accelerate launch train.py \
#   --batch_size 128 \
#   --epochs 2000 \
#   --feature_type jukebox \
#   --learning_rate 0.0002 \
#   --save_latest_interval 10 \
#   --save_interval 100
