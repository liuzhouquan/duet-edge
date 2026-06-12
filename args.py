import argparse


def parse_train_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="runs/train", help="project/name")
    parser.add_argument("--exp_name", default="exp", help="save to project/name")
    parser.add_argument("--data_path", type=str, default="data/", help="raw data path")
    parser.add_argument(
        "--processed_data_dir",
        type=str,
        default="data/dataset_backups/",
        help="Dataset backup path",
    )
    parser.add_argument(
        "--render_dir", type=str, default="renders/", help="Sample render path"
    )

    parser.add_argument("--feature_type", type=str, default="jukebox")
    parser.add_argument(
        "--wandb_pj_name", type=str, default="EDGE", help="project name"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--learning_rate", type=float, default=4e-4, help="optimizer learning rate")
    parser.add_argument(
        "--force_reload", action="store_true", help="force reloads the datasets"
    )
    parser.add_argument(
        "--no_cache", action="store_true", help="don't reuse / cache loaded dataset"
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=100,
        help="Save a numbered checkpoint every N epochs (e.g. train-100.pt, train-200.pt)",
    )
    parser.add_argument(
        "--save_latest_interval",
        type=int,
        default=10,
        help="Overwrite latest.pt every N epochs — limits loss to N epochs if job is preempted",
    )
    parser.add_argument("--ema_interval", type=int, default=1, help="ema every x steps")
    parser.add_argument(
        "--checkpoint", type=str, default="",
        help="Checkpoint path to load. For training, resumes from the saved epoch automatically.",
    )
    parser.add_argument(
        "--duet", action="store_true",
        help="双人舞模式：输入=主舞动作+音乐，输出=伴舞动作（需配合 DuetDataset 训练的 checkpoint）",
    )
    parser.add_argument(
        "--drop_prob_music", type=float, default=0.15,
        help=(
            "仅双人模式有效。训练时随机将音乐特征置零的概率。"
            "使模型学会 (∅, lead) 条件下的生成（无音乐推理）。建议值 0.1~0.2。"
        ),
    )
    parser.add_argument(
        "--drop_prob_lead", type=float, default=0.15,
        help=(
            "仅双人模式有效。训练时随机将主舞动作置零的概率。"
            "使模型学会 (music, ∅) 条件下的生成（无主舞推理，退化为单人模式）。建议值 0.1~0.2。"
        ),
    )
    opt = parser.parse_args()
    return opt


def parse_test_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_type", type=str, default="jukebox")
    parser.add_argument("--out_length", type=float, default=30, help="max. length of output, in seconds")
    parser.add_argument(
        "--processed_data_dir",
        type=str,
        default="data/dataset_backups/",
        help="Dataset backup path",
    )
    parser.add_argument(
        "--render_dir", type=str, default="renders/", help="Sample render path"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoint.pt", help="checkpoint"
    )
    parser.add_argument(
        "--music_dir",
        type=str,
        default="data/test/wavs",
        help="folder containing input music",
    )
    parser.add_argument(
        "--save_motions", action="store_true", help="Saves the motions for evaluation"
    )
    parser.add_argument(
        "--motion_save_dir",
        type=str,
        default="eval/motions",
        help="Where to save the motions",
    )
    parser.add_argument(
        "--cache_features",
        action="store_true",
        help="Save the jukebox features for later reuse",
    )
    parser.add_argument(
        "--no_render",
        action="store_true",
        help="Don't render the video",
    )
    parser.add_argument(
        "--use_cached_features",
        action="store_true",
        help="Use precomputed features instead of music folder",
    )
    parser.add_argument(
        "--feature_cache_dir",
        type=str,
        default="cached_features/",
        help="Where to save/load the features",
    )
    # ---- 双人舞推理参数 ----
    parser.add_argument(
        "--duet", action="store_true",
        help="双人舞模式：使用双人 checkpoint，支持多路 CFG 控制",
    )
    parser.add_argument(
        "--lead_motion_dir",
        type=str,
        default="",
        help=(
            "主舞动作的预处理切片目录（每个 .pkl 含 pos[300,3] 和 q[300,72]，60fps）。"
            "不提供或设 --guidance_lead 0 时退化为纯音乐生成模式。"
        ),
    )
    parser.add_argument(
        "--guidance_music", type=float, default=2.0,
        help=(
            "音乐条件的 CFG guidance weight。"
            "0 = 不使用音乐条件；建议值 1.5~3.0。"
        ),
    )
    parser.add_argument(
        "--guidance_lead", type=float, default=2.0,
        help=(
            "主舞动作条件的 CFG guidance weight（仅双人模式有效）。"
            "0 = 不使用主舞条件（退化为单人生成）；建议值 1.5~3.0。"
            "需要训练时使用了 --drop_prob_lead > 0。"
        ),
    )
    opt = parser.parse_args()
    return opt
