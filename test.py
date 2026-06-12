import glob
import os
import pickle
from functools import cmp_to_key
from pathlib import Path
from tempfile import TemporaryDirectory
import random

import jukemirlib
import numpy as np
import torch
from tqdm import tqdm

from args import parse_test_opt
from data.slice import slice_audio
from EDGE import EDGE
from data.audio_extraction.baseline_features import extract as baseline_extract
from data.audio_extraction.jukebox_features import extract as juke_extract
from dataset.dance_dataset import preprocess_motion_to_tensor

# sort filenames that look like songname_slice{number}.ext
key_func = lambda x: int(os.path.splitext(x)[0].split("_")[-1].split("slice")[-1])


def stringintcmp_(a, b):
    aa, bb = "".join(a.split("_")[:-1]), "".join(b.split("_")[:-1])
    ka, kb = key_func(a), key_func(b)
    if aa < bb:
        return -1
    if aa > bb:
        return 1
    if ka < kb:
        return -1
    if ka > kb:
        return 1
    return 0


stringintkey = cmp_to_key(stringintcmp_)


def test(opt):
    feature_func = juke_extract if opt.feature_type == "jukebox" else baseline_extract
    sample_length = opt.out_length
    sample_size = int(sample_length / 2.5) - 1

    # ---- 加载主舞动作切片（仅双人舞模式）----
    # --duet 未指定时完全跳过，使用原始单人逻辑（兼容预训练 checkpoint）。
    lead_motion_chunks = None
    if opt.duet and opt.lead_motion_dir:
        # 从 checkpoint 里取出归一化器，和训练时完全一致
        ckpt = torch.load(opt.checkpoint, map_location="cpu")
        normalizer = ckpt["normalizer"]

        # 按 slice 编号排序加载所有主舞切片
        lead_files = sorted(
            glob.glob(os.path.join(opt.lead_motion_dir, "*.pkl")),
            key=stringintkey,
        )
        if len(lead_files) < sample_size:
            raise ValueError(
                f"主舞目录只有 {len(lead_files)} 个切片，"
                f"但生成 {opt.out_length}s 需要 {sample_size} 个。"
                f"请提供更多切片或缩短 --out_length。"
            )

        print(f"Loading lead motion from {opt.lead_motion_dir} ({len(lead_files)} slices)")
        processed = []
        for f in lead_files[:sample_size]:
            data = pickle.load(open(f, "rb"))
            # 每个 pkl 包含 pos[T_raw,3] 和 q[T_raw,72]（60fps 原始帧率）
            tensor = preprocess_motion_to_tensor(data["pos"], data["q"], normalizer)
            processed.append(tensor)
        # [sample_size, 150, 151]
        lead_motion_chunks = torch.stack(processed)

    temp_dir_list = []
    all_cond = []
    all_filenames = []
    if opt.use_cached_features:
        print("Using precomputed features")
        # all subdirectories
        dir_list = glob.glob(os.path.join(opt.feature_cache_dir, "*/"))
        for dir in dir_list:
            file_list = sorted(glob.glob(f"{dir}/*.wav"), key=stringintkey)
            juke_file_list = sorted(glob.glob(f"{dir}/*.npy"), key=stringintkey)
            assert len(file_list) == len(juke_file_list)
            # random chunk after sanity check
            rand_idx = random.randint(0, len(file_list) - sample_size)
            file_list = file_list[rand_idx : rand_idx + sample_size]
            juke_file_list = juke_file_list[rand_idx : rand_idx + sample_size]
            cond_list = [np.load(x) for x in juke_file_list]
            all_filenames.append(file_list)
            all_cond.append(torch.from_numpy(np.array(cond_list)))
    else:
        print("Computing features for input music")
        for wav_file in glob.glob(os.path.join(opt.music_dir, "*.wav")):
            # create temp folder (or use the cache folder if specified)
            if opt.cache_features:
                songname = os.path.splitext(os.path.basename(wav_file))[0]
                save_dir = os.path.join(opt.feature_cache_dir, songname)
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                dirname = save_dir
            else:
                temp_dir = TemporaryDirectory()
                temp_dir_list.append(temp_dir)
                dirname = temp_dir.name
            # slice the audio file
            print(f"Slicing {wav_file}")
            slice_audio(wav_file, 2.5, 5.0, dirname)
            file_list = sorted(glob.glob(f"{dirname}/*.wav"), key=stringintkey)
            # randomly sample a chunk of length at most sample_size
            rand_idx = random.randint(0, len(file_list) - sample_size)
            cond_list = []
            # generate juke representations
            print(f"Computing features for {wav_file}")
            for idx, file in enumerate(tqdm(file_list)):
                # if not caching then only calculate for the interested range
                if (not opt.cache_features) and (not (rand_idx <= idx < rand_idx + sample_size)):
                    continue
                # audio = jukemirlib.load_audio(file)
                # reps = jukemirlib.extract(
                #     audio, layers=[66], downsample_target_rate=30
                # )[66]
                reps, _ = feature_func(file)
                # save reps
                if opt.cache_features:
                    featurename = os.path.splitext(file)[0] + ".npy"
                    np.save(featurename, reps)
                # if in the random range, put it into the list of reps we want
                # to actually use for generation
                if rand_idx <= idx < rand_idx + sample_size:
                    cond_list.append(reps)
            cond_list = torch.from_numpy(np.array(cond_list))
            all_cond.append(cond_list)
            all_filenames.append(file_list[rand_idx : rand_idx + sample_size])

    model = EDGE(
        opt.feature_type,
        opt.checkpoint,
        duet=opt.duet,
        guidance_weight_music=opt.guidance_music,
        guidance_weight_lead=opt.guidance_lead if opt.duet else None,
    )
    model.eval()

    # directory for optionally saving the dances for eval
    fk_out = None
    if opt.save_motions:
        fk_out = opt.motion_save_dir

    print("Generating dances")
    for i in range(len(all_cond)):
        music_cond = all_cond[i]  # [N_chunks, 150, music_feature_dim]

        if lead_motion_chunks is not None:
            # 双人舞模式：cond = cat([主舞动作(151), 音乐特征], dim=-1)
            # → [N_chunks, 150, 151 + music_feature_dim]
            # guidance_music / guidance_lead 控制各路条件的权重，
            # guided_forward 内部会按需将对应部分置零，无需在此手动清零。
            cond = torch.cat(
                [lead_motion_chunks.to(music_cond.dtype), music_cond], dim=-1
            )
        else:
            # 无主舞动作：lead 部分填零，guided_forward 的 (music, ∅) 路径生效
            if opt.duet:
                lead_zeros = torch.zeros(
                    music_cond.shape[0], music_cond.shape[1], 151,
                    dtype=music_cond.dtype,
                )
                cond = torch.cat([lead_zeros, music_cond], dim=-1)
            else:
                # 单人模式（兼容原始 checkpoint）
                cond = music_cond

        data_tuple = None, cond, all_filenames[i]
        model.render_sample(
            data_tuple, "test", opt.render_dir, render_count=-1, fk_out=fk_out, render=not opt.no_render
        )
    print("Done")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    for temp_dir in temp_dir_list:
        temp_dir.cleanup()


if __name__ == "__main__":
    opt = parse_test_opt()
    test(opt)
