#!/usr/bin/env python3
"""prepare_sfm_demo.py — preprocess 10 sFM sequences for the ep1800 demo render.

For each genre in {gBR, gHO, gJB, gJS, gKR, gLH, gLO, gMH, gPO, gWA} we pick the
LONGEST sFM sequence (after ignore_list filtering), convert raw AIST++ pkl into
EDGE's {pos, q, scale} format, copy the wav, then run slice + jukebox extract.

Output layout (mirrors data/val/ structure):
  data/sFM_demo/motions/<seq>.pkl
  data/sFM_demo/wavs/<seq>.wav
  data/sFM_demo/motions_sliced/<seq>_slice{N}.pkl
  data/sFM_demo/wavs_sliced/<seq>_slice{N}.wav
  data/sFM_demo/jukebox_feats/<seq>_slice{N}.npy

Pairs JSON:
  data/splits/sFM_demo_pairs.json   (lead == follower = sFM seq itself)
"""

import json
import os
import pickle
import shutil
import sys
from collections import defaultdict
from pathlib import Path


def pick_longest_per_genre(raw_motion_dir, ignore_set):
    by_genre = defaultdict(list)
    for fn in sorted(os.listdir(raw_motion_dir)):
        if "_sFM_" not in fn or not fn.endswith(".pkl"):
            continue
        name = fn[:-4]
        if name in ignore_set:
            continue
        m = pickle.load(open(os.path.join(raw_motion_dir, fn), "rb"))
        n_frames = len(m["smpl_trans"])
        by_genre[name.split("_")[0]].append((name, n_frames))
    chosen = []
    for g in sorted(by_genre):
        seq, n = max(by_genre[g], key=lambda x: x[1])
        chosen.append((g, seq, n))
    return chosen


def convert_raw_to_filtered(raw_pkl, out_pkl):
    """raw AIST++ -> EDGE format {pos, q, scale}. Mirrors filter_split_data.split_data."""
    m = pickle.load(open(raw_pkl, "rb"))
    out = {"pos": m["smpl_trans"], "q": m["smpl_poses"], "scale": m["smpl_scaling"]}
    pickle.dump(out, open(out_pkl, "wb"))


def main():
    edge_root = Path("/data/zliu753/EDGE")
    os.chdir(edge_root / "data")
    sys.path.insert(0, str(edge_root / "data"))

    raw_motion_dir = "edge_aistpp/motions"
    raw_wav_dir = "edge_aistpp/wavs"
    demo_root = Path("sFM_demo")
    demo_motions = demo_root / "motions"
    demo_wavs = demo_root / "wavs"
    demo_motions.mkdir(parents=True, exist_ok=True)
    demo_wavs.mkdir(parents=True, exist_ok=True)

    ignore = set(open("splits/ignore_list.txt").read().split())
    chosen = pick_longest_per_genre(raw_motion_dir, ignore)

    print(f"\n[step 1/4] picked {len(chosen)} sFM sequences (longest per genre)")
    for g, s, n in chosen:
        print(f"  {g}: {s}  ({n} frames, ~{n/60:.1f}s)")

    print(f"\n[step 2/4] converting raw pkl -> {{pos, q, scale}}, copying wavs")
    for _, seq, _ in chosen:
        convert_raw_to_filtered(
            f"{raw_motion_dir}/{seq}.pkl", demo_motions / f"{seq}.pkl"
        )
        shutil.copyfile(f"{raw_wav_dir}/{seq}.wav", demo_wavs / f"{seq}.wav")
    print(f"  wrote {len(chosen)} motion + {len(chosen)} wav files")

    print(f"\n[step 3/4] slicing motion + wav into 5s / 0.5s-stride windows")
    from slice import slice_aistpp
    slice_aistpp(str(demo_motions), str(demo_wavs))
    n_motion = len(list((demo_root / "motions_sliced").glob("*.pkl")))
    n_wav = len(list((demo_root / "wavs_sliced").glob("*.wav")))
    print(f"  produced {n_motion} motion slices + {n_wav} wav slices")

    print(f"\n[step 4/4] extracting jukebox features (GPU, ~5-15 min)")
    from audio_extraction.jukebox_features import extract_folder as jukebox_extract
    jukebox_extract(str(demo_root / "wavs_sliced"), str(demo_root / "jukebox_feats"))
    n_feat = len(list((demo_root / "jukebox_feats").glob("*.npy")))
    print(f"  produced {n_feat} jukebox feature .npy files")

    pairs_json_path = Path("splits/sFM_demo_pairs.json")
    pairs = [{"lead": s, "follower": s} for _, s, _ in chosen]
    pairs_json_path.write_text(json.dumps(pairs, indent=2))
    print(f"\n[done] wrote {pairs_json_path} with {len(pairs)} self-pairs")
    print(f"\n        sFM_demo/ ready for rendering")


if __name__ == "__main__":
    main()
