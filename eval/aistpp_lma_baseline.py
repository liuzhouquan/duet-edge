"""
AIST++ LMA Baseline Scorer

Groups raw AIST++ motion files by music ID, pairs different dancers that
performed to the same BGM, computes LMA similarity for every pair, and
prints a structured English log report.

Run from the EDGE project root:
    conda run -n edge-mac python eval/aistpp_lma_baseline.py \
        --data_dir ../aist_plusplus_final/motions

Dependencies: numpy, scipy  (no pytorch3d required)
"""

import argparse
import glob
import os
import pickle
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent.parent))
from eval.lma_features import extract_lma_features

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── SMPL skeleton constants (from EDGE/vis.py) ────────────────────────────
SMPL_PARENTS = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6,
     7, 8, 9, 9, 9,12,13,14,16,17,
    18,19,20,21,
]
SMPL_OFFSETS = np.array([
    [ 0.0,        0.0,        0.0       ],
    [ 0.05858135,-0.08228004,-0.01766408],
    [-0.06030973,-0.09051332,-0.01354254],
    [ 0.00443945, 0.12440352,-0.03838522],
    [ 0.04345142,-0.38646945, 0.008037  ],
    [-0.04325663,-0.38368791,-0.00484304],
    [ 0.00448844, 0.1379564,  0.02682033],
    [-0.01479032,-0.42687458,-0.037428  ],
    [ 0.01905555,-0.4200455, -0.03456167],
    [-0.00226458, 0.05603239, 0.00285505],
    [ 0.04105436,-0.06028581, 0.12204243],
    [-0.03483987,-0.06210566, 0.13032329],
    [-0.0133902,  0.21163553,-0.03346758],
    [ 0.07170245, 0.11399969,-0.01889817],
    [-0.08295366, 0.11247234,-0.02370739],
    [ 0.01011321, 0.08893734, 0.05040987],
    [ 0.12292141, 0.04520509,-0.019046  ],
    [-0.11322832, 0.04685326,-0.00847207],
    [ 0.2553319, -0.01564902,-0.02294649],
    [-0.26012748,-0.01436928,-0.03126873],
    [ 0.26570925, 0.01269811,-0.00737473],
    [-0.26910836, 0.00679372,-0.00602676],
    [ 0.08669055,-0.01063603,-0.01559429],
    [-0.0887537, -0.00865157,-0.01010708],
], dtype=np.float32)


def smpl_fk(trans: np.ndarray, poses_aa: np.ndarray) -> np.ndarray:
    """
    Pure numpy/scipy forward kinematics using the SMPL skeleton.

    Parameters
    ----------
    trans    : [T, 3]  root translations  (AIST++ smpl_trans / smpl_scaling)
    poses_aa : [T, 72] joint axis-angle rotations (smpl_poses)

    Returns
    -------
    joints   : [T, 24, 3]  world-space joint positions (y-up coordinates)
    """
    T   = trans.shape[0]
    n   = len(SMPL_PARENTS)
    rv  = poses_aa.reshape(T, n, 3)         # [T, 24, 3]

    positions = np.zeros((T, n, 3), dtype=np.float64)
    rot_mats  = np.zeros((T, n, 3, 3), dtype=np.float64)

    for j in range(n):
        R_local = Rotation.from_rotvec(rv[:, j]).as_matrix()   # [T,3,3]
        p = SMPL_PARENTS[j]
        if p == -1:
            positions[:, j] = trans
            rot_mats[:, j]  = R_local
        else:
            R_par  = rot_mats[:, p]                              # [T,3,3]
            R_abs  = R_par @ R_local                             # [T,3,3]
            rot_mats[:, j] = R_abs
            world_off = np.einsum('tij,j->ti', R_par, SMPL_OFFSETS[j])
            positions[:, j] = positions[:, p] + world_off

    return positions


def load_joints(pkl_path: str) -> np.ndarray:
    """
    Load an AIST++ motion pkl and return [T, 24, 3] joint positions.
    Normalises root translation by smpl_scaling if present.
    """
    data = pickle.load(open(pkl_path, "rb"))

    if "smpl_poses" in data:
        trans = np.array(data["smpl_trans"], dtype=np.float64)   # [T, 3]
        poses = np.array(data["smpl_poses"], dtype=np.float64)   # [T, 72]
        if "smpl_scaling" in data:
            scale = float(np.array(data["smpl_scaling"]).ravel()[0])
            trans /= scale
        return smpl_fk(trans, poses)

    if "pos" in data and "q" in data:
        trans = np.array(data["pos"], dtype=np.float64)
        poses = np.array(data["q"],   dtype=np.float64)
        return smpl_fk(trans, poses)

    raise ValueError(f"Unrecognised pkl format: {list(data.keys())} in {pkl_path}")


def parse_filename(fname: str):
    """
    gBR_sBM_cAll_d04_mBR1_ch02  ->  (music_id='mBR1', dancer_id='d04', chunk_id='ch02')
    chunk_id is the last field (ch01, ch02, …) — each is a 5-second segment of the
    full performance.  Pairing by the same chunk_id ensures both sequences correspond
    to the same musical window.
    """
    base  = os.path.splitext(os.path.basename(fname))[0]
    parts = base.split("_")
    # Expect at least 6 parts: genre, style, cAll, dancer, music, chunk
    if len(parts) < 6:
        return None, None, None
    dancer_id = parts[3]   # e.g. 'd04'
    music_id  = parts[4]   # e.g. 'mBR1'
    chunk_id  = parts[5]   # e.g. 'ch02'
    return music_id, dancer_id, chunk_id


def pearson_windows(f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
    """Per-window Pearson r; NaN → 0."""
    W = min(f1.shape[0], f2.shape[0])
    rs = np.zeros(W)
    for i in range(W):
        a, b = f1[i], f2[i]
        if a.std() < 1e-8 or b.std() < 1e-8:
            rs[i] = 0.0
        else:
            r, _ = pearsonr(a, b)
            rs[i] = 0.0 if np.isnan(r) else r
    return rs


def score_pair(joints1: np.ndarray, joints2: np.ndarray) -> dict:
    """
    Compute LMA similarity for a (lead, follower) joint pair.
    Returns per-component scores and a weighted total.
    """
    feats1 = extract_lma_features(joints1)
    feats2 = extract_lma_features(joints2)

    weights = {"body": 0.30, "effort": 0.30, "shape": 0.20, "space": 0.20}
    results = {}
    total   = 0.0

    for comp in ["body", "effort", "shape", "space"]:
        r_seq  = pearson_windows(feats1[comp], feats2[comp])
        score  = float(np.clip((r_seq.mean() + 1) / 2, 0, 1))
        results[f"{comp}_score"] = score
        total += weights[comp] * score

    results["total_score"] = total
    return results


# ── main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str,
        default="../aist_plusplus_final/motions",
        help="Directory containing raw AIST++ .pkl motion files",
    )
    parser.add_argument(
        "--max_pairs_per_group", type=int, default=0,
        help="Cap ordered pairs per (music_id, chunk_id) group (0 = no cap)",
    )
    parser.add_argument(
        "--downsample", type=int, default=2,
        help="Temporal downsample factor applied before FK (default 2 → 60fps→30fps)",
    )
    parser.add_argument(
        "--out", type=str, default="",
        help="Optional path to save the report as a .txt file",
    )
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    if not os.path.isdir(data_dir):
        print(f"ERROR: --data_dir not found: {data_dir}")
        sys.exit(1)

    # ── 1. Collect files, group by music_id → chunk_id → {dancer_id: path} ──
    #
    # AIST++ has two recording conditions in the same directory:
    #   sBM  (Style: Basic Movement) — multiple dancers captured simultaneously,
    #        pre-sliced into equal-length chunks ch01-ch10.  Same chunk_id across
    #        dancers IS the same musical time window.  ← use these.
    #   sFM  (Style: Free Movement)  — solo full-length recordings; their ch
    #        numbers are camera/take IDs, NOT sequential time slices.  ← exclude.
    #
    # Only pairing same-chunk_id sBM sequences guarantees time alignment.
    all_files = sorted(glob.glob(os.path.join(data_dir, "*.pkl")))
    if not all_files:
        print(f"ERROR: No .pkl files found in {data_dir}")
        sys.exit(1)

    sbm_files  = [f for f in all_files if "_sBM_" in os.path.basename(f)]
    skip_count = len(all_files) - len(sbm_files)

    # chunk_groups[music_id][chunk_id] = {dancer_id: path}
    chunk_groups: dict = defaultdict(lambda: defaultdict(dict))
    for fpath in sbm_files:
        music_id, dancer_id, chunk_id = parse_filename(fpath)
        if music_id is None:
            skip_count += 1
            continue
        chunk_groups[music_id][chunk_id][dancer_id] = fpath

    # Summary counts
    n_music_ids   = len(chunk_groups)
    n_total_seqs  = sum(
        len(dancers)
        for cmap in chunk_groups.values()
        for dancers in cmap.values()
    )

    lines = []
    def log(s=""):
        print(s)
        lines.append(s)

    # ── 2. Print header ────────────────────────────────────────────────────
    log("=" * 72)
    log("  AIST++ LMA Baseline Similarity Report")
    log("=" * 72)
    log(f"  Data directory : {data_dir}")
    log(f"  Total pkl files: {len(all_files)}  |  sBM used: {len(sbm_files)}  |  sFM excluded: {skip_count}")
    log(f"  Unique music IDs        : {n_music_ids}")
    log(f"  Total sBM seqs indexed  : {n_total_seqs}")
    log(f"  Pairing strategy        : same chunk_id (time-aligned), different dancer_id, sBM only")
    log(f"  Downsample factor       : {args.downsample}  "
        f"(raw 60fps → {60 // args.downsample}fps)")
    log(f"  Max pairs per chunk     : "
        f"{'unlimited' if args.max_pairs_per_group == 0 else args.max_pairs_per_group}")
    log()

    # ── 3. Process pairs ───────────────────────────────────────────────────
    all_results = []
    total_pairs = 0
    t0 = time.time()

    for music_id in sorted(chunk_groups.keys()):
        cmap         = chunk_groups[music_id]   # chunk_id → {dancer_id: path}
        group_scores = []

        for chunk_id in sorted(cmap.keys()):
            dancer_map = cmap[chunk_id]         # {dancer_id: path}
            dancers    = sorted(dancer_map.keys())
            if len(dancers) < 2:
                continue

            # All ordered (lead, follower) pairs where lead ≠ follower
            pairs = [
                (dancers[i], dancers[j])
                for i in range(len(dancers))
                for j in range(len(dancers))
                if i != j
            ]
            if args.max_pairs_per_group > 0:
                pairs = pairs[:args.max_pairs_per_group]

            for lead_id, follower_id in pairs:
                lead_path     = dancer_map[lead_id]
                follower_path = dancer_map[follower_id]
                try:
                    lead_j = load_joints(lead_path)[::args.downsample]
                    foll_j = load_joints(follower_path)[::args.downsample]
                    res    = score_pair(lead_j, foll_j)
                    group_scores.append(res)
                    all_results.append(res)
                    total_pairs += 1
                except Exception as e:
                    log(f"  [WARN] {music_id}/{chunk_id} {lead_id}→{follower_id}: {e}")

        if group_scores:
            g_total  = np.mean([r["total_score"]  for r in group_scores])
            g_body   = np.mean([r["body_score"]   for r in group_scores])
            g_effort = np.mean([r["effort_score"] for r in group_scores])
            g_shape  = np.mean([r["shape_score"]  for r in group_scores])
            g_space  = np.mean([r["space_score"]  for r in group_scores])
            log(f"  {music_id:6s}  n_pairs={len(group_scores):3d}"
                f"  total={g_total:.3f}"
                f"  body={g_body:.3f}  effort={g_effort:.3f}"
                f"  shape={g_shape:.3f}  space={g_space:.3f}")

    elapsed = time.time() - t0

    # ── 4. Aggregate statistics ────────────────────────────────────────────
    log()
    log("=" * 72)
    log("  AGGREGATE STATISTICS")
    log("=" * 72)

    if not all_results:
        log("  No pairs were successfully scored.")
    else:
        comps = ["body", "effort", "shape", "space", "total"]
        header = f"  {'Component':<10}  {'Mean':>6}  {'Std':>6}  "  \
                 f"{'Min':>6}  {'Q1':>6}  {'Median':>7}  {'Q3':>6}  {'Max':>6}"
        log(header)
        log("  " + "-" * 66)

        for comp in comps:
            key    = f"{comp}_score"
            vals   = np.array([r[key] for r in all_results])
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            log(f"  {comp:<10}  {vals.mean():>6.3f}  {vals.std():>6.3f}  "
                f"{vals.min():>6.3f}  {q1:>6.3f}  {med:>7.3f}  "
                f"{q3:>6.3f}  {vals.max():>6.3f}")

        log()
        log(f"  Total pairs scored : {total_pairs}")
        log(f"  Processing time    : {elapsed:.1f}s  "
            f"({elapsed/total_pairs:.2f}s per pair)")
        log()
        log("  Interpretation guide:")
        log("    Score = (mean Pearson r + 1) / 2  mapped to [0, 1]")
        log("    0.5 = uncorrelated (random pair baseline)")
        log("    >0.5 = positively correlated (more similar)")
        log("    <0.5 = negatively correlated (mirror/opposite movement)")
        log()
        log("  Pairing: same 5-second chunk, different dancers performing")
        log("  the same BGM independently — NOT reacting to each other.")
        log("  Use these scores as the reference ceiling for the duet model.")

    log()
    log("=" * 72)

    # ── 5. Save report ─────────────────────────────────────────────────────
    if args.out:
        with open(args.out, "w") as f:
            f.write("\n".join(lines))
        print(f"\nReport saved to {args.out}")


if __name__ == "__main__":
    main()
