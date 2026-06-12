#!/usr/bin/env python3
"""
eval/run_ep1800_cfg_sweep.py

Single-checkpoint CFG-weight sweep at epoch 1800. For each of 15 (mode, weight)
configurations, runs duet inference on all 10 val pairs, computes PFC + the
17-metric LMA report, and renders stitched mp4 videos for the 5 longest pairs.
Per-config English reports plus a master comparison report at the end.

Configurations
--------------
For w ∈ {1, 1.5, 2, 2.5, 3}:
  full_w{w}      : (w_music=w, w_lead=w)   — both streams, symmetric scaling
  lead_w{w}      : (w_music=0, w_lead=w)   — music stream silenced at CFG level
  music_w{w}     : (w_music=w, w_lead=0)   — lead stream silenced at CFG level

Outputs
-------
eval/ep1800_render_eval/
  <cfg_name>/
    report.md                # English per-config report
    per_slice/*.pkl          # generated motions ({full_pose: [150, 24, 3]})
  MASTER_REPORT.md           # 15-config comparison
  summary.json               # all numerics
renders/ep1800_cfg_sweep/
  <cfg_name>/                # 5 stitched mp4 videos per config (longest pairs)

Smoke-test mode
---------------
    python eval/run_ep1800_cfg_sweep.py --smoke
runs 1 config (full_w2.0) × 1 pair × 1 video, used to validate the pipeline
before launching the full ~50-minute SLURM job.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.dance_dataset import preprocess_motion_to_tensor
from dataset.quaternion import ax_from_6v
from eval.eval_pfc import calc_physical_score   # noqa: F401  (kept for reference)
from eval.lma17_features import (
    COMPONENT_OF, METRIC_NAMES, N_METRICS,
    aggregate_paper, extract_17_timeseries,
)
from eval.run_lma17_sweep import _safe_pearson, lma_for_slice
from eval.run_val_pfc import calc_pfc, motion_to_joints, slice_index
from render import _stitch_and_fk, load_lead_chunks, load_music_chunks
from vis import SMPLSkeleton, skeleton_render


# ─── Sweep design ─────────────────────────────────────────────────────────────

WEIGHTS = [1.0, 1.5, 2.0, 2.5, 3.0]


def build_configs() -> List[Dict]:
    cfgs = []
    for w in WEIGHTS:
        cfgs.append({"name": f"full_w{w}",  "mode": "full",  "w_music": w,   "w_lead": w})
        cfgs.append({"name": f"lead_w{w}",  "mode": "lead",  "w_music": 0.0, "w_lead": w})
        cfgs.append({"name": f"music_w{w}", "mode": "music", "w_music": w,   "w_lead": 0.0})
    return cfgs


# Indices into `duet_pairs_val.json` to render videos for (the 5 longest pairs,
# all ≥ 2 stride-2.5s chunks; verified previously).
RENDER_PAIR_INDICES = [0, 3, 4, 8, 9]


# ─── Pair preparation ─────────────────────────────────────────────────────────

def build_aligned_pairs(pairs_json: str, data_dir: str,
                        split_subdir: str = "val",
                        feature_type: str = "jukebox") -> List[Dict]:
    """For each pair, find aligned dense slices and the stride-2.5s subset."""
    pairs = json.load(open(pairs_json))
    motion_dir = os.path.join(data_dir, split_subdir, "motions_sliced")
    feat_dir   = os.path.join(data_dir, split_subdir, f"{feature_type}_feats")
    wav_dir    = os.path.join(data_dir, split_subdir, "wavs_sliced")

    out = []
    for pair_idx, pair in enumerate(pairs):
        lead_map = {slice_index(p): p for p in
                    glob.glob(os.path.join(motion_dir, f"{pair['lead']}_slice*.pkl"))}
        foll_map = {slice_index(p): p for p in
                    glob.glob(os.path.join(motion_dir, f"{pair['follower']}_slice*.pkl"))}
        feat_map = {slice_index(p): p for p in
                    glob.glob(os.path.join(feat_dir, f"{pair['follower']}_slice*.npy"))}
        wav_map = {slice_index(p): p for p in
                   glob.glob(os.path.join(wav_dir, f"{pair['follower']}_slice*.wav"))}

        # Dense slices (0.5s stride) for numerics: every aligned slice.
        dense_common = sorted(set(lead_map) & set(foll_map) & set(feat_map))
        # Stride-2.5s subset for rendering (matches long_ddim_sample's half-overlap).
        stitch_common = [i for i in dense_common if i % 5 == 0
                         and i in wav_map]

        out.append({
            "pair_idx":       pair_idx,
            "lead_name":      pair["lead"],
            "follower_name":  pair["follower"],
            "dense_slices":   dense_common,
            "stitch_slices":  stitch_common,
            "lead_map":       lead_map,
            "foll_map":       foll_map,
            "feat_map":       feat_map,
            "wav_map":        wav_map,
        })
    return out


# ─── Model and FK helpers ─────────────────────────────────────────────────────

def set_guidance(model, w_music: float, w_lead: float):
    """Mutate inference-time CFG weights on the diffusion module."""
    model.diffusion.guidance_weight = w_music
    # guidance_weight_lead=None triggers solo-mode 2-pass CFG; for duet always set scalar.
    model.diffusion.guidance_weight_lead = w_lead


def sample_to_fk(sample: torch.Tensor, normalizer, repr_dim: int, horizon: int,
                 smpl, device) -> np.ndarray:
    """[1, T, 151] (normalised) → [T, 24, 3] world-space joints."""
    sample_unn = normalizer.unnormalize(sample.cpu())
    _, pose = torch.split(sample_unn, (4, repr_dim - 4), dim=2)
    pos = pose[:, :, :3].to(device)
    q   = ax_from_6v(pose[:, :, 3:].reshape(1, horizon, 24, 6).to(device))
    return smpl.forward(q, pos)[0].detach().cpu().numpy()


# ─── Numeric eval per slice ───────────────────────────────────────────────────

def numeric_eval_pair(pair: Dict, gen_dir: Path, smpl_cpu: SMPLSkeleton
                      ) -> Tuple[List[float], List[np.ndarray]]:
    """For one pair, return per-slice (overall_r, per_metric_r) over its dense
    slices. Uses GT lead FK + generated follower FK from gen_dir pkls."""
    overall, per_metric = [], []
    for si in pair["dense_slices"]:
        gen_pkl = gen_dir / f"{pair['follower_name']}_slice{si}.pkl"
        if not gen_pkl.exists():
            continue
        ld = pickle.load(open(pair["lead_map"][si], "rb"))
        lead_joints = motion_to_joints(ld["pos"], ld["q"], smpl_cpu, "cpu")
        fd = pickle.load(open(gen_pkl, "rb"))
        foll_joints = np.asarray(fd["full_pose"])
        ovr, pmr = lma_for_slice(lead_joints, foll_joints)
        overall.append(ovr)
        per_metric.append(pmr)
    return overall, per_metric


def component_means(per_metric: List[float]) -> Dict[str, float]:
    out = {"Body": [], "Effort": [], "Space": []}
    for v, c in zip(per_metric, COMPONENT_OF):
        out[c].append(v)
    return {k: float(np.mean(vs)) if vs else float("nan") for k, vs in out.items()}


# ─── Per-config driver ────────────────────────────────────────────────────────

def run_one_config(cfg: Dict,
                   model,
                   pairs: List[Dict],
                   smpl_gpu,
                   smpl_cpu: SMPLSkeleton,
                   device,
                   normalizer,
                   repr_dim: int,
                   horizon: int,
                   eval_root: Path,
                   render_root: Path,
                   render_pair_idxs: List[int],
                   render_videos: bool,
                   render_subdir: str = "",
                   skip_numerics: bool = False,
                   seed: int = 1234,
                   ) -> Dict:
    """Run inference + numerics (+ optional rendering) for one (mode, weight)."""
    print(f"\n[{cfg['name']}] w_music={cfg['w_music']}, w_lead={cfg['w_lead']}")
    set_guidance(model, cfg["w_music"], cfg["w_lead"])

    cfg_eval_dir   = eval_root / cfg["name"]
    cfg_slice_dir  = cfg_eval_dir / "per_slice"
    cfg_render_dir = render_root / cfg["name"]
    if render_subdir:
        cfg_render_dir = cfg_render_dir / render_subdir
    cfg_slice_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: dense per-slice inference (for numerics) ────────────
    if skip_numerics:
        print(f"  [skip] dense per-slice inference (demo mode)")
    print(f"  [inference] dense slices for {len(pairs)} pairs")
    for pair in tqdm([] if skip_numerics else pairs, desc=f"  {cfg['name']} dense", leave=False):
        for si in pair["dense_slices"]:
            out_pkl = cfg_slice_dir / f"{pair['follower_name']}_slice{si}.pkl"
            if out_pkl.exists():
                continue
            ld = pickle.load(open(pair["lead_map"][si], "rb"))
            lead_t = preprocess_motion_to_tensor(ld["pos"], ld["q"], normalizer
                                                 ).unsqueeze(0).to(device)
            music = torch.from_numpy(np.load(pair["feat_map"][si])).unsqueeze(0).to(device)
            cond = torch.cat([lead_t, music.to(lead_t.dtype)], dim=-1)
            torch.manual_seed(seed + pair["pair_idx"] * 100 + si)
            with torch.no_grad():
                sample = model.diffusion.ddim_sample((1, horizon, repr_dim), cond)
            joints = sample_to_fk(sample, normalizer, repr_dim, horizon, smpl_gpu, device)
            pickle.dump({"full_pose": joints}, open(out_pkl, "wb"))

    # ── Phase 2: PFC ────────────────────────────────────────────────
    if skip_numerics:
        print("  [skip] PFC + LMA + report (demo mode)")
        pfc = float("nan")
    else:
        pfc = calc_pfc(str(cfg_slice_dir))
        print(f"  [pfc] mean PFC = {pfc:.4f}")

    # ── Phase 3: 17-metric LMA per pair ─────────────────────────────
    pair_rows = []
    overall_list, per_metric_list = [], []
    for pair in [] if skip_numerics else pairs:
        slice_overall, slice_pm = numeric_eval_pair(pair, cfg_slice_dir, smpl_cpu)
        if not slice_overall:
            continue
        ov = float(np.mean(slice_overall))
        pm = np.mean(np.stack(slice_pm, axis=0), axis=0)
        overall_list.append(ov)
        per_metric_list.append(pm)
        pair_rows.append({
            "lead":         pair["lead_name"],
            "follower":     pair["follower_name"],
            "n_slices":     len(slice_overall),
            "overall_r":    ov,
            "per_metric_r": pm.tolist(),
        })

    pm_stack = np.stack(per_metric_list, axis=0) if per_metric_list else None
    summary = {
        "name":              cfg["name"],
        "mode":              cfg["mode"],
        "w_music":           cfg["w_music"],
        "w_lead":            cfg["w_lead"],
        "pfc":               float(pfc),
        "n_pairs":           len(overall_list),
        "overall_r_mean":    float(np.mean(overall_list)) if overall_list else float("nan"),
        "overall_r_std":     float(np.std(overall_list))  if overall_list else float("nan"),
        "per_metric_r_mean": pm_stack.mean(axis=0).tolist() if pm_stack is not None else None,
        "per_metric_r_std":  pm_stack.std(axis=0).tolist()  if pm_stack is not None else None,
        "per_pair":          pair_rows,
    }
    if not skip_numerics:
        print(f"  [lma17] overall r = {summary['overall_r_mean']:.4f}, "
              f"Body = {component_means(summary['per_metric_r_mean'])['Body']:.3f}")

    # ── Phase 4: stitched rendering ─────────────────────────────────
    if render_videos:
        cfg_render_dir.mkdir(parents=True, exist_ok=True)
        for pair_idx in render_pair_idxs:
            pair = pairs[pair_idx]
            if len(pair["stitch_slices"]) < 1:
                continue
            common = pair["stitch_slices"]
            n_slices = len(common)
            print(f"  [render] pair {pair_idx} ({pair['follower_name']}), "
                  f"{n_slices} chunks")
            lead_chunks  = load_lead_chunks(pair["lead_map"],  common, normalizer, device)
            music_chunks = load_music_chunks(pair["feat_map"], common, device)
            cond = torch.cat([lead_chunks, music_chunks], dim=-1)
            torch.manual_seed(seed + pair_idx * 7919)  # distinct from per-slice seed
            with torch.no_grad():
                shape = (n_slices, horizon, repr_dim)
                if n_slices == 1:
                    sample = model.diffusion.ddim_sample(shape, cond)
                else:
                    sample = model.diffusion.long_ddim_sample(shape, cond)
            foll_joints = _stitch_and_fk(sample, normalizer, repr_dim, horizon,
                                         smpl_gpu, device,
                                         apply_drift_correction=True)
            lead_joints = _stitch_and_fk(lead_chunks, normalizer, repr_dim, horizon,
                                         smpl_gpu, device,
                                         apply_drift_correction=False)
            wav_paths = [pair["wav_map"][si] for si in common]
            skeleton_render(
                foll_joints,
                epoch=cfg["name"],
                out=str(cfg_render_dir),
                name=wav_paths if n_slices > 1 else wav_paths[0],
                sound=True,
                stitch=(n_slices > 1),
                poses_lead=lead_joints,
            )

    # ── Phase 5: per-config English report ──────────────────────────
    if not skip_numerics:
        write_config_report(cfg_eval_dir / "report.md", cfg, summary)
    return summary


# ─── Report writers ───────────────────────────────────────────────────────────

def fmt(x: float, n: int = 4) -> str:
    return "n/a" if x is None or np.isnan(x) else f"{x:.{n}f}"


def write_config_report(path: Path, cfg: Dict, summary: Dict):
    L = []
    L.append(f"# Inference config: `{cfg['name']}`")
    L.append("")
    L.append(f"- Mode: **{cfg['mode']}**  (full / lead / music)")
    L.append(f"- CFG weights: w_music = **{cfg['w_music']}**,  "
             f"w_lead = **{cfg['w_lead']}**")
    L.append(f"- Checkpoint: `runs/train/exp9/weights/train-1800.pt`")
    L.append(f"- Eval data: val split (10 pairs, AIST++ ch01 sBM)")
    L.append("")

    L.append("## Headline numbers (mean over 10 val pairs)")
    L.append("")
    L.append("| Metric | Value |")
    L.append("|---|---:|")
    L.append(f"| PFC (Physical Foot Contact) ↓ | {fmt(summary['pfc'])} |")
    L.append(f"| LMA 51-dim Pearson r (paper-style) ↑ | "
             f"{fmt(summary['overall_r_mean'])} ± {fmt(summary['overall_r_std'])} |")
    cm = component_means(summary['per_metric_r_mean'])
    L.append(f"| LMA Body component (mean per-metric r) | {fmt(cm['Body'])} |")
    L.append(f"| LMA Effort component (mean per-metric r) | {fmt(cm['Effort'])} |")
    L.append(f"| LMA Space component (mean per-metric r) | {fmt(cm['Space'])} |")
    L.append("")
    L.append("Reference: GT human follower PFC = 1.2607, "
             "EDGE-paper solo PFC = 1.5363, "
             "EDGE solo reproduced on this val = 1.2984.")
    L.append("")

    L.append("## 17-metric LMA Pearson r breakdown (mean ± std across pairs)")
    L.append("")
    L.append("| # | Metric (component) | r mean ± std |")
    L.append("|---|---|---:|")
    for k in range(N_METRICS):
        m = summary['per_metric_r_mean'][k]
        s = summary['per_metric_r_std'][k]
        L.append(f"| {k+1} | `{METRIC_NAMES[k]}` *({COMPONENT_OF[k]})* | "
                 f"{m:.3f} ± {s:.3f} |")
    L.append("")

    L.append("## Per-pair detail")
    L.append("")
    L.append("| Pair (lead → follower) | n_slices | overall r | Body | Effort | Space |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for pr in summary.get("per_pair", []):
        cmp = component_means(pr["per_metric_r"])
        L.append(f"| {pr['lead']} → {pr['follower']} | {pr['n_slices']} | "
                 f"{fmt(pr['overall_r'])} | "
                 f"{fmt(cmp['Body'], 3)} | "
                 f"{fmt(cmp['Effort'], 3)} | "
                 f"{fmt(cmp['Space'], 3)} |")
    L.append("")
    path.write_text("\n".join(L))


def write_master_report(path: Path, summaries: List[Dict], solo_pfc: Optional[float]):
    L = []
    L.append("# Master Comparison Report — ep1800 CFG sweep")
    L.append("")
    L.append("Single checkpoint `runs/train/exp9/weights/train-1800.pt`, "
             "evaluated under 15 inference configurations.")
    L.append("")
    L.append("- 3 inference modes: `full` (both streams), `lead` (music silenced "
             "at CFG level), `music` (lead silenced at CFG level).")
    L.append("- 5 CFG weight points each: w ∈ {1.0, 1.5, 2.0, 2.5, 3.0}.")
    L.append("- Eval data: val split (10 pairs, ch01 sBM).")
    L.append("- Videos rendered for the 5 longest pairs only "
             "(indices 0, 3, 4, 8, 9).")
    L.append("")
    L.append("## Reference points")
    L.append("")
    L.append("| Reference | PFC | LMA (4-comp) |")
    L.append("|---|---:|---:|")
    L.append("| GT (real human duet, val) | 1.2607 | 0.9416 |")
    L.append("| EDGE paper solo (AIST++ test) | 1.5363 | n/a |")
    if solo_pfc is not None:
        L.append(f"| EDGE solo reproduced on our val | {fmt(solo_pfc)} | n/a |")
    L.append("")

    L.append("## Headline comparison")
    L.append("")
    L.append("| Config | w_music | w_lead | PFC ↓ | LMA 51-dim ↑ | Body | Effort | Space |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for s in summaries:
        cm = component_means(s["per_metric_r_mean"])
        L.append(f"| `{s['name']}` | {s['w_music']} | {s['w_lead']} | "
                 f"{fmt(s['pfc'])} | {fmt(s['overall_r_mean'])} | "
                 f"{fmt(cm['Body'])} | {fmt(cm['Effort'])} | {fmt(cm['Space'])} |")
    L.append("")

    L.append("## Per-metric Pearson r across configs (mean across val pairs)")
    L.append("")
    cells = ["# | Metric (component)"] + [f"`{s['name']}`" for s in summaries]
    L.append("| " + " | ".join(cells) + " |")
    L.append("|" + "|".join(["---"] * len(cells)) + "|")
    for k in range(N_METRICS):
        row = [f"{k+1} | `{METRIC_NAMES[k]}` *({COMPONENT_OF[k]})*"]
        for s in summaries:
            row.append(f"{s['per_metric_r_mean'][k]:.3f}")
        L.append("| " + " | ".join(row) + " |")
    L.append("")
    path.write_text("\n".join(L))


# ─── Driver ───────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint",
                   default="runs/train/exp9/weights/train-1800.pt")
    p.add_argument("--data_dir",      default="data")
    p.add_argument("--feature_type",  default="jukebox")
    p.add_argument("--eval_root",     default="eval/ep1800_render_eval")
    p.add_argument("--render_root",   default="renders/ep1800_cfg_sweep")
    p.add_argument("--seed",          type=int, default=1234)
    p.add_argument("--smoke",         action="store_true",
                   help="Run 1 config × 1 pair × 1 render (pipeline sanity test)")
    p.add_argument("--no_render",     action="store_true",
                   help="Skip video rendering (numerics only)")
    p.add_argument("--cfg_filter",    default="",
                   help="Run only these config names (comma-separated)")
    p.add_argument("--render_subdir", default="",
                   help="Put videos under render_root/<cfg>/<subdir>/")
    p.add_argument("--render_indices", default="",
                   help="Comma-separated val-pair indices to render "
                        "(default: 5 longest)")
    p.add_argument("--split_subdir", default="val",
                   help="Data subdir under data_dir/ (val, test, sFM_demo)")
    p.add_argument("--pairs_json", default="",
                   help="Custom pairs JSON path (overrides default val pairs)")
    p.add_argument("--skip_numerics", action="store_true",
                   help="Skip PFC + LMA + per-config report (demo mode)")
    p.add_argument("--include_uncond", action="store_true",
                   help="Append a pure-unconditional 'w0' config "
                        "(w_music=w_lead=0) as an ablation control")
    opt = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device     = {device}")
    print(f"[setup] checkpoint = {opt.checkpoint}")

    # ── Load model once ──────────────────────────────────────────────
    from EDGE import EDGE
    print("[setup] loading EDGE model (duet mode)")
    model = EDGE(opt.feature_type, opt.checkpoint, duet=True)
    model.eval()
    normalizer = model.normalizer
    repr_dim   = model.repr_dim
    horizon    = model.horizon
    smpl_gpu   = model.diffusion.smpl
    smpl_cpu   = SMPLSkeleton(device=None)

    # ── Configs + pairs ──────────────────────────────────────────────
    cfgs = build_configs()
    if opt.include_uncond:
        # Pure unconditional control: both guidance streams off (w_music=w_lead=0).
        # The 3-pass CFG collapses to eps = eps_unc, so the follower is sampled
        # from the model's learned unconditional dance prior — NOT coordinated
        # with the lead or synced to the music. Useful as an ablation baseline.
        cfgs.append({"name": "w0", "mode": "uncond", "w_music": 0.0, "w_lead": 0.0})
    if opt.smoke:
        cfgs = [c for c in cfgs if c["name"] == "full_w2.0"]
    if opt.cfg_filter:
        wanted = set(opt.cfg_filter.split(","))
        cfgs = [c for c in cfgs if c["name"] in wanted]
    print(f"[setup] {len(cfgs)} config(s) to run: "
          f"{', '.join(c['name'] for c in cfgs)}")

    pairs_json = opt.pairs_json or os.path.join(
        opt.data_dir, "splits", "duet_pairs_val.json")
    pairs = build_aligned_pairs(pairs_json, opt.data_dir,
                                opt.split_subdir, opt.feature_type)
    print(f"[setup] {len(pairs)} val pairs, dense slice counts: "
          f"{[len(p['dense_slices']) for p in pairs]}")
    if opt.smoke:
        pairs = pairs[:1]
        print(f"[smoke] reduced to 1 pair: {pairs[0]['follower_name']}")

    if opt.render_indices:
        render_idxs = [int(x) for x in opt.render_indices.split(",") if x.strip()]
    else:
        render_idxs = [0] if opt.smoke else RENDER_PAIR_INDICES
    render_videos = (not opt.no_render)

    eval_root   = Path(opt.eval_root)
    render_root = Path(opt.render_root)
    eval_root.mkdir(parents=True, exist_ok=True)
    if render_videos:
        render_root.mkdir(parents=True, exist_ok=True)

    # ── Sweep ────────────────────────────────────────────────────────
    summaries = []
    t0 = time.time()
    for cfg in cfgs:
        s = run_one_config(cfg, model, pairs, smpl_gpu, smpl_cpu, device,
                           normalizer, repr_dim, horizon,
                           eval_root, render_root, render_idxs,
                           render_videos and (not opt.smoke or cfg["name"] == "full_w2.0"),
                           render_subdir=opt.render_subdir,
                           skip_numerics=opt.skip_numerics,
                           seed=opt.seed)
        summaries.append(s)
        print(f"[{cfg['name']}] done, elapsed {time.time() - t0:.1f}s")

    # ── Master report + summary.json ─────────────────────────────────
    solo_pfc = None
    solo_path = Path("eval/solo_pfc_baseline/summary.json")
    if solo_path.exists():
        solo_pfc = json.load(open(solo_path)).get("edge_solo_pfc_repro")

    if not opt.skip_numerics:
        write_master_report(eval_root / "MASTER_REPORT.md", summaries, solo_pfc)
        json.dump({"configs": summaries,
                   "metric_names": METRIC_NAMES,
                   "component_of": COMPONENT_OF,
                   "solo_pfc_reference": solo_pfc},
                  open(eval_root / "summary.json", "w"), indent=2)

    print(f"\n[done] {len(summaries)} configs, total elapsed "
          f"{time.time() - t0:.1f}s")
    print(f"        reports -> {eval_root}")
    if render_videos:
        print(f"        videos  -> {render_root}")


if __name__ == "__main__":
    main()
