#!/usr/bin/env python3
"""
eval/run_lma17_sweep.py — 17-metric LMA evaluation across epochs and configs.

Faithfully reproduces the methodology of Zhou et al. IUI '25 Section 4.2
on this duet codebase. Reuses the already-generated per-slice follower pkls
in `eval/pfc_val_per_slice/` and `eval/pfc_val_leadonly_per_slice/` — no
model inference is run here. The lead FK is computed on the fly from
`data/val/motions_sliced/` and cached in memory.

Outputs
-------
A single Markdown report containing:
  1. Overall paper-style 51-dim Pearson r per config, per epoch
  2. Per-metric (17) time-series Pearson r, mean ± std across all pairs
  3. Component-grouped scores (Body / Effort / Space)
  4. Per-pair breakdown for the best checkpoint of each config

Also writes a sibling JSON file with the raw numbers for downstream plotting.

Usage
-----
    python eval/run_lma17_sweep.py \\
        --out_dir eval/lma17_report

Optional: limit which epochs to sweep with --epochs 1700,1800,1900,2000
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.lma17_features import (
    COMPONENT_OF, METRIC_NAMES, N_METRICS,
    aggregate_paper, extract_17_timeseries,
)
from eval.run_val_pfc import motion_to_joints, slice_index
from vis import SMPLSkeleton


# ---------------------------------------------------------------------------
# Pearson helpers
# ---------------------------------------------------------------------------

def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r with std-degeneracy guard. Returns 0.0 instead of NaN."""
    if a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0
    r, _ = pearsonr(a, b)
    return 0.0 if np.isnan(r) else float(r)


def lma_for_slice(lead_joints: np.ndarray,
                  foll_joints: np.ndarray) -> Tuple[float, np.ndarray]:
    """Compute paper-style 51-dim r AND per-metric time-series r for one slice.

    Returns
    -------
    overall_r : float
        Pearson r between the two 51-dim (= 17 metrics × {max, mean, std})
        segment descriptors. This is the paper's headline LMA similarity.
    per_metric_r : np.ndarray [17]
        Pearson r between the per-frame trajectories of each individual
        metric. This is the appendix-style breakdown.
    """
    ts_lead = extract_17_timeseries(lead_joints)   # [T, 17]
    ts_foll = extract_17_timeseries(foll_joints)

    # Paper-style: 51-dim segment descriptor
    desc_lead = aggregate_paper(ts_lead).reshape(-1)   # 17 * 3 = 51
    desc_foll = aggregate_paper(ts_foll).reshape(-1)
    overall_r = _safe_pearson(desc_lead, desc_foll)

    # Per-metric time-series r — interpretable per-metric similarity
    per_metric_r = np.zeros(N_METRICS)
    for k in range(N_METRICS):
        per_metric_r[k] = _safe_pearson(ts_lead[:, k], ts_foll[:, k])

    return overall_r, per_metric_r


# ---------------------------------------------------------------------------
# Lead FK cache and slice indexing
# ---------------------------------------------------------------------------

def build_aligned_slices(pairs: list,
                         motion_dir: str) -> List[Dict]:
    """One row per (pair, slice) with both lead and follower pkl paths."""
    aligned = []
    for pair in pairs:
        lead_map = {slice_index(p): p for p in
                    glob.glob(os.path.join(motion_dir,
                                           f"{pair['lead']}_slice*.pkl"))}
        foll_map = {slice_index(p): p for p in
                    glob.glob(os.path.join(motion_dir,
                                           f"{pair['follower']}_slice*.pkl"))}
        for si in sorted(set(lead_map) & set(foll_map)):
            aligned.append({
                "pair_lead":     pair["lead"],
                "pair_follower": pair["follower"],
                "slice_idx":     si,
                "lead_pkl":      lead_map[si],
                "follower_pkl": foll_map[si],
                "gen_name":      f"{pair['follower']}_slice{si}.pkl",
            })
    return aligned


def precompute_joints(aligned: List[Dict], smpl: SMPLSkeleton,
                       verbose: bool = True) -> Dict[str, np.ndarray]:
    """FK both lead and follower GT joints once, keyed by gen_name."""
    cache = {}
    it = tqdm(aligned, desc="FK lead+followerGT") if verbose else aligned
    for row in it:
        key = row["gen_name"]
        ld = pickle.load(open(row["lead_pkl"], "rb"))
        fd = pickle.load(open(row["follower_pkl"], "rb"))
        cache[("lead", key)] = motion_to_joints(ld["pos"], ld["q"], smpl, "cpu")
        cache[("foll_gt", key)] = motion_to_joints(fd["pos"], fd["q"], smpl, "cpu")
    return cache


# ---------------------------------------------------------------------------
# Per-config evaluation
# ---------------------------------------------------------------------------

def evaluate_dir(follower_dir: str,
                 aligned: List[Dict],
                 lead_cache: Dict,
                 source: str = "generated",
                 ) -> Dict:
    """Compute LMA stats for follower pkls in `follower_dir`.

    `source` is 'generated' (read full_pose from pkl) or 'gt' (use cached GT
    FK joints for the follower).
    """
    per_pair: Dict[str, Dict] = {}

    for row in aligned:
        key = row["gen_name"]
        lead_joints = lead_cache[("lead", key)]

        if source == "gt":
            foll_joints = lead_cache[("foll_gt", key)]
        else:
            gen_pkl = os.path.join(follower_dir, key)
            if not os.path.exists(gen_pkl):
                continue
            fd = pickle.load(open(gen_pkl, "rb"))
            foll_joints = np.asarray(fd["full_pose"])

        ovr, pmr = lma_for_slice(lead_joints, foll_joints)

        pair_key = f"{row['pair_lead']}__{row['pair_follower']}"
        d = per_pair.setdefault(pair_key, {"overall": [], "per_metric": []})
        d["overall"].append(ovr)
        d["per_metric"].append(pmr)

    # Aggregate per pair, then over pairs
    pair_rows = []
    overall_means, pm_means = [], []
    for k, v in per_pair.items():
        if not v["overall"]:
            continue
        p_overall = float(np.mean(v["overall"]))
        p_per_metric = np.mean(np.stack(v["per_metric"], axis=0), axis=0)
        pair_rows.append({
            "pair":          k,
            "n_slices":      len(v["overall"]),
            "overall_r":     p_overall,
            "per_metric_r": p_per_metric.tolist(),
        })
        overall_means.append(p_overall)
        pm_means.append(p_per_metric)

    if not overall_means:
        return {"n_pairs": 0}

    pm_stack = np.stack(pm_means, axis=0)
    return {
        "n_pairs":            len(overall_means),
        "overall_r_mean":     float(np.mean(overall_means)),
        "overall_r_std":      float(np.std(overall_means)),
        "per_metric_r_mean":  pm_stack.mean(axis=0).tolist(),
        "per_metric_r_std":   pm_stack.std(axis=0).tolist(),
        "per_pair":           pair_rows,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def fmt_pct(x: float) -> str:
    return f"{x:.4f}" if not (np.isnan(x) or x is None) else "n/a"


def component_means(per_metric: List[float]) -> Dict[str, float]:
    """Group mean per body/effort/space."""
    out = {"Body": [], "Effort": [], "Space": []}
    for v, c in zip(per_metric, COMPONENT_OF):
        out[c].append(v)
    return {k: float(np.mean(vs)) if vs else float("nan")
            for k, vs in out.items()}


def write_report(results: Dict[str, Dict],
                 out_md: str,
                 out_json: str) -> None:
    """Write the full Markdown appendix report + JSON dump."""

    lines: List[str] = []
    lines.append("# 17-Metric LMA Evaluation Report")
    lines.append("")
    lines.append("Following Zhou et al., *Real-Time Full-body Interaction with "
                 "AI Dance Models* (IUI '25, Section 4.2). The 17 metrics "
                 "comprise Body (8) + Effort (7) + Space (2). Shape (10 "
                 "features) is excluded per the paper's stated methodology.")
    lines.append("")
    lines.append("Each segment (one 5 s / 150-frame slice at 30 fps) is "
                 "summarised by a 51-dim descriptor (17 metrics × {max, mean, "
                 "std}). The headline similarity is the Pearson r between the "
                 "lead's and follower's 51-dim descriptors. Per-metric scores "
                 "are Pearson r between the two per-frame metric "
                 "trajectories.")
    lines.append("")

    # ── Section 1: headline summary across configs ─────────────────
    lines.append("## 1. Headline 51-dim LMA Similarity")
    lines.append("")
    lines.append("| Config | Epoch | n_pairs | r mean | r std |")
    lines.append("|---|---:|---:|---:|---:|")
    for label, res in results.items():
        if "overall_r_mean" not in res:
            continue
        epoch = res.get("epoch", "—")
        lines.append(f"| {res['config']} | {epoch} | "
                     f"{res['n_pairs']} | "
                     f"{fmt_pct(res['overall_r_mean'])} | "
                     f"{fmt_pct(res['overall_r_std'])} |")
    lines.append("")

    # ── Section 2: per-metric breakdown — restricted to comparable configs
    lines.append("## 2. Per-Metric Pearson r (mean ± std across val pairs)")
    lines.append("")
    lines.append("To keep this section readable, the breakdown is shown only "
                 "for GT and the *best* (highest headline r) checkpoint of "
                 "each configuration. Full per-epoch numbers are in the "
                 "companion JSON.")
    lines.append("")

    # Find best per config family
    fam_to_best: Dict[str, str] = {}
    for key, res in results.items():
        if "overall_r_mean" not in res:
            continue
        fam = res["config"]
        if (fam not in fam_to_best
                or res["overall_r_mean"] > results[fam_to_best[fam]]["overall_r_mean"]):
            fam_to_best[fam] = key

    showcase_keys = ([k for k in ["gt"] if k in results]
                     + [v for k, v in fam_to_best.items() if k != "GT"])
    valid = [(k, results[k]) for k in showcase_keys
             if "per_metric_r_mean" in results[k]]

    if valid:
        header_cells = ["# | Metric (component)"] + [v["label"] for _, v in valid]
        lines.append("| " + " | ".join(header_cells) + " |")
        lines.append("|" + "|".join(["---"] * (len(header_cells))) + "|")
        for k in range(N_METRICS):
            row = [f"{k+1} | `{METRIC_NAMES[k]}` *({COMPONENT_OF[k]})*"]
            for _, v in valid:
                m = v["per_metric_r_mean"][k]
                s = v["per_metric_r_std"][k]
                row.append(f"{m:.3f} ± {s:.3f}")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

        lines.append("### 2a. Component means (Body / Effort / Space)")
        lines.append("")
        lines.append("| Config | Body | Effort | Space |")
        lines.append("|---|---:|---:|---:|")
        for _, v in valid:
            cm = component_means(v["per_metric_r_mean"])
            lines.append(f"| {v['label']} | "
                         f"{fmt_pct(cm['Body'])} | "
                         f"{fmt_pct(cm['Effort'])} | "
                         f"{fmt_pct(cm['Space'])} |")
        lines.append("")

    # ── Section 3: per-pair breakdown for top configs ──────────────
    lines.append("## 3. Per-Pair Breakdown (best checkpoint of each config)")
    lines.append("")

    # Pick best (highest overall r) per config family
    family_best: Dict[str, str] = {}
    for key, res in results.items():
        if "overall_r_mean" not in res:
            continue
        fam = res["config"]
        if fam not in family_best or \
                res["overall_r_mean"] > results[family_best[fam]]["overall_r_mean"]:
            family_best[fam] = key

    for fam, key in family_best.items():
        res = results[key]
        lines.append(f"### {res['label']}  (Epoch = {res.get('epoch', '—')}, "
                     f"overall r = {fmt_pct(res['overall_r_mean'])})")
        lines.append("")
        lines.append("| Pair | n_slices | overall r | Body mean | Effort mean | Space mean |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for pr in res.get("per_pair", []):
            cm = component_means(pr["per_metric_r"])
            lines.append(f"| {pr['pair']} | {pr['n_slices']} | "
                         f"{fmt_pct(pr['overall_r'])} | "
                         f"{fmt_pct(cm['Body'])} | "
                         f"{fmt_pct(cm['Effort'])} | "
                         f"{fmt_pct(cm['Space'])} |")
        lines.append("")

    # ── Footer ─────────────────────────────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("Generated by `eval/run_lma17_sweep.py`. Raw numbers in the "
                 "companion JSON.")

    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(out_md).write_text("\n".join(lines))

    # JSON: trim ordering, keep everything
    payload = {
        "metric_names": METRIC_NAMES,
        "component_of": COMPONENT_OF,
        "results":      results,
    }
    Path(out_json).write_text(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def discover_epochs(weights_root: str) -> List[int]:
    epochs = []
    for p in glob.glob(os.path.join(weights_root, "epoch_*")):
        name = os.path.basename(p)
        if name.startswith("epoch_"):
            try:
                epochs.append(int(name[6:]))
            except ValueError:
                pass
    return sorted(epochs)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir",       default="data")
    parser.add_argument("--pairs_json",
                        default="data/splits/duet_pairs_val.json")
    parser.add_argument("--full_sweep_dir",
                        default="eval/pfc_val_per_slice",
                        help="Per-slice pkls from full config sweep")
    parser.add_argument("--lead_sweep_dir",
                        default="eval/pfc_val_leadonly_per_slice",
                        help="Per-slice pkls from lead_only config sweep")
    parser.add_argument("--music_sweep_dir",
                        default="eval/pfc_val_music_only",
                        help="Per-slice pkls from music_only config sweep")
    parser.add_argument("--epochs", default="",
                        help="Comma-separated epoch list; empty = all "
                             "available")
    parser.add_argument("--out_dir", default="eval/lma17_report")
    parser.add_argument("--no_full",      action="store_true")
    parser.add_argument("--no_leadonly",  action="store_true")
    parser.add_argument("--no_musiconly", action="store_true")
    parser.add_argument("--no_gt",        action="store_true")
    opt = parser.parse_args()

    pairs = json.load(open(opt.pairs_json))
    motion_dir = os.path.join(opt.data_dir, "val", "motions_sliced")
    aligned = build_aligned_slices(pairs, motion_dir)
    print(f"[setup] {len(pairs)} pairs, {len(aligned)} aligned slices")

    smpl = SMPLSkeleton(device=None)
    lead_cache = precompute_joints(aligned, smpl, verbose=True)
    print(f"[setup] FK cache size = {len(lead_cache)} entries")

    epoch_filter: Optional[List[int]] = None
    if opt.epochs:
        epoch_filter = [int(x) for x in opt.epochs.split(",") if x.strip()]

    results: Dict[str, Dict] = {}

    # GT baseline
    if not opt.no_gt:
        print("[gt] evaluating ground-truth follower (real human duet)")
        res = evaluate_dir("", aligned, lead_cache, source="gt")
        res.update(config="GT", label="GT (real follower)", epoch=None)
        results["gt"] = res

    # full sweep
    if not opt.no_full:
        epochs = discover_epochs(opt.full_sweep_dir)
        if epoch_filter is not None:
            epochs = [e for e in epochs if e in epoch_filter]
        for ep in tqdm(epochs, desc="full sweep"):
            fdir = os.path.join(opt.full_sweep_dir, f"epoch_{ep:04d}")
            res = evaluate_dir(fdir, aligned, lead_cache, source="generated")
            if res.get("n_pairs", 0) == 0:
                continue
            res.update(config="full",
                       label=f"full @ ep{ep}",
                       epoch=ep)
            results[f"full_{ep:04d}"] = res

    # lead_only sweep
    if not opt.no_leadonly:
        epochs = discover_epochs(opt.lead_sweep_dir)
        if epoch_filter is not None:
            epochs = [e for e in epochs if e in epoch_filter]
        for ep in tqdm(epochs, desc="lead_only sweep"):
            fdir = os.path.join(opt.lead_sweep_dir, f"epoch_{ep:04d}")
            res = evaluate_dir(fdir, aligned, lead_cache, source="generated")
            if res.get("n_pairs", 0) == 0:
                continue
            res.update(config="lead_only",
                       label=f"lead_only @ ep{ep}",
                       epoch=ep)
            results[f"leadonly_{ep:04d}"] = res

    # music_only sweep
    if not opt.no_musiconly:
        epochs = discover_epochs(opt.music_sweep_dir)
        if epoch_filter is not None:
            epochs = [e for e in epochs if e in epoch_filter]
        for ep in tqdm(epochs, desc="music_only sweep"):
            fdir = os.path.join(opt.music_sweep_dir, f"epoch_{ep:04d}")
            res = evaluate_dir(fdir, aligned, lead_cache, source="generated")
            if res.get("n_pairs", 0) == 0:
                continue
            res.update(config="music_only",
                       label=f"music_only @ ep{ep}",
                       epoch=ep)
            results[f"musiconly_{ep:04d}"] = res

    out_md   = os.path.join(opt.out_dir, "lma17_report.md")
    out_json = os.path.join(opt.out_dir, "lma17_report.json")
    write_report(results, out_md, out_json)
    print(f"\n[done] {len(results)} configs evaluated")
    print(f"       report  -> {out_md}")
    print(f"       payload -> {out_json}")


if __name__ == "__main__":
    main()
