"""
Create train/val/test splits for the AIST++ duet task.

Follows the official AIST++ crossmodal split logic:
  - Train  : all sBM sequences for the 50 training music_ids
             (reads music_ids from aist_plusplus_final/splits/crossmodal_train.txt)
  - Test   : all sBM sequences for the 10 held-out music_ids
             → goes to data/test/ via filter_split_data.py
             → used for monitoring during training (DuetDataset train=False)
  - Val    : sBM ch01 sequences for the 10 held-out music_ids (20 seqs)
             → goes to data/val/
             → used by final LMA evaluation scripts (not the training loop)

Why only sBM?
  The directory contains two recording conditions:
    sBM  (Style: Basic Movement) — multiple dancers simultaneously captured,
         pre-sliced into equal-length chunks ch01-ch10.  Same chunk_id across
         dancers IS the same 10-second musical window.  These form valid
         lead/follower pairs.
    sFM  (Style: Free Movement)  — solo full-length recordings; their ch
         numbers are camera/take IDs, NOT sequential time slices.  Two sFM
         files with the same ch number are NOT from the same time window.
         Excluded from all duet splits.

Output files (written to data/splits/):
  duet_train.txt        — ~1000 sBM sequences for 50 train music_ids
  duet_test.txt         — all sBM sequences for 10 held-out music_ids
  duet_val.txt          — sBM ch01 only for 10 held-out music_ids (20 seqs,
                          mirrors official crossmodal_val.txt)
  duet_split_summary.txt

Run from the EDGE project root:
    python data/create_duet_splits.py \
        --motions_dir  /path/to/edge_aistpp/motions \
        --official_dir /path/to/aist_plusplus_final/splits \
        --out_dir      data/splits
"""

import argparse
import os
from collections import defaultdict
from pathlib import Path


def parse_seq(filename: str):
    """Return (music_id, dancer_id, chunk_id, style) or None on failure."""
    base  = os.path.splitext(os.path.basename(filename))[0]
    parts = base.split("_")
    if len(parts) < 6:
        return None
    style     = parts[1]   # sBM or sFM
    dancer_id = parts[3]
    music_id  = parts[4]
    chunk_id  = parts[5]
    return music_id, dancer_id, chunk_id, style


def read_music_ids(split_txt: Path) -> set:
    """Extract unique music_ids from an official split .txt file."""
    ids = set()
    for line in split_txt.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("_")
        if len(parts) >= 5:
            ids.add(parts[4])
    return ids


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--motions_dir",
        default="data/edge_aistpp/motions",
        help="Directory containing AIST++ .pkl motion files",
    )
    parser.add_argument(
        "--official_dir",
        default="../aist_plusplus_final/splits",
        help="Directory containing official AIST++ split .txt files",
    )
    parser.add_argument(
        "--out_dir",
        default="data/splits",
        help="Output directory for duet split .txt files",
    )
    args = parser.parse_args()

    motions_dir  = Path(args.motions_dir)
    official_dir = Path(args.official_dir)
    out_dir      = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not motions_dir.is_dir():
        raise FileNotFoundError(f"--motions_dir not found: {motions_dir}")
    if not official_dir.is_dir():
        raise FileNotFoundError(f"--official_dir not found: {official_dir}")

    # ── 1. Read official music_id sets ────────────────────────────────────────
    train_music_ids  = read_music_ids(official_dir / "crossmodal_train.txt")
    heldout_music_ids = read_music_ids(official_dir / "crossmodal_test.txt")

    print(f"Official train music_ids : {len(train_music_ids)}")
    print(f"Official held-out music_ids: {len(heldout_music_ids)}")
    print(f"  held-out: {sorted(heldout_music_ids)}")

    # ── 2. Scan motions_dir, keep only sBM files ──────────────────────────────
    train_seqs = []
    test_seqs  = []   # all sBM for held-out (→ data/test/, monitoring)
    val_seqs   = []   # sBM ch01 for held-out (→ data/val/, final eval)

    skipped_sfm   = 0
    skipped_other = 0

    for f in sorted(motions_dir.iterdir()):
        if f.suffix != ".pkl":
            continue
        parsed = parse_seq(f.name)
        if parsed is None:
            skipped_other += 1
            continue
        music_id, dancer_id, chunk_id, style = parsed

        if style != "sBM":
            skipped_sfm += 1
            continue

        seq_name = f.stem

        if music_id in train_music_ids:
            train_seqs.append(seq_name)
        elif music_id in heldout_music_ids:
            test_seqs.append(seq_name)          # all chunks → monitoring
            if chunk_id == "ch01":
                val_seqs.append(seq_name)       # ch01 only  → final eval

    print(f"\nSkipped: {skipped_sfm} sFM files, {skipped_other} unparseable files")

    # ── 3. Write split files ───────────────────────────────────────────────────
    splits = {
        "duet_train": sorted(train_seqs),
        "duet_test":  sorted(test_seqs),
        "duet_val":   sorted(val_seqs),
    }
    for name, seqs in splits.items():
        out_path = out_dir / f"{name}.txt"
        out_path.write_text("\n".join(seqs) + "\n")
        print(f"Wrote {len(seqs):4d} sequences → {out_path}")

    # ── 4. Verify no leakage ──────────────────────────────────────────────────
    train_set  = set(train_seqs)
    test_set   = set(test_seqs)
    val_set    = set(val_seqs)
    assert not (train_set & test_set),  "train ∩ test not empty!"
    assert not (train_set & val_set),   "train ∩ val not empty!"
    assert val_set.issubset(test_set),  "val is not a subset of test!"
    print("\n✓ No sequence leakage between splits")

    # ── 5. Pairing stats ──────────────────────────────────────────────────────
    # For each (music_id, chunk_id), count how many dancers are available
    from collections import Counter

    def pairing_stats(seq_list):
        groups = defaultdict(set)  # (music_id, chunk_id) → {dancer_ids}
        for name in seq_list:
            parsed = parse_seq(name)
            if parsed:
                mid, did, cid, _ = parsed
                groups[(mid, cid)].add(did)
        n_groups     = len(groups)
        n_pairs      = sum(len(d) * (len(d) - 1) for d in groups.values())
        valid_groups = sum(1 for d in groups.values() if len(d) >= 2)
        return n_groups, valid_groups, n_pairs

    print()
    for name, seqs in splits.items():
        ng, vg, np_ = pairing_stats(seqs)
        print(f"  {name}: {len(seqs)} seqs  "
              f"chunk_groups={ng}  paireable_groups={vg}  "
              f"ordered_pairs={np_}")

    # ── 6. Summary file ───────────────────────────────────────────────────────
    summary = [
        "AIST++ Duet Split Summary (sBM only, official music_id split)",
        "=" * 68,
        "",
        "  Recording condition: sBM only",
        "    sBM = multiple dancers captured simultaneously, pre-sliced to",
        "          equal-length chunks ch01-ch10.  Same chunk_id across",
        "          dancers IS the same musical time window.",
        "    sFM = solo full-length recordings, excluded (ch numbers are",
        "          camera/take IDs, not time-aligned across dancers).",
        "",
        f"  Official train music_ids: {len(train_music_ids)} (50 music_ids)",
        f"  Official held-out music_ids: {len(heldout_music_ids)}",
        f"    {sorted(heldout_music_ids)}",
        "",
        f"  duet_train : {len(train_seqs):4d} seqs  → data/train/  (training)",
        f"  duet_test  : {len(test_seqs):4d} seqs  → data/test/   (monitoring during training)",
        f"  duet_val   : {len(val_seqs):4d} seqs  → data/val/    (final LMA eval, ch01 only)",
        "",
        "  Note: val ⊂ test (val is the ch01 subset of test).",
        "  No sequence leakage between train and held-out sets.",
    ]
    summary_path = out_dir / "duet_split_summary.txt"
    summary_path.write_text("\n".join(summary) + "\n")
    print()
    print("\n".join(summary))
    print(f"\nSummary → {summary_path}")


if __name__ == "__main__":
    main()
