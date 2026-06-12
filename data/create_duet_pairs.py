"""
Generate explicit lead-follower pair files for the duet task.

Pairing logic
-------------
- sBM only (sFM excluded entirely)
- Pairing key: (music_id, ch_id) — same music, same choreography number
- Each group has exactly 2 dancers; lead = lexicographically smaller dancer_id
- No symmetric pairs (each group produces exactly one ordered pair)
- Train / val / test split follows the official AIST++ crossmodal music_id split
- Sequences listed in ``splits/ignore_list.txt`` (AIST++ official "bad data"
  list — motion/audio desync, mocap failure, etc.) are removed BEFORE pairing.
  This must match ``filter_split_data.py`` which physically excludes the same
  sequences when copying data into ``train/`` / ``val/`` / ``test/``; otherwise
  the JSON contains phantom pairs whose underlying files do not exist.

Output (written to data/splits/):
  duet_pairs_train.json  — list of {"lead": "...", "follower": "..."} objects
  duet_pairs_val.json
  duet_pairs_test.json

Note: val and test share the same 10 music IDs by AIST++ design (different
choreography: val=ch01, test=ch02). This is intentional crossmodal evaluation
design, not a leakage issue — lead motion conditioning and ground-truth follower
motions are completely different between val and test.

Run from the EDGE project root:
    python data/create_duet_pairs.py
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path


def parse_seq(filename):
    """Return (music_id, dancer_id, ch_id, style) or None."""
    base  = os.path.splitext(os.path.basename(filename))[0]
    parts = base.split("_")
    if len(parts) < 6:
        return None
    style     = parts[1]   # sBM or sFM
    dancer_id = parts[3]   # d04
    music_id  = parts[4]   # mBR0
    ch_id     = parts[5]   # ch01
    return music_id, dancer_id, ch_id, style


def read_seqs(split_txt):
    """Return set of sequence stems (no extension) from a split .txt file."""
    seqs = set()
    for line in Path(split_txt).read_text().splitlines():
        line = line.strip()
        if line:
            seqs.add(line)
    return seqs


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--motions_dir",  default="data/edge_aistpp/motions")
    parser.add_argument("--official_dir", default="../aist_plusplus_final/splits")
    parser.add_argument("--out_dir",      default="data/splits")
    args = parser.parse_args()

    motions_dir  = Path(args.motions_dir)
    official_dir = Path(args.official_dir)
    out_dir      = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not motions_dir.is_dir():
        raise FileNotFoundError(f"--motions_dir not found: {motions_dir}")
    if not official_dir.is_dir():
        raise FileNotFoundError(f"--official_dir not found: {official_dir}")

    # AIST++ official "bad data" list. ``filter_split_data.py`` also excludes
    # these sequences when copying motion/wav into train/val/test dirs, so the
    # pair JSON must agree — otherwise we generate pairs whose data files are
    # absent downstream (the "phantom pair" bug).
    ignore_path = out_dir / "ignore_list.txt"
    if ignore_path.exists():
        ignore_list = {
            line.strip() for line in ignore_path.read_text().splitlines()
            if line.strip()
        }
        print(f"Loaded {len(ignore_list)} entries from {ignore_path}")
    else:
        ignore_list = set()
        print(f"[WARN] no {ignore_path}; not filtering AIST++ bad sequences")

    # Use full sequence names for split assignment.
    # Val and test share the same 10 music_ids (AIST++ design); differentiating
    # by music_id alone would assign all pairs to val. We use sequence membership
    # instead: if either dancer in a pair appears in the official val/test split,
    # the pair is assigned there.
    train_seqs = read_seqs(official_dir / "crossmodal_train.txt")
    val_seqs   = read_seqs(official_dir / "crossmodal_val.txt")
    test_seqs  = read_seqs(official_dir / "crossmodal_test.txt")
    print(f"Official train seqs : {len(train_seqs)}")
    print(f"Official val   seqs : {len(val_seqs)}")
    print(f"Official test  seqs : {len(test_seqs)}")

    # Group sBM sequences by (music_id, ch_id) → {dancer_id: seq_stem}
    groups   = defaultdict(dict)
    skipped_nonsBM = 0
    skipped_ignore = 0
    for f in sorted(motions_dir.iterdir()):
        if f.suffix != ".pkl":
            continue
        parsed = parse_seq(f.name)
        if parsed is None:
            skipped_nonsBM += 1
            continue
        music_id, dancer_id, ch_id, style = parsed
        if style != "sBM":
            skipped_nonsBM += 1
            continue
        if f.stem in ignore_list:
            skipped_ignore += 1
            continue
        groups[(music_id, ch_id)][dancer_id] = f.stem

    print(f"Skipped {skipped_nonsBM} non-sBM or unparseable files")
    print(f"Skipped {skipped_ignore} sequences in AIST++ ignore_list")
    print(f"Found {len(groups)} (music_id, ch_id) groups")

    train_pairs    = []
    val_pairs      = []
    test_pairs     = []
    skipped_groups = 0

    for (music_id, ch_id), dancer_map in sorted(groups.items()):
        if len(dancer_map) != 2:
            skipped_groups += 1
            print(f"  [SKIP] {music_id}/{ch_id} has {len(dancer_map)} dancer(s), expected 2")
            continue
        dancers = sorted(dancer_map.keys())
        pair = {
            "lead":     dancer_map[dancers[0]],
            "follower": dancer_map[dancers[1]],
        }
        lead, follower = pair["lead"], pair["follower"]
        if lead in val_seqs or follower in val_seqs:
            val_pairs.append(pair)
        elif lead in test_seqs or follower in test_seqs:
            test_pairs.append(pair)
        elif lead in train_seqs or follower in train_seqs:
            train_pairs.append(pair)
        # pairs where no member appears in any official split are silently ignored

    print(f"Skipped {skipped_groups} groups without exactly 2 dancers")

    # Verify no music_id leakage between train and eval sets
    def music_ids(pairs):
        return {p["lead"].split("_")[4] for p in pairs}
    train_mids = music_ids(train_pairs)
    val_mids   = music_ids(val_pairs)
    test_mids  = music_ids(test_pairs)
    assert not (train_mids & val_mids),  "music_id leakage between train and val!"
    assert not (train_mids & test_mids), "music_id leakage between train and test!"
    # val and test intentionally share the same 10 music_ids (AIST++ crossmodal design)
    print("✓ No music_id leakage between train and eval sets")
    print(f"  (val ∩ test music_ids = {len(val_mids & test_mids)} — expected 10, by design)")

    for name, pairs in [
        ("duet_pairs_train", train_pairs),
        ("duet_pairs_val",   val_pairs),
        ("duet_pairs_test",  test_pairs),
    ]:
        out_path = out_dir / f"{name}.json"
        out_path.write_text(json.dumps(pairs, indent=2) + "\n")
        print(f"Wrote {len(pairs):4d} pairs → {out_path}")


if __name__ == "__main__":
    main()
