"""
Generate explicit lead-follower pair files for the duet task.

Pairing logic
-------------
- sBM only (sFM excluded entirely)
- Pairing key: (music_id, ch_id) — same music, same choreography number
- Each group has exactly 2 dancers; lead = lexicographically smaller dancer_id
- No symmetric pairs (each group produces exactly one ordered pair)
- Train / test split follows the official AIST++ crossmodal music_id split

Output (written to data/splits/):
  duet_pairs_train.json  — list of {"lead": "...", "follower": "..."} objects
  duet_pairs_test.json

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


def read_music_ids(split_txt):
    ids = set()
    for line in Path(split_txt).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("_")
        if len(parts) >= 5:
            ids.add(parts[4])
    return ids


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

    train_music_ids = read_music_ids(official_dir / "crossmodal_train.txt")
    test_music_ids  = read_music_ids(official_dir / "crossmodal_test.txt")
    print(f"Official train music_ids : {len(train_music_ids)}")
    print(f"Official test  music_ids : {len(test_music_ids)}")

    # Group sBM sequences by (music_id, ch_id) → {dancer_id: seq_stem}
    groups   = defaultdict(dict)
    skipped  = 0
    for f in sorted(motions_dir.iterdir()):
        if f.suffix != ".pkl":
            continue
        parsed = parse_seq(f.name)
        if parsed is None:
            skipped += 1
            continue
        music_id, dancer_id, ch_id, style = parsed
        if style != "sBM":
            skipped += 1
            continue
        groups[(music_id, ch_id)][dancer_id] = f.stem

    print(f"Skipped {skipped} non-sBM or unparseable files")
    print(f"Found {len(groups)} (music_id, ch_id) groups")

    train_pairs    = []
    test_pairs     = []
    skipped_groups = 0

    for (music_id, ch_id), dancer_map in sorted(groups.items()):
        if len(dancer_map) != 2:
            skipped_groups += 1
            print(f"  [SKIP] {music_id}/{ch_id} has {len(dancer_map)} dancer(s), expected 2")
            continue
        dancers       = sorted(dancer_map.keys())
        pair = {
            "lead":     dancer_map[dancers[0]],
            "follower": dancer_map[dancers[1]],
        }
        if music_id in train_music_ids:
            train_pairs.append(pair)
        elif music_id in test_music_ids:
            test_pairs.append(pair)
        # sequences belonging to neither split are silently ignored

    print(f"Skipped {skipped_groups} groups without exactly 2 dancers")

    # Verify no music_id leakage
    train_mids = {p["lead"].split("_")[4] for p in train_pairs}
    test_mids  = {p["lead"].split("_")[4] for p in test_pairs}
    assert not (train_mids & test_mids), "music_id leakage between train and test!"
    print("✓ No music_id leakage between train and test")

    for name, pairs in [("duet_pairs_train", train_pairs), ("duet_pairs_test", test_pairs)]:
        out_path = out_dir / f"{name}.json"
        out_path.write_text(json.dumps(pairs, indent=2) + "\n")
        print(f"Wrote {len(pairs):4d} pairs → {out_path}")


if __name__ == "__main__":
    main()
