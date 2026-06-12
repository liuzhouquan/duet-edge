# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EDGE (Editable Dance Generation From Music) is a PyTorch implementation of a diffusion-based model for generating physically-plausible, music-synchronized dance motions (CVPR 2023).

**Current development goal:** Extend the model to support **duet/reactive dance generation** — changing the conditioning input from music features alone to (lead dancer motion + music features), so the model learns to generate a follower's motion that is both music-synchronized and responsive to a lead dancer.

## Environment

**Target platform:** Linux x86_64 with NVIDIA GPU, Slurm job scheduler.

```bash
conda env create -f environment-linux.yml
conda activate edge
# then manually install pytorch3d:
pip install pytorch3d \
  --extra-index-url https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt210/download.html
accelerate config          # configure fp16 / single-GPU
bash download_model.sh     # download pretrained checkpoint
```

See `TODOLIST.md` for the full reproduction checklist, and `RunInML.md` for Slurm job submission instructions.

## Common Commands

### Single-person mode (original EDGE, fully compatible with pretrained checkpoint.pt)

**Train:**
```bash
sbatch run_training.sh
# or directly:
accelerate launch train.py --batch_size 128 --epochs 2000 --feature_type jukebox --learning_rate 0.0002
```

**Inference on custom music:**
```bash
python test.py --music_dir custom_music/
```

### Duet mode (extended — lead motion + music → follower motion)

**Important:** Duet and single-person checkpoints are NOT interchangeable.
The only difference is `cond_feature_dim`: 4800 (single) vs 4951 (duet).
Loading a solo checkpoint in duet mode will fail with a shape mismatch on `cond_projection`.

**Prepare data (train/test split + explicit lead-follower pairs):**
```bash
cd data
python create_duet_pairs.py         # generates data/splits/duet_pairs_{train,test}.json
python create_dataset.py --dataset_folder edge_aistpp --duet --extract-jukebox
cd ..
```

**Train from scratch:**
```bash
accelerate launch train.py --batch_size 128 --epochs 2000 --feature_type jukebox --learning_rate 0.0002 --duet
```

**Fine-tune from the pretrained single-person checkpoint (recommended — much faster):**
```bash
# All weights are inherited except cond_projection (re-initialized to accept 4951-dim input)
accelerate launch train.py --batch_size 128 --epochs 2000 --feature_type jukebox --learning_rate 0.0002 --duet --checkpoint checkpoint.pt
```

**Inference (requires a duet-trained checkpoint and preprocessed lead motion slices):**
```bash
python test.py \
  --feature_type jukebox \
  --checkpoint runs/train/exp/weights/train-2000.pt \
  --duet \
  --lead_motion_dir path/to/lead_slices/ \   # folder of .pkl files: {pos:[300,3], q:[300,72]} at 60fps
  --music_dir custom_music/ \
  --save_motions
```

**Evaluate duet quality (LMA similarity between lead and follower):**
```bash
# Baseline: measure LMA scores on raw AIST++ pairs (reference ceiling)
python eval/aistpp_lma_baseline.py --data_dir ../aist_plusplus_final/motions --out eval/baseline_report.txt

# Evaluate generated duet motions
python eval/lma_similarity.py --lead_dir eval/motions/lead/ --follower_dir eval/motions/follower/
```

**Paper-aligned 17-metric LMA report (Zhou et al. IUI '25 §4.2):**
```bash
# Sweeps full + lead_only configs across all available checkpoints, reusing
# eval/pfc_val_*_per_slice/ pkls (no model inference). Writes a Markdown
# appendix report + JSON to eval/lma17_report/.
python eval/run_lma17_sweep.py --out_dir eval/lma17_report
```
The 17 metrics = Body (8) + Effort (7) + Space (2). Shape (10) is excluded
per the paper. See `eval/lma17_features.py` for the metric definitions and
`eval/lma17_report/lma17_report.md` for the rendered appendix.

**Evaluate (Physical Foot Contact metric):**
```bash
python eval/eval_pfc.py --motion_path eval/motions/
```

**Monitor job:**
```bash
squeue -u $USER
tail -f logs/train-output.log
```

## Architecture

**Entry points:** `train.py` and `test.py` instantiate `EDGE` (in `EDGE.py`), which wires together the diffusion model, optimizer, and distributed training via `accelerate`.

**Core modules:**

- **`EDGE.py`** — Top-level class. `train_loop()` runs multi-GPU training with EMA updates and wandb logging; `render_sample()` runs full inference including music feature extraction, DDIM sampling, and video rendering.

- **`model/diffusion.py`** — `GaussianDiffusion`: cosine noise schedule (1000 timesteps), DDIM sampling, in-painting, long-sequence stitching, and a 4-component weighted loss (reconstruction 0.636×, velocity 2.964×, forward kinematics 0.646×, foot contact 10.942×).

- **`model/model.py`** — `DanceDecoder`: 8-layer transformer with FiLM conditioning on diffusion timestep, cross-attention to music embeddings, and rotary positional embeddings. Input is 151-dimensional motion (root pos 3D + 24 joint rotations as 6D vectors × 24 + 4 contact markers).

- **`vis.py`** — `SMPLSkeleton`: 24-joint SMPL kinematic tree, forward kinematics, foot contact detection (joint indices 7, 8, 10, 11). Also handles matplotlib animation and ffmpeg video rendering.

**Current data flow:**
```
music file → feature extraction (Jukebox 4800-dim or baseline 35-dim)
           → DanceDecoder (cross-attention to music embeddings)
           → GaussianDiffusion (DDIM sampler)
           → SMPLSkeleton FK → rendered video
```

**Duet data flow (implemented):**
```
lead motion (151-dim) + music features (4800-dim)
  → cat(dim=-1) → cond (4951-dim)  [DuetDataset / test.py]
  → DanceDecoder (cond_feature_dim=4951)
  → GaussianDiffusion → follower motion
```

## Key Conventions

**Motion representation (151-dim):** foot contacts (4) + root position (3) + joint rotations in 6D format (24×6=144). Order: [contacts, root_pos, rotations].

**Sequence length:** 150 frames (5 seconds @ 30 FPS). Long sequences use `long_ddim_sample()` with overlapping chunks to prevent discontinuities.

**Classifier-free guidance (solo):** 25% of training samples drop the full conditioning via `cond_drop_prob=0.25` (token-level: the *projected* cond embedding is replaced by a learned `null_cond_embed` parameter in `DanceDecoder.forward`); inference uses `guidance_weight=2` (standard 2-pass CFG). This follows Ho & Salimans (arXiv 2207.12598).

**Classifier-free guidance (duet — multi-conditional):** TWO LAYERS of dropout stack during duet training. Understanding the interaction matters for paper writing and any future re-training decisions.

*Layer 1 — Input-level partial dropout* (duet-specific, applied in `EDGE.py:247-261` before `diffusion()` is called):
- `--drop_prob_lead` (default `0.15`): zeros the first 151 dims of the cond vector (lead motion portion) with probability 0.15 per sample.
- `--drop_prob_music` (default `0.15`): zeros the remaining dims (music features) with probability 0.15 per sample.
- These two masks are sampled **independently**, so the four conditioning regimes appear with the following pre-Layer-2 probabilities:

  | Cond seen by `diffusion()` | Probability |
  |---|---|
  | `(lead, music)` — both present | `0.85 × 0.85 = 72.25%` |
  | `(lead, ∅)`     — music zeroed | `0.85 × 0.15 = 12.75%` |
  | `(∅, music)`    — lead zeroed  | `0.15 × 0.85 = 12.75%` |
  | `(∅, ∅)`        — both zeroed  | `0.15 × 0.15 =  2.25%` |

*Layer 2 — Token-level dropout* (inherited from EDGE, applied inside `DanceDecoder.forward` via `cond_drop_prob=0.25`):
- After Layer 1, the cond vector is projected through `cond_projection`. With probability 0.25 per sample, this projected embedding is **then** entirely replaced by `null_cond_embed` (the same learned-null mechanism EDGE solo uses).
- This produces a "completely unconditional" signal regardless of what Layer 1 did, so the model learns a stable `ε_∅` baseline (necessary for CFG to work).

*Resulting effective distribution* (Layer 1 ⊗ Layer 2, approximate):
- Fully unconditional `ε_∅` seen by model: `≈ 25% + (1−25%) × 2.25% ≈ 27%`
- Full `(lead, music)` conditioning: `≈ 75% × 72.25% ≈ 54%`
- Partial `(lead, ∅)` and `(∅, music)`: `≈ 10%` each
- This explains why the lead-only inference mode (`--guidance_music 0 --guidance_lead 2`) generalizes well at test time: the model has explicitly seen ~10% of training samples in that exact regime.

*Inference — Composable Diffusion three-pass formula* (`DanceDecoder.guided_forward`, lines ~362-368):
```
ε_∅       = forward(x, cond, cond_drop_prob=1)         # both nulled at token level
ε_music   = forward(x, cond_with_lead_zeroed,   …)     # only music
ε_lead    = forward(x, cond_with_music_zeroed,  …)     # only lead
ε         = ε_∅  +  w_music · (ε_music − ε_∅)  +  w_lead · (ε_lead − ε_∅)
```
This is the additive multi-conditional CFG from Liu et al., **ECCV 2022** ("Compositional Visual Generation with Composable Diffusion Models"), specialised to two conditioning streams. Setting `--guidance_lead 0` reduces to EDGE-style music-only generation; `--guidance_music 0` is the lead-only mode that empirically performs best in our duet setting. `--lead_motion_dir` is optional at inference — if omitted, the lead portion defaults to zeros and `--guidance_lead` must also be `0` (otherwise you guide toward a zero-lead OOD point).

**Important: any change to `cond_drop_prob`, `--drop_prob_lead`, or `--drop_prob_music` REQUIRES full retraining.** These probabilities shape the implicit training-time conditioning distribution that the model has converged to; modifying them invalidates the existing checkpoints' learned `ε_∅` and partial-cond paths. Current defaults (0.25 / 0.15 / 0.15) have produced LMA ≈ 0.93 on val (close to the 0.94 human-pair ceiling), so retuning is generally not worth a full retrain.

**Checkpoint format:** `{"ema_state_dict", "model_state_dict", "optimizer_state_dict", "normalizer"}`. Always load with EMA weights for inference.

**Feature caching:** Jukebox features can be cached as `.npy` files; use `--use_cached_features` to skip recomputation during iterative testing.

**Audio filenames:** Must be simple (e.g., `song.wav`, not filenames with spaces/special chars). Slices are named `songname_slice{N}.wav` and sorted numerically.

**Normalizer:** Motion statistics (mean/std) are computed from the training set and stored in checkpoints — apply consistently across splits.

**⚠️ Root-position scaling — `motions/` vs `motions_sliced/` (THIS HAS BIT US):**
The raw mocap `pos` field (= AIST++ `smpl_trans`) is **in unscaled mocap units** (typically ~50–180 cm or so per coordinate). AIST++ provides a `smpl_scaling` value per sequence (typically ~88) that converts to model units.

| File location | `pos` field state | `scale` key present? |
|---|---|---|
| `data/<split>/motions/<seq>.pkl` (raw, written by `filter_split_data.py`) | **NOT divided by scale** (raw mocap units, max ~180) | ✓ yes |
| `data/<split>/motions_sliced/<seq>_slice{N}.pkl` (written by `data/slice.py:34`) | **already divided by scale** (model units, max ~2) | ✗ no |

Symptom of mishandling: PFC = ~8000 instead of ~1.5 (since PFC ∝ velocity²; 88² ≈ 7700). LMA also collapses to ~0.6 when mixing scaled and unscaled joint trajectories across lead vs follower (Pearson is scale-invariant per-axis but Laban features mix axes).

**Rule:** any helper that reads from `data/<split>/motions/<seq>.pkl` directly (e.g., `eval/run_test_eval_longmode.py`'s GT FK step) must apply `pos = pos / data["scale"][0]` before calling `motion_to_joints`. Reading from `motions_sliced/` requires no scaling adjustment.

**Device compatibility:** `pin_memory` in DataLoader is enabled only when CUDA is available. `torch.cuda.empty_cache()` is guarded by `is_available()`.

## Data Splits

We use the **official AIST++ crossmodal splits** directly (located at `aist_plusplus_final/splits/`). Do NOT create custom split files.

| Official file | Seqs | Style | Choreography | Final duet pairs | Disk location |
|---|---|---|---|---|---|
| `crossmodal_train.txt` | 980 | sBM + sFM | ch03–ch10 | **383 pairs** (after ignore_list filter) | `data/train/` |
| `crossmodal_val.txt`   |  20 | sBM only  | ch01 | **10 pairs** | `data/val/` |
| `crossmodal_test.txt`  |  20 | sBM only  | ch02 | **10 pairs** | `data/test/` |

`383 = 400 raw pairs − 17 filtered by AIST++ `ignore_list.txt`` (motion/audio desync, mocap failure). `create_duet_pairs.py` enforces this filter so the JSON contains no phantom pairs.

**Why val and test share the same 10 music IDs:** This is intentional AIST++ design for crossmodal evaluation — val uses choreography `ch01` and test uses `ch02` for the same 10 songs. For the duet task this is fine: the music features are shared, but the lead motion conditioning (different choreography) and the ground-truth follower motions are completely different between val and test.

**Duet pairing rule (sBM sequences only):** Group by `(music_id, ch_id)` — same music, same choreography, different `dancer_id`. Each group of 2 dancers becomes one (lead, follower) pair (lead = lexicographically smaller dancer_id). sFM sequences are excluded. Sequences listed in `ignore_list.txt` are removed BEFORE pairing (must match `filter_split_data.py`).

**Files in `data/splits/`:**
- `crossmodal_train.txt` / `crossmodal_val.txt` / `crossmodal_test.txt` — copies of official files
- `ignore_list.txt` — AIST++ official "bad data" list (46 sequences, applied by both `create_duet_pairs.py` and `filter_split_data.py`)
- `duet_pairs_{train,val,test}.json` — generated by `create_duet_pairs.py`

Re-generate pair files if data changes:
```bash
cd data && python create_duet_pairs.py
```

### ⚠️ IMPORTANT: directory name ≠ ML-semantic role

Due to EDGE's original 2-split design (train/test only, no `val`), the duet
extension added a `val/` directory but **the training loop (`EDGE.py`) was
never updated** to read it. This produces a counter-intuitive mapping:

| Disk directory | Loaded by | What the code prints | Actual ML role |
|---|---|---|---|
| `data/train/` | `train_data_loader` in `EDGE.py` | `"Train Loss"` | Training (gradient updates) |
| **`data/test/`** | **`test_data_loader` in `EDGE.py`** (`train=False`) | **`"Val Loss"`, in-training LMA** | **Training monitor** (looked at every `save_interval` epoch — *not* held-out!) |
| **`data/val/`** | **`eval/run_full_sweep.py`, `eval/run_leadonly_eval.py`, `eval/run_val_pfc.py`, `render.py`** | sweep outputs in `eval/full_sweep*/` | **Held-out evaluation** (clean, never seen during training) |

**Consequence for paper reporting:**
- `data/val/` is the **actually clean** held-out set — use it for sweep-based checkpoint selection AND final reported numbers.
- `data/test/` was repeatedly observed during training (`save_interval=50` epoch loss + LMA + render demos) — treat it as the in-training monitor split, NOT as a final test set.
- In writing, refer to `data/val/` as the **evaluation set**. The reason `test/` exists but isn't reported on: EDGE's original train-loop uses `test/` as a monitoring split (a carry-over from the solo two-split design), so the `val/` directory added for the duet extension is the genuine held-out evaluation set.
- Do NOT swap them. The `EDGE.py:178` hard-coded `train=False → "test"` path is intentional to stay drop-in compatible with EDGE solo checkpoints; renaming would break checkpoint loading and future merges from EDGE upstream.

**TL;DR: always use `data/val/` for reporting metrics. `data/test/` is a monitoring split (observed during training), not a held-out test set, due to the legacy two-split naming.**
