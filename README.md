# Duet-EDGE — Reactive Duet Dance Generation

> Built on **EDGE** (Tseng et al., *Editable Dance Generation From Music*, CVPR 2023).
> This fork extends the original music→solo-dance diffusion model into a **duet model**: given a **lead** dancer's motion and the accompanying music, it generates a **follower** dancer's motion that is both music-synchronized *and* responsive to the lead.

---

## 1. Overview

### Goal
The original EDGE generates a single dancer's motion from music. The prior duet work it builds on (Zhou et al., IUI '25) produced two dancers by running solo inference twice on the *same music* — the two dancers never "see" each other, so any coordination between them is incidental (it comes only from a shared beat).

**Our goal is to model the lead→follower interaction directly.** We change the conditioning input from *music alone* to *(lead motion + music)* and **train** the model to generate a follower that coordinates with a specific lead. The model therefore learns genuine lead–follower coordination rather than borrowing it from a shared rhythm.

### What we implemented
- **Duet conditioning.** The conditioning vector is `concat(lead_motion, music)`. Dimensions: solo `cond_feature_dim = 4800` (music only) → duet `4951` (`4800` Jukebox music + `151` lead motion).
- **Compositional classifier-free guidance** (3-pass, after Liu et al., ECCV 2022). The two conditioning streams can be scaled independently at inference time:
  ```
  ε = ε_uncond  +  w_music · (ε_music − ε_uncond)  +  w_lead · (ε_lead − ε_uncond)
  ```
  Setting `w_lead = 0` recovers music-only (original EDGE) behaviour; `w_music = 0` gives lead-only generation; both `= 0` gives a pure-unconditional control.
- **Two-layer conditioning dropout** during training (input-level partial dropout of each stream + EDGE's token-level null dropout), so the model has a well-formed unconditional prior and can be guided by either stream alone at inference.
- **LMA-based evaluation.** A 17-metric Laban Movement Analysis pipeline (after Zhou et al., IUI '25 §4.2) measuring lead–follower kinematic agreement, alongside the Physical Foot Contact (PFC) plausibility metric from the EDGE paper.

### Results (epoch-1800 checkpoint, AIST++ `val`, 10 held-out pairs)

| Configuration | PFC ↓ | LMA 51-dim r ↑ | Body r | Effort r | Space r |
|---|---:|---:|---:|---:|---:|
| GT (real human follower) | 1.26 | 0.994 | 0.330 | 0.258 | 0.370 |
| **full** (w_music = w_lead = 1.0) | 1.80 | 0.989 | 0.196 | 0.154 | 0.165 |
| lead-only (w_music = 0, w_lead = 1.0) | 1.95 | 0.989 | 0.193 | 0.161 | 0.181 |
| music-only (w_music = 1.0, w_lead = 0) | 2.23 | 0.978 | **−0.001** | 0.067 | 0.077 |

Reading of the numbers:
- **Style-level coordination is near the human ceiling.** The 51-dim aggregate LMA r reaches ≈ 0.99 vs the real-human-pair ceiling of 0.994 — the follower's overall movement profile closely matches the lead's.
- **Frame-by-frame responsiveness has room to grow.** The per-metric Body r (≈ 0.20) is about 60 % of the human-pair value (0.33). Note that even *real* partners only reach 0.33 — coordination in real duet is partial and stylistic, not literal mirroring.
- **The lead stream is what carries pose coordination.** In the music-only ablation the Body-component r collapses to ≈ 0, confirming that body-shape agreement comes from the lead conditioning, not the shared music.
- **No physical-plausibility penalty.** Duet PFC (≈ 1.8) sits in the same range as EDGE-solo (1.30 reproduced / 1.54 published).

Full per-metric breakdown and the 15-configuration CFG sweep are in `eval/lma17_report/` and `eval/ep1800_render_eval/MASTER_REPORT.md`. Rendered example videos and a viewing guide are under `renders/ep1800_cfg_sweep/`.

---

## 2. Dataset & lead–follower pairing

We use the **official AIST++ crossmodal splits** directly — do not create custom split files.

| Split | Seqs | Choreography | Style | Duet pairs | Directory |
|---|---:|---|---|---:|---|
| `crossmodal_train` | 980 | ch03–ch10 | sBM + sFM | **383** | `data/train/` |
| `crossmodal_val`   | 20  | ch01 | sBM | **10** | `data/val/` |
| `crossmodal_test`  | 20  | ch02 | sBM | **10** | `data/test/` |

(`383 = 400 raw pairs − 17` removed by AIST++'s official `ignore_list.txt` for motion/audio desync or mocap failure.)

**Pairing rule (sBM only).** Group sequences by `(music_id, choreography_id)` — i.e. *same music, same choreography, different dancer*. Each such group of two dancers becomes one `(lead, follower)` pair, with the **lexicographically smaller `dancer_id` chosen as the lead**. `sFM` (free-style) sequences have only one dancer per `(music, choreography)` and so cannot form pairs — they are excluded from the duet dataset. Sequences in `ignore_list.txt` are removed *before* pairing.

> ⚠️ **Directory-name caveat (important for reporting).** Due to EDGE's original train/test-only design, the in-training loop monitors `data/test/`, while `data/val/` (added for the duet extension) is the genuinely held-out evaluation set. **Report metrics on `data/val/`.** `data/test/` was observed during training and should be treated as a monitoring split, not a final test set.

### Downloading data & the trained checkpoint

The AIST++ data and the trained duet checkpoint (`train-1800.pt`) are **not** in this repository (too large). Download them here:

**📦 https://uoa-my.sharepoint.com/:f:/g/personal/zliu753_uoa_auckland_ac_nz/IgBJ0-qFIogFSKnizuJcgx0TAXsfBZE_gOzsF3_foWKCq7g?e=87sQGi**

After downloading:
- The AIST++ folder contains **raw data** (motion + wav only — *not* preprocessed). Place it under `data/` per the layout above, then run the preprocessing in §4 (`create_duet_pairs.py` + `create_dataset.py --extract-jukebox`) to produce the sliced motions and Jukebox features before training or evaluation. This step needs `jukemirlib` (see §3) and can take many hours.
- Place the trained duet checkpoint at `runs/train/exp9/weights/train-1800.pt` (or anywhere, and pass its path via `--checkpoint`).
- Separately, the original EDGE solo checkpoint (`checkpoint.pt`, used for fine-tuning) can be fetched with `bash download_model.sh`.

---

## 3. Environment setup

Target platform: Linux x86_64 with an NVIDIA GPU (≥ 16 GB), CUDA 11.8.

**Recommended — one step:**

```bash
bash setup_env.sh        # creates the `edge` conda env and installs everything below
conda activate edge
```

**Or manually** (these are the steps `setup_env.sh` automates — the order matters):

```bash
# 1. conda packages (the yml's pip section is skipped on purpose)
conda env create -f environment-linux.yml || true
conda activate edge

# 2. downgrade setuptools FIRST — jukebox's setup.py needs pkg_resources,
#    which setuptools >= 70 removed
pip install "setuptools<70"

# 3. pip deps incl. jukemirlib (GitHub-only; provides the Jukebox music features)
pip install einops librosa soundfile p_tqdm wandb "accelerate>=0.24,<1.0" \
    "git+https://github.com/rodrigo-castellon/jukemirlib.git" --no-build-isolation

# 4. pytorch3d — must use --find-links (--extra-index-url silently finds nothing here)
pip install pytorch3d \
    --find-links https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt210/download.html

# 5. configure accelerate (single GPU, fp16)
accelerate config
```

> **`jukemirlib` is required, not optional.** Jukebox features are the model's music input, used by both training and inference — skipping it fails at feature extraction. Jukebox feature extraction is also the heaviest step and benefits from a recent GPU.

---

## 4. Training

First build the duet pairs and preprocess the dataset (this extracts Jukebox features and can take many hours):

```bash
cd data
python create_duet_pairs.py        # → data/splits/duet_pairs_{train,val,test}.json
python create_dataset.py --dataset_folder edge_aistpp --duet --extract-jukebox
cd ..
```

**Fine-tune from the EDGE solo checkpoint (recommended — much faster to converge):**

```bash
accelerate launch train.py \
    --duet --checkpoint checkpoint.pt \
    --batch_size 128 --epochs 2000 \
    --feature_type jukebox --learning_rate 0.0002
```

All solo weights are inherited except `cond_projection`, which is re-initialized to accept the 4951-dim duet conditioning. To train from scratch, drop `--checkpoint`.

Two duet-specific training flags shape the conditioning dropout: `--drop_prob_lead` and `--drop_prob_music` (both default `0.15`). **Changing them requires a full retrain** — they define the conditioning distribution the model converges to. The current defaults produced the results in §1.

---

## 5. Inference

Duet inference needs a duet-trained checkpoint, a folder of music `.wav` files, and a folder of preprocessed lead-motion slices (`.pkl` files containing `pos [300,3]` and `q [300,72]` at 60 fps):

```bash
python test.py \
    --duet \
    --checkpoint runs/train/exp9/weights/train-1800.pt \
    --feature_type jukebox \
    --music_dir custom_music/ \
    --lead_motion_dir path/to/lead_slices/ \
    --guidance_music 0 --guidance_lead 2 \
    --save_motions
```

The two guidance weights select the conditioning regime:

| Flags | Regime |
|---|---|
| `--guidance_music W --guidance_lead W` | **full** duet (both streams) |
| `--guidance_music 0 --guidance_lead W` | **lead-only** (empirically the best / simplest; matches `full` on coordination) |
| `--guidance_music W --guidance_lead 0` | **music-only** (≈ original EDGE; no lead coordination) |
| `--guidance_music 0 --guidance_lead 0` | pure unconditional control |

If `--lead_motion_dir` is omitted, the lead portion defaults to zeros and you must also set `--guidance_lead 0` (otherwise you guide toward an out-of-distribution zero-lead point).

---

## 6. Evaluation

```bash
# 17-metric LMA report (lead–follower coordination), sweeping available checkpoints:
python eval/run_lma17_sweep.py --out_dir eval/lma17_report

# Physical Foot Contact (plausibility) on generated motions:
python eval/eval_pfc.py --motion_path eval/motions/
```

The reproducible single-checkpoint CFG sweep that produced the §1 numbers and the example videos is `eval/run_ep1800_cfg_sweep.py` (see its docstring for options).

---

## 7. Repository guide — how the codebase is organized

```
EDGE/
├── train.py / test.py          entry points (training / inference + rendering)
├── EDGE.py                     top-level model wrapper (train loop, sampling, rendering)
├── args.py                     all CLI flags (train opts + test opts, incl. duet flags)
├── render.py                   long-form stitched rendering helpers
├── model/
│   ├── diffusion.py            GaussianDiffusion: DDIM sampling, long-form stitching, losses
│   └── model.py                DanceDecoder transformer + compositional CFG (guided_forward)
├── dataset/
│   └── dance_dataset.py        DuetDataset: builds concat(lead, music) conditioning
├── data/
│   ├── create_duet_pairs.py    builds (lead, follower) pair JSONs from official splits
│   ├── create_dataset.py       slicing + Jukebox feature extraction
│   ├── splits/                 official AIST++ split copies + ignore_list + pair JSONs
│   └── {train,val,test}/       processed motion/audio/features (downloaded, git-ignored)
├── eval/                       LMA + PFC evaluation scripts and generated reports
├── vis.py                      SMPL skeleton, forward kinematics, video rendering
├── runs/                       training checkpoints (git-ignored; download train-1800.pt)
└── renders/                    rendered videos + per-package READMEs (git-ignored)
```

**Conditioning data flow (duet):**
```
lead motion (151-dim) + music features (4800-dim)
   → concat → cond (4951-dim)            [DuetDataset / test.py]
   → DanceDecoder (cross-attention, FiLM, compositional CFG)
   → GaussianDiffusion (DDIM sampler)
   → follower motion (151-dim) → SMPL forward kinematics → rendered video
```

**Conventions worth knowing:**
- **Motion representation (151-dim):** `[4 foot-contacts, 3 root position, 24 joints × 6D rotation]`.
- **Sequence length:** 150 frames = 5 s @ 30 fps. Longer outputs use overlapping-window stitching (`long_ddim_sample`) — see `renders/ep1800_cfg_sweep/` for a diagram of the overlap handling.
- **Solo vs duet checkpoints are not interchangeable** (the `cond_projection` shape differs: 4800 vs 4951). Loading the wrong one fails with a shape mismatch.
- **Large artifacts are git-ignored** (`runs/`, `data/`, `renders/`, `*.pt`, `*.pkl`, `*.npy`, …) and distributed via the SharePoint link in §2.
- **Optional Blender/FBX export** (`SMPL-to-FBX/Convert.py`) needs the binary assets `ybot.fbx` and `FbxFormatConverter.exe`, which are *not* tracked in git — obtain them from the [original SMPL-to-FBX repo](https://github.com/softcat477/SMPL-to-FBX) if needed.

`CLAUDE.md` contains deeper implementation notes (the exact CFG dropout math, the root-position scaling gotcha, etc.) for anyone modifying the conditioning or evaluation pipeline.

---

## 8. Future work — interactive demo

The natural next step is a **real-time interactive demo**: a lead dancer moves, and the model generates the follower live, so the system can be driven by a human partner rather than a pre-recorded lead. The latency and streaming design have been scoped (the 5-second window and overlap-stitching are the relevant constraints), **but the interactive demo is not yet implemented** — the current pipeline is offline. This is the main direction for follow-up work.

---

## Acknowledgements & citation

This project is built on [EDGE: Editable Dance Generation From Music](https://arxiv.org/abs/2211.10658) (Tseng, Castellon, Liu — CVPR 2023). The 17-metric LMA evaluation follows Zhou et al., *Real-Time Full-body Interaction with AI Dance Models* (IUI '25).

```bibtex
@inproceedings{tseng2023edge,
  title={EDGE: Editable Dance Generation From Music},
  author={Tseng, Jonathan and Castellon, Rodrigo and Liu, C Karen},
  booktitle={CVPR},
  year={2023}
}
```
