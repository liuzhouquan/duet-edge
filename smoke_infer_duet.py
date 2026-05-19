"""
Duet inference smoke test — macOS, no pytorch3d/wandb/librosa needed.

Takes 10 lead motion sequences from the validation split, runs DDIM sampling
(50 steps, baseline 35-dim music features zeroed out = no-music mode), and
saves the generated follower motions to smoke_infer_out/.

NOTE: Uses the solo pretrained checkpoint with strict=False — the cond_projection
layer (151+35=186 → 512) is randomly initialized since the solo checkpoint was
trained with 4800-dim jukebox conditioning. Output quality is therefore random/
meaningless; the point is to verify the full pipeline runs without errors.

Run from the EDGE project root:
    conda run -n edge-mac python smoke_infer_duet.py
"""

import sys, types, os, pickle
import torch
import numpy as np

# ── 1. Stub unavailable modules (same set as smoke_test_duet.py) ──────────────

def _axis_angle_to_quaternion(aa):
    angle = aa.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    axis  = aa / angle
    s, c  = (angle / 2).sin(), (angle / 2).cos()
    return torch.cat([c, axis * s], dim=-1)

def _quaternion_to_axis_angle(q):
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    sin_half = q[..., 1:].norm(dim=-1, keepdim=True).clamp(min=1e-8)
    cos_half = q[..., :1]
    angle    = 2 * torch.atan2(sin_half, cos_half)
    axis     = q[..., 1:] / sin_half
    return axis * angle

def _quaternion_multiply(q1, q2):
    w1,x1,y1,z1 = q1.unbind(-1)
    w2,x2,y2,z2 = q2.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)

def _quaternion_apply(q, v):
    qv   = q[..., 1:]
    uv   = torch.cross(qv, v, dim=-1)
    uuv  = torch.cross(qv, uv, dim=-1)
    return v + 2 * (q[..., :1] * uv + uuv)

class _RotateAxisAngle:
    def __init__(self, angle, axis="X", degrees=True):
        rad = torch.tensor(angle * (3.14159265 / 180) if degrees else angle)
        c, s = rad.cos(), rad.sin()
        if axis == "X":
            self.R = torch.tensor([[1,0,0],[0,c,-s],[0,s,c]], dtype=torch.float32)
        elif axis == "Y":
            self.R = torch.tensor([[c,0,s],[0,1,0],[-s,0,c]], dtype=torch.float32)
        else:
            self.R = torch.tensor([[c,-s,0],[s,c,0],[0,0,1]], dtype=torch.float32)
    def transform_points(self, pts):
        return pts @ self.R.T

def _axis_angle_to_matrix(aa):
    angle = aa.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    axis  = aa / angle
    # squeeze keepdim so shapes broadcast with x/y/z scalars
    c = angle.cos().squeeze(-1)
    s = angle.sin().squeeze(-1)
    t = 1 - c
    x, y, z = axis.unbind(-1)
    R = torch.stack([
        torch.stack([t*x*x+c,   t*x*y-s*z, t*x*z+s*y], dim=-1),
        torch.stack([t*x*y+s*z, t*y*y+c,   t*y*z-s*x], dim=-1),
        torch.stack([t*x*z-s*y, t*y*z+s*x, t*z*z+c  ], dim=-1),
    ], dim=-2)
    return R

def _matrix_to_rotation_6d(R):
    return R[..., :2].permute(*range(R.dim()-2), -1, -2).reshape(*R.shape[:-2], 6)

def _rotation_6d_to_matrix(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)

def _matrix_to_quaternion(R):
    t = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    q = torch.zeros(*R.shape[:-2], 4, device=R.device, dtype=R.dtype)
    s = (t + 1).clamp(min=0).sqrt() * 2
    q[..., 0] = 0.25 * s
    q[..., 1] = (R[..., 2, 1] - R[..., 1, 2]) / s.clamp(min=1e-8)
    q[..., 2] = (R[..., 0, 2] - R[..., 2, 0]) / s.clamp(min=1e-8)
    q[..., 3] = (R[..., 1, 0] - R[..., 0, 1]) / s.clamp(min=1e-8)
    return q

def _quaternion_to_matrix(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        torch.stack([1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], dim=-1),
        torch.stack([2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], dim=-1),
        torch.stack([2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)], dim=-1),
    ], dim=-2)

def _matrix_to_axis_angle(R):
    return _quaternion_to_axis_angle(_matrix_to_quaternion(R))

_p3d_transforms = types.ModuleType("pytorch3d.transforms")
for _name, _fn in [
    ("axis_angle_to_quaternion", _axis_angle_to_quaternion),
    ("quaternion_to_axis_angle", _quaternion_to_axis_angle),
    ("quaternion_multiply",      _quaternion_multiply),
    ("quaternion_apply",         _quaternion_apply),
    ("RotateAxisAngle",          _RotateAxisAngle),
    ("axis_angle_to_matrix",     _axis_angle_to_matrix),
    ("matrix_to_axis_angle",     _matrix_to_axis_angle),
    ("matrix_to_rotation_6d",    _matrix_to_rotation_6d),
    ("rotation_6d_to_matrix",    _rotation_6d_to_matrix),
    ("matrix_to_quaternion",     _matrix_to_quaternion),
    ("quaternion_to_matrix",     _quaternion_to_matrix),
]:
    setattr(_p3d_transforms, _name, _fn)

_p3d = types.ModuleType("pytorch3d")
sys.modules["pytorch3d"]            = _p3d
sys.modules["pytorch3d.transforms"] = _p3d_transforms

for _mod in ["wandb", "librosa", "soundfile", "jukemirlib"]:
    sys.modules[_mod] = types.ModuleType(_mod)

_p_tqdm = types.ModuleType("p_tqdm")
_p_tqdm.p_map = lambda fn, it: list(map(fn, it))
sys.modules["p_tqdm"] = _p_tqdm

import unittest.mock as _mock
for _mod in ["matplotlib", "matplotlib.pyplot", "matplotlib.animation",
             "matplotlib.cm", "matplotlib.colors"]:
    sys.modules[_mod] = _mock.MagicMock()

# ── 2. Import project code ────────────────────────────────────────────────────
sys.path.insert(0, ".")
import torch.nn.functional as F
from model.model import DanceDecoder
from model.diffusion import GaussianDiffusion
from vis import SMPLSkeleton
from dataset.dance_dataset import preprocess_motion_to_tensor

# ── 3. Config ─────────────────────────────────────────────────────────────────
REPR_DIM    = 151
MUSIC_DIM   = 35          # baseline features (zeroed — no-music mode)
HORIZON     = 150         # 5 s × 30 fps
COND_DIM    = REPR_DIM + MUSIC_DIM  # 186

DATA_DIR    = "/Volumes/gm7000/EDGE/data/edge_aistpp/motions"
VAL_SPLIT   = "/Volumes/gm7000/EDGE/data/splits/duet_val.txt"
CKPT_PATH   = "checkpoint.pt"
OUT_DIR     = "smoke_infer_out"
N_LEAD      = 10           # how many val sequences to use as lead


def load_val_sequences(n=10):
    with open(VAL_SPLIT) as f:
        names = [l.strip() for l in f if l.strip()]
    # Take first n sequences spread across the val set
    step = max(1, len(names) // n)
    selected = names[::step][:n]
    print(f"Selected {len(selected)} val sequences:")
    for s in selected:
        print(f"  {s}")
    return selected


def preprocess_lead(seq_names, normalizer):
    """Load raw pkl, slice to first 300 raw frames (=150 @ 30fps), preprocess."""
    chunks = []
    for name in seq_names:
        path = os.path.join(DATA_DIR, name + ".pkl")
        data = pickle.load(open(path, "rb"))
        pos = data["smpl_trans"]   # [T_raw, 3]
        q   = data["smpl_poses"]   # [T_raw, 72]

        # Take first 300 raw frames → 150 frames after stride-2 downsampling
        T_needed = HORIZON * 2     # 300
        if pos.shape[0] < T_needed:
            # Pad by repeating last frame
            pad = T_needed - pos.shape[0]
            pos = np.concatenate([pos, np.tile(pos[-1:], (pad, 1))], axis=0)
            q   = np.concatenate([q,   np.tile(q[-1:],  (pad, 1))], axis=0)

        tensor = preprocess_motion_to_tensor(
            pos[:T_needed], q[:T_needed], normalizer
        )  # [T, 151] — T may be slightly off due to stride
        # Ensure exactly HORIZON frames
        if tensor.shape[0] > HORIZON:
            tensor = tensor[:HORIZON]
        elif tensor.shape[0] < HORIZON:
            pad_len = HORIZON - tensor.shape[0]
            tensor = torch.cat([tensor, tensor[-1:].expand(pad_len, -1)], dim=0)

        chunks.append(tensor)
        print(f"  {name}: preprocessed → {tensor.shape}")

    return torch.stack(chunks)  # [N, 150, 151]


def build_model(device):
    model = DanceDecoder(
        nfeats=REPR_DIM, seq_len=HORIZON, latent_dim=512, ff_size=1024,
        num_layers=8, num_heads=8, dropout=0.1,
        cond_feature_dim=COND_DIM,
        activation=F.gelu,
    )
    smpl = SMPLSkeleton(device)
    # guidance_weight=0: use only the unconditioned path (pretrained null_cond_embed).
    # With weight=2, CFG amplifies the randomly-initialized cond_projection output and
    # subtracts the good unconditioned signal — producing systematically inverted poses.
    # weight=0 lets the pretrained backbone generate valid motion without conditioning.
    diffusion = GaussianDiffusion(
        model, HORIZON, REPR_DIM, smpl,
        schedule="cosine", n_timestep=1000,
        predict_epsilon=False, loss_type="l2",
        use_p2=False, cond_drop_prob=0.25, guidance_weight=0,
    )
    return diffusion


def run():
    device = "cpu"
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print("EDGE Duet Inference Smoke Test (no-music mode)")
    print("=" * 60)

    # ── Load checkpoint normalizer ────────────────────────────────────────────
    print(f"\nLoading normalizer from {CKPT_PATH} ...")
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    normalizer = ckpt["normalizer"]
    print(f"  Normalizer type: {type(normalizer).__name__}")

    # ── Load & preprocess lead motion ─────────────────────────────────────────
    print(f"\nLoading {N_LEAD} lead motion sequences from val split ...")
    seq_names = load_val_sequences(N_LEAD)
    lead_chunks = preprocess_lead(seq_names, normalizer)
    print(f"\nlead_chunks shape: {lead_chunks.shape}  (N, T, 151)")

    # ── Build duet model ──────────────────────────────────────────────────────
    print(f"\nBuilding duet model (cond_dim={COND_DIM}) ...")
    diffusion = build_model(device)

    # Load solo checkpoint, skipping any key whose shape doesn't match the
    # duet model (cond_projection: 4800→512 in solo vs 186→512 here).
    solo_state = ckpt["ema_state_dict"]
    current_state = diffusion.model.state_dict()
    compatible = {
        k: v for k, v in solo_state.items()
        if k in current_state and v.shape == current_state[k].shape
    }
    skipped = [k for k in solo_state if k not in compatible]
    missing, unexpected = diffusion.model.load_state_dict(compatible, strict=False)
    print(f"  Loaded {len(compatible)} compatible keys; "
          f"skipped {len(skipped)} shape-mismatched key(s): {skipped[:3]}")
    diffusion.eval()

    # ── Generate follower motions one at a time ───────────────────────────────
    print(f"\nGenerating {N_LEAD} follower motions (DDIM 50 steps each) ...")
    results = []

    with torch.no_grad():
        for i, name in enumerate(seq_names):
            lead  = lead_chunks[i:i+1]                          # [1, 150, 151]
            music = torch.zeros(1, HORIZON, MUSIC_DIM)          # [1, 150, 35] zeros
            cond  = torch.cat([lead, music], dim=-1)            # [1, 150, 186]

            print(f"\n  [{i+1}/{N_LEAD}] {name}")
            print(f"    cond shape: {cond.shape}")

            shape = (1, HORIZON, REPR_DIM)
            sample = diffusion.ddim_sample(shape, cond)         # [1, 150, 151]

            assert not torch.isnan(sample).any(), "NaN in output!"
            print(f"    output shape: {sample.shape}  "
                  f"mean={sample.mean().item():.3f}  "
                  f"std={sample.std().item():.3f}")

            out_path = os.path.join(OUT_DIR, f"follower_{i:02d}_{name}.npy")
            np.save(out_path, sample.squeeze(0).numpy())        # [150, 151]
            results.append(out_path)

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"Done. {len(results)} follower motions saved to {OUT_DIR}/")
    print()
    print("Files (normalized 151-dim motion vectors, shape [150, 151]):")
    for p in results:
        print(f"  {os.path.basename(p)}")
    print()
    print("NOTE: Quality is meaningless — cond_projection was randomly")
    print("initialized because no duet checkpoint exists yet.")
    print("Pipeline correctness confirmed: preprocessing + DDIM sampling")
    print("ran end-to-end without errors.")
    print("=" * 60)


if __name__ == "__main__":
    run()
