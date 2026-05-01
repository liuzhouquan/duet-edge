"""
2-D stick-figure duet renderer — macOS, no pytorch3d/wandb needed.

Reads the 10 follower motions from smoke_infer_out/ and the matching lead
motions from the AIST++ val split.  Renders each pair as an animated GIF:
  • lead   dancer — royalblue,  shown on the LEFT
  • follower dancer — tomato,     shown on the RIGHT

The 3-D joint positions are projected to a front view (X horizontal, Z vertical).
Each dancer is centred on their own root so side-by-side layout is stable.

Output: smoke_infer_out/render_XX_<name>.gif  (10 GIFs, ~5 s each at 30 fps)

Run from EDGE project root:
    conda run -n edge-mac python smoke_render_duet.py
"""

import sys, types, os, glob, re, pickle
import numpy as np
import torch

# ── 1. Stub only the non-rendering libraries ──────────────────────────────────
# matplotlib is NOT mocked here — we need it for actual rendering.

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
    qv  = q[..., 1:]
    uv  = torch.cross(qv, v, dim=-1)
    uuv = torch.cross(qv, uv, dim=-1)
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
    c = angle.cos().squeeze(-1)
    s = angle.sin().squeeze(-1)
    t = 1 - c
    x, y, z = axis.unbind(-1)
    return torch.stack([
        torch.stack([t*x*x+c,   t*x*y-s*z, t*x*z+s*y], dim=-1),
        torch.stack([t*x*y+s*z, t*y*y+c,   t*y*z-s*x], dim=-1),
        torch.stack([t*x*z-s*y, t*y*z+s*x, t*z*z+c  ], dim=-1),
    ], dim=-2)

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

# ── 2. Now import project code (matplotlib is real, not mocked) ───────────────
sys.path.insert(0, ".")

import matplotlib
matplotlib.use("Agg")  # headless backend for saving files
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D

from vis import SMPLSkeleton, smpl_parents
from dataset.dance_dataset import preprocess_motion_to_tensor
from dataset.quaternion import ax_from_6v

# ── 3. Paths & constants ──────────────────────────────────────────────────────
DATA_DIR   = "/Volumes/gm7000/EDGE/data/edge_aistpp/motions"
VAL_SPLIT  = "/Volumes/gm7000/EDGE/data/splits/duet_val.txt"
CKPT_PATH  = "checkpoint.pt"
OUT_DIR    = "smoke_infer_out"
INFER_DIR  = OUT_DIR

REPR_DIM   = 151
HORIZON    = 150
FPS        = 30

# SMPL 24-joint parent list — build edge list for drawing
EDGES = [(i, p) for i, p in enumerate(smpl_parents) if p != -1]


# ── 4. FK helper ─────────────────────────────────────────────────────────────
def motion_to_joints(vec_norm, normalizer, smpl):
    """
    vec_norm: [T, 151] normalised motion tensor
    Returns: np.ndarray [T, 24, 3]  joint world positions
    """
    vec = normalizer.unnormalize(vec_norm.unsqueeze(0))  # [1, T, 151]
    motion = vec[0, :, 4:]                               # strip contacts → [T, 147]
    pos = motion[:, :3]                                  # [T, 3]
    q6d = motion[:, 3:].reshape(HORIZON, 24, 6)          # [T, 24, 6]
    q_aa = ax_from_6v(q6d)                               # [T, 24, 3]
    joints = smpl.forward(
        q_aa.unsqueeze(0), pos.unsqueeze(0)              # [1,T,24,3], [1,T,3]
    )                                                    # → [1, T, 24, 3]
    return joints[0].detach().cpu().numpy()              # [T, 24, 3]


# ── 5. 2-D projection & centering ────────────────────────────────────────────
def project_front(joints_3d, x_offset=0.0):
    """
    Front-view projection: keep X (sideways) and Z (up).
    Per-frame: subtract root X so the dancer stays horizontally fixed.
    Vertically: subtract frame-0 root Z so the floor starts at 0 but
    jumps / bends are preserved.
    joints_3d: [T, 24, 3]
    Returns: [T, 24, 2]  (x, z in metres)
    """
    xy = joints_3d[:, :, [0, 2]].copy()   # [T, 24, 2]: x, z
    # Remove per-frame root horizontal drift
    root_x = xy[:, 0, 0:1]               # [T, 1]
    xy[:, :, 0] -= root_x               # all joints stay relative to root x
    # Remove initial vertical offset (keep vertical motion)
    xy[:, :, 1] -= xy[0, 0, 1]
    # Place dancer at designated side-by-side position
    xy[:, :, 0] += x_offset
    return xy


# ── 6. GIF renderer ──────────────────────────────────────────────────────────
def render_pair_gif(joints_lead, joints_follow, out_path,
                    color_lead="royalblue", color_follow="tomato",
                    gap=0.6, fps=FPS):
    """
    joints_lead, joints_follow: [T, 24, 2]  projected 2-D coords
    """
    T = joints_lead.shape[0]

    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    ax.set_facecolor("#1a1a1a")
    fig.patch.set_facecolor("#1a1a1a")
    ax.set_aspect("equal")
    ax.axis("off")

    # Pre-compute plot bounds (fixed across all frames for stable view)
    all_xy = np.concatenate([joints_lead, joints_follow], axis=1)  # [T, 48, 2]
    pad = 0.15
    x_min, x_max = all_xy[:, :, 0].min() - pad, all_xy[:, :, 0].max() + pad
    z_min, z_max = all_xy[:, :, 1].min() - pad, all_xy[:, :, 1].max() + pad
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(z_min, z_max)

    # Create line artists for each bone, both dancers
    lead_lines   = [ax.plot([], [], lw=2,   color=color_lead,   solid_capstyle="round")[0] for _ in EDGES]
    follow_lines = [ax.plot([], [], lw=2,   color=color_follow, solid_capstyle="round")[0] for _ in EDGES]
    lead_dots    =  ax.plot([], [], "o", ms=3, color=color_lead)[0]
    follow_dots  =  ax.plot([], [], "o", ms=3, color=color_follow)[0]

    # Legend
    legend_elements = [
        Line2D([0], [0], color=color_lead,   lw=2, label="Lead"),
        Line2D([0], [0], color=color_follow, lw=2, label="Follower"),
    ]
    ax.legend(handles=legend_elements, loc="upper right",
              facecolor="#333", labelcolor="white", fontsize=8, framealpha=0.7)

    def _draw(frame_idx, joints, lines):
        j = joints[frame_idx]          # [24, 2]
        for line, (i, p) in zip(lines, EDGES):
            line.set_data([j[p, 0], j[i, 0]], [j[p, 1], j[i, 1]])

    def animate(t):
        _draw(t, joints_lead,   lead_lines)
        _draw(t, joints_follow, follow_lines)
        lead_dots.set_data(joints_lead[t, :, 0],   joints_lead[t, :, 1])
        follow_dots.set_data(joints_follow[t, :, 0], joints_follow[t, :, 1])
        return lead_lines + follow_lines + [lead_dots, follow_dots]

    anim = animation.FuncAnimation(
        fig, animate, frames=T, interval=1000 / fps, blit=True
    )
    writer = animation.PillowWriter(fps=fps)
    anim.save(out_path, writer=writer)
    plt.close(fig)
    print(f"  → {out_path}")


# ── 7. Main ───────────────────────────────────────────────────────────────────
def run():
    print("=" * 60)
    print("Duet 2-D Stick-Figure Renderer")
    print("=" * 60)

    # Load checkpoint normalizer
    ckpt       = torch.load(CKPT_PATH, map_location="cpu")
    normalizer = ckpt["normalizer"]
    smpl       = SMPLSkeleton()

    # Discover inferred follower files (sorted by index)
    follower_files = sorted(glob.glob(os.path.join(INFER_DIR, "follower_*.npy")))
    if not follower_files:
        print(f"No follower_*.npy files found in {INFER_DIR}/")
        return

    # Re-build the same val sequence selection used in smoke_infer_duet.py
    with open(VAL_SPLIT) as f:
        val_names = [l.strip() for l in f if l.strip()]
    n = len(follower_files)
    step = max(1, len(val_names) // n)
    lead_names = val_names[::step][:n]

    print(f"\nRendering {n} lead+follower pairs as GIFs ...\n")

    for idx, (fpath, lead_name) in enumerate(zip(follower_files, lead_names)):
        print(f"[{idx+1}/{n}]  lead={lead_name}")

        # ── Load lead motion ──────────────────────────────────────────────────
        pkl_path = os.path.join(DATA_DIR, lead_name + ".pkl")
        data     = pickle.load(open(pkl_path, "rb"))
        pos_raw  = data["smpl_trans"]
        q_raw    = data["smpl_poses"]
        T_needed = HORIZON * 2
        if pos_raw.shape[0] < T_needed:
            pad = T_needed - pos_raw.shape[0]
            pos_raw = np.concatenate([pos_raw, np.tile(pos_raw[-1:], (pad, 1))], 0)
            q_raw   = np.concatenate([q_raw,   np.tile(q_raw[-1:],  (pad, 1))], 0)
        lead_norm = preprocess_motion_to_tensor(
            pos_raw[:T_needed], q_raw[:T_needed], normalizer
        )
        if lead_norm.shape[0] > HORIZON:
            lead_norm = lead_norm[:HORIZON]
        elif lead_norm.shape[0] < HORIZON:
            lead_norm = torch.cat(
                [lead_norm, lead_norm[-1:].expand(HORIZON - lead_norm.shape[0], -1)], 0
            )

        # ── Load follower motion ──────────────────────────────────────────────
        follow_norm = torch.from_numpy(np.load(fpath)).float()  # [150, 151]

        # ── Forward kinematics → 3-D joint positions ─────────────────────────
        with torch.no_grad():
            j_lead   = motion_to_joints(lead_norm,   normalizer, smpl)  # [T,24,3]
            j_follow = motion_to_joints(follow_norm, normalizer, smpl)  # [T,24,3]

        # ── Project to 2-D and apply side-by-side offset ─────────────────────
        # Compute a natural gap: use the lead's shoulder width as reference
        shoulder_span = float(np.abs(j_lead[:, 16, 0] - j_lead[:, 17, 0]).mean())
        gap = max(0.6, shoulder_span * 3)

        xy_lead   = project_front(j_lead,   x_offset=-gap / 2)
        xy_follow = project_front(j_follow, x_offset=+gap / 2)

        # ── Render GIF ────────────────────────────────────────────────────────
        out_name = re.sub(r"follower_(\d+)_", r"render_\1_", os.path.basename(fpath))
        out_name = out_name.replace(".npy", ".gif")
        out_path = os.path.join(OUT_DIR, out_name)
        render_pair_gif(xy_lead, xy_follow, out_path)

    print()
    print("=" * 60)
    print(f"Done. {n} GIFs saved to {OUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    run()
