"""Time-compress already-converted NPZ motion clips (NPZ → NPZ), for AMP reference-pace
augmentation.

Unlike pkl_to_npz.py's speed_factor (which compresses from raw PKL mocap), this operates
directly on already-converted NPZ clips -- reusing their already-correct frame alignment
(z-shift, ±90° yaw rotation from pkl_to_npz.py's convert_one) instead of re-deriving it from
raw PKL data (not all source PKLs are still present locally). Resampling stays at the SAME
output fps as the input, so joint_vel/body_lin_vel_w/body_ang_vel_w are recomputed via
finite-differencing the resampled (shorter) sequence -- never scaled from the original -- which
keeps every reference transition self-consistent with the sim's own dt for the AMP discriminator.

2026-07-12: added to close a physically-verified timing gap -- wide-crossing ball-flight
windows (0.58-1.01s at full difficulty) are shorter than the DoubleStep/TripleStep reference
clips' 1.44s duration. See docs/BugFixes.md for the full investigation. Compression factor
capped at 1.5x per literature precedent (FARM, arXiv:2508.19926, reports naturalness/learning
degradation beyond ~1.5x for this exact kind of clip time-compression augmentation).

Usage:
    uv run sgk_retime --speed-factor 1.5 --suffix 1p5x \
        --names LeftDoubleStep_own_booster_t1 RightDoubleStep_own_booster_t1 \
                LeftTripleStep_own_booster_t1 RightTripleStep_own_booster_t1
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tyro

from .pkl_to_npz import _finite_diff_vel, _quat_ang_vel

_HERE = Path(__file__).parent


def retime_one(input_path: Path, output_path: Path, speed_factor: float) -> None:
    """Time-compress one already-converted NPZ clip by speed_factor (>1 = faster/shorter)."""
    data = np.load(input_path)
    fps = float(data["fps"])
    joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)      # (T, 21)
    body_pos_w = np.asarray(data["body_pos_w"], dtype=np.float32)    # (T, B, 3)
    body_quat_w = np.asarray(data["body_quat_w"], dtype=np.float32)  # (T, B, 4)

    T_in = joint_pos.shape[0]
    duration = (T_in - 1) / fps
    compressed_duration = duration / speed_factor
    t_in = np.linspace(0, duration, T_in)
    t_out = np.arange(0, compressed_duration, 1.0 / fps)
    T_out = len(t_out)

    def resample(arr: np.ndarray) -> np.ndarray:
        flat = arr.reshape(T_in, -1)
        out = np.zeros((T_out, flat.shape[1]), dtype=np.float32)
        for j in range(flat.shape[1]):
            out[:, j] = np.interp(t_out, t_in, flat[:, j])
        return out.reshape((T_out,) + arr.shape[1:])

    joint_pos_r = resample(joint_pos)
    body_pos_w_r = resample(body_pos_w)
    body_quat_w_r = resample(body_quat_w)
    # nlerp (linear-interp components + renormalize) -- matches the same approximation
    # pkl_to_npz.py's convert_one already uses for root_rot_r. Adjacent motion-capture
    # frames are close enough in orientation for this to be an acceptable stand-in for
    # slerp, and it keeps this tool numerically consistent with the existing pipeline.
    body_quat_w_r /= np.linalg.norm(body_quat_w_r, axis=-1, keepdims=True).clip(min=1e-8)

    dt = 1.0 / fps
    joint_vel_r = _finite_diff_vel(joint_pos_r, dt)
    body_lin_vel_w_r = _finite_diff_vel(body_pos_w_r, dt)
    body_ang_vel_w_r = np.zeros_like(body_lin_vel_w_r)
    for b in range(body_quat_w_r.shape[1]):
        body_ang_vel_w_r[:, b, :] = _quat_ang_vel(body_quat_w_r[:, b, :], dt)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(output_path),
        fps=np.array(fps),
        joint_pos=joint_pos_r,
        joint_vel=joint_vel_r,
        body_pos_w=body_pos_w_r,
        body_quat_w=body_quat_w_r,
        body_lin_vel_w=body_lin_vel_w_r,
        body_ang_vel_w=body_ang_vel_w_r,
    )
    peak_joint_vel = float(np.abs(joint_vel_r).max())
    print(
        f"  {input_path.stem} -> {output_path.name}  "
        f"{T_in}->{T_out} frames, {duration:.3f}s->{compressed_duration:.3f}s, "
        f"peak_joint_vel={peak_joint_vel:.2f} rad/s"
    )


def main(
    input_dir: str = str(_HERE.parent / "motions" / "data"),
    output_dir: str = str(_HERE.parent / "motions" / "data"),
    speed_factor: float = 1.5,
    suffix: str = "1p5x",
    names: list[str] | None = None,
) -> None:
    """Retime *.npz files in input_dir, writing <name>_<suffix>.npz to output_dir.

    names: optional list of NPZ stems to retime (e.g. LeftDoubleStep_own_booster_t1).
           If omitted, all *.npz files in input_dir are retimed.
    """
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    npz_files = sorted(p for p in in_dir.glob("*.npz") if not p.stem.endswith(f"_{suffix}"))
    if names is not None:
        name_set = set(names)
        npz_files = [p for p in npz_files if p.stem in name_set]
        if not npz_files:
            raise FileNotFoundError(f"None of {name_set} found in {in_dir}")
    print(f"Retiming {len(npz_files)} files x{speed_factor} -> {out_dir}")
    for npz in npz_files:
        retime_one(npz, out_dir / f"{npz.stem}_{suffix}.npz", speed_factor)
    print("Done.")


def cli() -> None:
    tyro.cli(main)


if __name__ == "__main__":
    cli()
