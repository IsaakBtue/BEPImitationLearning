# Left<->Right motion mirror tool

**Date:** 2026-07-26

## Problem

The right side of the goalkeeper is not converging in training while the left side is.
Comparing the `left_far` and `right_far` AMP reference clips (`new_doublestep{left,right}_{short,,wide}_booster_t1.npz`,
used both as ghost-overlay and as the per-region AMP discriminator reference in
`goalkeeper_multidisc_amp_cfg.py`) shows the two sides are not kinematic mirrors of
each other:

| | trunk yaw range across the clip | swing |
|---|---|---|
| left_far (all 3 clips) | +9.46&deg; to +9.54&deg; | ~0.09&deg; (flat) |
| right_far (current) | -14.3&deg; to +9.5&deg; | ~23.7&deg; |

The left clips hold an almost perfectly constant trunk yaw for the whole step. The
right clips swing through ~24&deg; and end up yawed the *same sign* as left instead of
mirrored negative. This is independent of the deliberate +5&deg; CW bias rotation applied
to `right_far` in commit `db4bdf6` (a separate, intentional experiment, layered on top
of this pre-existing asymmetry). This asymmetry is a plausible concrete contributor to
right-side non-convergence: the AMP discriminator for `right_far` is scoring against a
reference motion with a large, wrong-direction trunk rotation baked in.

## Goal

1. A general, reusable script that mirrors any T1 21-DOF headless NPZ motion clip
   left<->right (works in either direction, since the transform is its own inverse).
2. Apply it to the 3 `left_far` clips (confirmed by the user, via the WithOverlay
   viewer, to look correct) and use the output to temporarily replace the 3
   `right_far` clips, so a training run can test whether fixing this asymmetry helps
   right-side convergence.
3. Back up the current `right_far` clips first. (They're already fully pushed to
   `origin/v2-blue-ball-waypoint` at HEAD `ece0247`, recoverable via
   `git checkout ece0247 -- <path>`; an additional labeled on-disk copy is made for
   convenience.)
4. Document the tool's location in `CLAUDE.md` and log this fix in `docs/BugFixes.md`.

## Mirror math

NPZ schema (from `pkl_to_npz.py`): `fps` (scalar), `joint_pos`/`joint_vel` (T,21),
`body_pos_w`/`body_quat_w`/`body_lin_vel_w`/`body_ang_vel_w` (T, n_bodies, 3 or 4),
with body index 0 = Trunk (root).

Robot convention: X=forward, Y=lateral(left), Z=up. Mirror plane = XZ (world Y
negates). Derived by conjugating each quantity's transform matrix/pseudovector rule
by `R=diag(1,-1,1)`:

- **Root position** (`body_pos_w[:,0,:]`): `(x,y,z) -> (x,-y,z)`.
- **Root quaternion** (`body_quat_w[:,0,:]`, wxyz): `(w,x,y,z) -> (w,-x,y,-z)`
  (derived from `q' = n*q*n` with `n=(0,0,1,0)` the pure quaternion for the Y axis;
  sign-equivalent to `(-w,x,-y,z)` since q and -q represent the same rotation).
- **Joint angles**: per-joint rule based on rotation axis (from `t1_headless.xml`):
  - `*_Pitch` (axis Y): value unchanged, swap Left_X <-> Right_X slots.
  - `*_Roll` (axis X): negate value, swap Left_X <-> Right_X slots.
  - `*_Yaw` (axis Z): negate value, swap Left_X <-> Right_X slots.
  - `Waist` (axis Z, no L/R pair): negate value in place.
  - Cross-checked against `t1_constants.py`'s `HOME_KEYFRAME` comment (documents the
    same Roll/Yaw-negate, Pitch-unchanged convention for a mirror-symmetric standing
    pose) and against the sign-mirrored joint ranges in the XML (e.g.
    `Left_Hip_Roll` range `(-0.2,1.57)` vs `Right_Hip_Roll` `(-1.57,0.2)`).
  - Full 21-slot mapping (index: name -> mirror index, sign):
    ```
    0 Left_Shoulder_Pitch  <-> 4 Right_Shoulder_Pitch   (pitch, +)
    1 Left_Shoulder_Roll   <-> 5 Right_Shoulder_Roll    (roll,  -)
    2 Left_Elbow_Pitch     <-> 6 Right_Elbow_Pitch      (pitch, +)
    3 Left_Elbow_Yaw       <-> 7 Right_Elbow_Yaw        (yaw,   -)
    8 Waist                <-> 8 (self)                 (yaw,   -)
    9 Left_Hip_Pitch       <-> 15 Right_Hip_Pitch       (pitch, +)
    10 Left_Hip_Roll       <-> 16 Right_Hip_Roll        (roll,  -)
    11 Left_Hip_Yaw        <-> 17 Right_Hip_Yaw         (yaw,   -)
    12 Left_Knee_Pitch     <-> 18 Right_Knee_Pitch      (pitch, +)
    13 Left_Ankle_Pitch    <-> 19 Right_Ankle_Pitch     (pitch, +)
    14 Left_Ankle_Roll     <-> 20 Right_Ankle_Roll      (roll,  -)
    ```
- **joint_vel**: same slot-swap + sign rule as `joint_pos` (linear operation,
  commutes with finite differencing).
- **All-body kinematics** (`body_pos_w`, `body_quat_w`, `body_lin_vel_w`,
  `body_ang_vel_w`, every body incl. feet): recomputed from scratch via forward
  kinematics using the mirrored root pose + mirrored `joint_pos` at each frame,
  reusing `pkl_to_npz.py`'s own `mujoco.mj_kinematics` loop, then
  `_finite_diff_vel`/`_quat_ang_vel` for velocities. This is safer than hand-mirroring
  per-body arrays (would require manually swapping e.g. `left_foot_link` <->
  `right_foot_link` body indices) and guarantees byte-for-byte the same derivation
  pipeline as every other NPZ in the dataset.

## Implementation

- **New file:** `src/simple_goalkeeper/scripts/mirror_motion.py`
  - `mirror_npz(input_path: Path, output_path: Path) -> None`
  - `main()` / CLI: `uv run sgk_mirror <input.npz> <output.npz>`
  - Imports `_finite_diff_vel`, `_quat_ang_vel`, `_quat_mul_wxyz` from
    `pkl_to_npz.py` rather than duplicating them.
- **New entry point** in `pyproject.toml`: `sgk_mirror = "simple_goalkeeper.scripts.mirror_motion:cli"`.
- **Backup:** copy current `new_doublestepright_{short,,wide}_booster_t1.npz` to
  `src/simple_goalkeeper/motions/backups/2026-07-26-pre-mirror-right_far/` before
  overwriting.
- **Swap:** run `sgk_mirror` on the 3 `left_far` files, write output directly over
  the 3 `right_far` filenames in `motions/data/`. Left as an **uncommitted**
  working-tree change (per user preference) so it's trivially reversible with
  `git checkout -- <path>`.
- **Verification:** re-run the yaw/bearing measurement used to diagnose the problem
  against the new right_far files and confirm the trunk yaw is now flat (mirrors
  left_far's ~9.5&deg; constant, sign-flipped) instead of swinging ~24&deg;.
- **Docs:** add `mirror_motion.py` to `CLAUDE.md`'s Key Files table; log this fix
  (problem, root cause, fix, verification numbers) as a dated entry in
  `docs/BugFixes.md`.

## Out of scope

- Not fixing `left_far`'s own previously-flagged, never-applied rotation-drift
  correction (per user instruction: mirror left_far as-is, since it's the side
  that's actually converging).
- Not touching the `left_near`/`right_near` regions or any other motion pool.
- Not making the swap permanent/committed (explicit user choice: leave uncommitted).
