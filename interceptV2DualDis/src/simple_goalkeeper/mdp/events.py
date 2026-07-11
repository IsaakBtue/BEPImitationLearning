"""Event functions and curriculum helpers for SimpleGoalKeeper."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply, sample_uniform

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.curriculum_manager import CurriculumTermCfg

_MOTIONS_DIR = Path(__file__).parents[1] / "motions" / "data"
_DEFAULT_ROBOT_CFG = SceneEntityCfg("robot")


# Easy (difficulty=0.0) spawn ranges: short distance, centred, slow, low.
# Hard (difficulty=1.0) ranges are passed as params to reset_ball_local_frame.
_EASY_DIST    = (2.5, 3.5)
_EASY_Y_START = (-0.05, 0.05)  # lateral spawn: very centred on easy (hard = ±1.0 m via params)
_EASY_Y_END   = (-0.05, 0.05)  # goal target Y: dead centre on easy
_EASY_Z_START = (0.1, 0.25)
_EASY_Z_END   = (0.05, 0.15)
_EASY_SPEED   = (1.0, 1.5)     # slow on easy → longer t_flight, more reaction time


def _lerp_range(
    easy: tuple[float, float],
    hard: tuple[float, float],
    d: float,
) -> tuple[float, float]:
    """Linearly interpolate a (lo, hi) range between easy and hard endpoints."""
    return (
        easy[0] + d * (hard[0] - easy[0]),
        easy[1] + d * (hard[1] - easy[1]),
    )


def _yaw_only_quat(q_wxyz: torch.Tensor) -> torch.Tensor:
    """Return a quaternion containing ONLY the yaw component of q_wxyz.

    Isolating yaw ensures quat_apply only rotates XY — pitch/roll on the robot
    body does not tilt the ball spawn position or velocity.
    """
    w, x, y, z = q_wxyz[:, 0], q_wxyz[:, 1], q_wxyz[:, 2], q_wxyz[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    half = yaw * 0.5
    out = torch.zeros_like(q_wxyz)
    out[:, 0] = torch.cos(half)   # w
    out[:, 3] = torch.sin(half)   # z
    return out


# Z offset baked into NPZ files (2026-06-30): body_pos_w Z was shifted +0.030 m
# in all NPZ files so the foot capsule contact surface sits at floor level.
# No runtime correction needed here anymore.

# Lateral crossing-Y thresholds for 4-tier distance-conditioned RSI.
_SINGLE_THRESH = 0.20   # |cross_y| < 0.20          → single  (standing reset)
_DOUBLE_THRESH = 0.40   # 0.20 ≤ |cross_y| < 0.40   → double  (MediumStep, SafeMedium)
_TRIPLE_THRESH = 0.60   # 0.40 ≤ |cross_y| < 0.60   → triple  (FarStep, SafeFar)
                        # |cross_y| ≥ 0.60           → wide    (DoubleStep, TripleStep)

# Exact filename stem (lowercased) → (side, pool) mapping.
# Name-specific so adding new files never silently mis-classifies.
_STEM_TO_POOL: dict[str, tuple[str, str]] = {
    # double  (0.20–0.40 m, see _DOUBLE_THRESH above)
    "leftsafemedium1_booster_t1":     ("left",  "double"),
    "rightsafemedium1_booster_t1":    ("right", "double"),
    # triple  (0.40–0.60 m, see _TRIPLE_THRESH above)
    "leftsafefar1_booster_t1":        ("left",  "triple"),
    "rightsafefar1_booster_t1":       ("right", "triple"),
    # wide    (≥ 0.60 m, see _TRIPLE_THRESH above)
    "leftdoublestep_own_booster_t1":  ("left",  "wide"),
    "lefttriplestep_own_booster_t1":  ("left",  "wide"),
    "rightdoublestep_own_booster_t1": ("right", "wide"),
    "righttriplestep_own_booster_t1": ("right", "wide"),
    # 2026-07-12: 1.5x-retimed variants (retime_motion.py) of the same four
    # clips, same (side, wide) pool -- _load_pool computes frame_frac per
    # SOURCE FILE, so seed_blue_landed_practice's "late half" draw now pulls
    # from either pace. See docs/BugFixes.md for the timing-gap rationale.
    "leftdoublestep_own_booster_t1_1p5x":  ("left",  "wide"),
    "lefttriplestep_own_booster_t1_1p5x":  ("left",  "wide"),
    "rightdoublestep_own_booster_t1_1p5x": ("right", "wide"),
    "righttriplestep_own_booster_t1_1p5x": ("right", "wide"),
    # 2026-07-12 (same day): 2x-retimed variants too, same treatment.
    "leftdoublestep_own_booster_t1_2x":  ("left",  "wide"),
    "lefttriplestep_own_booster_t1_2x":  ("left",  "wide"),
    "rightdoublestep_own_booster_t1_2x": ("right", "wide"),
    "righttriplestep_own_booster_t1_2x": ("right", "wide"),
    # single-range files (< 0.20 m, see _SINGLE_THRESH above) → standing pose,
    # not RSI pools; listed so the init loop doesn't warn about unknown files.
    "leftstep_own_booster_t1":        None,
    "leftsafe1_booster_t1":           None,
    "leftsafefront1_booster_t1":      None,
    "rightstep_own_booster_t1":       None,
    "rightsafe1_booster_t1":          None,
    "rightsafefront1_booster_t1":     None,
}


def _load_pool(files: list, dev: str) -> dict[str, torch.Tensor]:
    """Concatenate NPZ frames from a list of files into one pool dict.

    Also computes `frame_frac`: this frame's position within ITS OWN source
    file, 0.0 (first frame) to 1.0 (last frame) -- reset per file, not global
    across the concatenated pool. Added 2026-07-11 so callers can bias
    sampling toward the back half of a clip (e.g. the post-plant/push-off
    portion of a DoubleStep/TripleStep demonstration) without needing
    per-file frame-range bookkeeping of their own. See
    MotionResetManager._seed_blue_landed_practice.
    """
    arrays: dict[str, list] = {k: [] for k in
                                ("joint_pos", "joint_vel", "root_pos",
                                 "root_quat", "root_lin_vel", "root_ang_vel")}
    fracs: list = []
    for f in files:
        data = np.load(str(f))
        arrays["joint_pos"].append(data["joint_pos"])
        arrays["joint_vel"].append(data["joint_vel"])
        arrays["root_pos"].append(data["body_pos_w"][:, 0, :])
        arrays["root_quat"].append(data["body_quat_w"][:, 0, :])
        arrays["root_lin_vel"].append(data["body_lin_vel_w"][:, 0, :])
        arrays["root_ang_vel"].append(data["body_ang_vel_w"][:, 0, :])
        n_frames = data["joint_pos"].shape[0]
        fracs.append(np.linspace(0.0, 1.0, n_frames, dtype=np.float32) if n_frames > 1
                     else np.zeros(n_frames, dtype=np.float32))
    out = {k: torch.from_numpy(np.vstack(v)).float().to(dev) for k, v in arrays.items()}
    out["frame_frac"] = torch.from_numpy(np.concatenate(fracs)).float().to(dev)
    return out


class MotionResetManager:
    """Loads NPZ motion frames into per-tier pools and the combined pool.

    `reset()` (registered as reset_from_motion_data) is a literal port of
    Humanoid-Goalkeeper G1's continue_keep mechanism (legged_gym/legged_gym/
    envs/base/legged_robot.py:657-682, _reset_dofs) — see docs/superpowers/
    plans/2026-07-01-live-env-rsi.md for the full comparison. It does not use
    the tier pools built here at all; those exist only for the sgk_play_rsi
    diagnostic script (scripts/play_rsi_doublestep.py), which locks playback
    to a specific motion pool for visual inspection.
    """

    _instance: "MotionResetManager | None" = None

    def __init__(self) -> None:
        # Combined pool (fallback / AMP reference compatibility)
        self.frames: dict[str, torch.Tensor] = {}
        # Per-type pools keyed by (side, steps): side ∈ {left, right}, steps ∈ {single, double, triple, wide}
        self.pools: dict[tuple[str, str], dict[str, torch.Tensor]] = {}

    @classmethod
    def get(cls) -> "MotionResetManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def init(self, env: "ManagerBasedRlEnv") -> None:
        """Load NPZ files into 8 per-type pools + one combined fallback pool."""
        if "joint_pos" in self.frames:
            return

        motion_files = sorted(_MOTIONS_DIR.glob("*.npz"))
        if not motion_files:
            raise FileNotFoundError(f"No NPZ motion files in {_MOTIONS_DIR}")

        dev = env.device

        bucket: dict[tuple[str, str], list] = {
            ("left",  "double"): [], ("left",  "triple"): [], ("left",  "wide"): [],
            ("right", "double"): [], ("right", "triple"): [], ("right", "wide"): [],
        }
        all_files = []
        for f in motion_files:
            stem = f.stem.lower()
            if stem not in _STEM_TO_POOL:
                print(f"[MotionResetManager] WARNING: {f.name} not in _STEM_TO_POOL — added to combined pool only")
                all_files.append(f)
                continue
            key = _STEM_TO_POOL[stem]
            all_files.append(f)
            if key is not None:   # None = single-range, goes to combined pool only
                bucket[key].append(f)

        for key, files in bucket.items():
            if files:
                self.pools[key] = _load_pool(files, dev)
                n = self.pools[key]["joint_pos"].shape[0]
                print(f"[MotionResetManager] {key}: {n} frames from {len(files)} file(s)")
            else:
                print(f"[MotionResetManager] WARNING: no files for pool {key}")

        self.frames = _load_pool(all_files, dev)
        print(f"[MotionResetManager] Combined pool: {self.frames['joint_pos'].shape[0]} frames")

    def _write_rsi_state(
        self,
        env: "ManagerBasedRlEnv",
        ids: torch.Tensor,
        pool: dict[str, torch.Tensor],
        robot: "Entity",
        frame_ids: torch.Tensor | None = None,
    ) -> None:
        """Write root pose + velocity + joint state from a frame in pool.

        frame_ids: explicit per-env frame indices into `pool` (added
        2026-07-11 for _seed_blue_landed_practice, which needs to bias
        sampling toward late-clip frames). Defaults to uniform random over
        the whole pool, matching original behavior, when not provided.
        """
        n = len(ids)
        if frame_ids is None:
            frame_ids = torch.randint(0, pool["joint_pos"].shape[0], (n,), device=env.device)

        root_pos  = pool["root_pos"][frame_ids]
        root_quat = pool["root_quat"][frame_ids]
        positions = env.scene.env_origins[ids].clone()
        positions[:, 2] = root_pos[:, 2]  # Z offset baked into NPZ (2026-06-30)
        robot.write_root_link_pose_to_sim(
            torch.cat([positions, root_quat], dim=-1), env_ids=ids
        )

        robot.write_root_link_velocity_to_sim(
            torch.cat([pool["root_lin_vel"][frame_ids],
                       pool["root_ang_vel"][frame_ids]], dim=-1),
            env_ids=ids,
        )

        joint_pos = pool["joint_pos"][frame_ids].clone()
        joint_vel = pool["joint_vel"][frame_ids].clone()
        limits = robot.data.soft_joint_pos_limits[ids]
        joint_pos.clamp_(limits[..., 0], limits[..., 1])
        robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=ids)

    def seed_blue_landed_practice(
        self,
        env: "ManagerBasedRlEnv",
        env_ids: torch.Tensor,
        asset_cfg: SceneEntityCfg,
        fraction: float,
    ) -> None:
        """FEAT 2026-07-11: for a fraction of newly-reset WIDE-crossing envs,
        seed the robot's starting state directly from a LATE-clip frame
        (frame_frac >= 0.5) of the appropriate side's "wide" pool -- the real
        LeftDoubleStep/RightDoubleStep/*TripleStep demonstration clips
        (already 4x-weighted in AMP sampling, see goalkeeper_amp_cfg.py
        _DOUBLE_TRIPLE_STEP_WEIGHT) -- rather than a static default pose or a
        donor borrowed from another live env's current (possibly
        already-failed) state, as blue_practice_fraction already does
        (mgr.reset's own docstring).

        Rationale: after ~10 fixes targeting the approach-and-plant mechanism
        itself, genuine blue landing rate was still ~0.1-3% at saturated
        ball_difficulty (docs/BugFixes.md, 2026-07-11 escalation). Even if a
        landing is eventually discovered, the policy has essentially never
        practiced what comes AFTER it (push off toward the real target,
        intercept, recover) -- footreach/foot_proximity/stopball/softstop's
        post-landing behavior for wide crossings has had almost no training
        exposure at all. Seeding directly from real demonstration data
        (rather than a live-env donor, which offers no guarantee of
        representing a genuinely successful state) gives dense, cheap
        exposure to that follow-through phase independent of whether the
        approach-and-plant sub-problem is solved yet, while a SEPARATE
        mechanism (rewards.py FIX 2026-07-11: footreach vel_sigma decay near
        blue) targets the approach itself.

        Only affects envs whose PERMANENT region assignment (env._region_id,
        set once at startup by assign_static_regions) is a "far" region --
        authoritative over any per-episode wide/narrow classification, same
        convention as rewards._get_reach_target_y's FIX 2026-07-07. Selected
        envs get env._blue_seed_landed_pending SET (and every other env in
        env_ids gets it explicitly CLEARED first, since this is a per-episode
        flag that must not leak from a previous episode) --
        rewards._get_reach_target_y CHECKS but never clears this flag itself
        (it's read-only there): _get_reach_target_y is called up to 7 times
        per real step, and its own reset-detection (episode_length_buf <= 1)
        stays true across all of them, so if it cleared the flag on the
        first call, the remaining 6 calls that same step would see it already
        gone and re-zero the seed they'd just set -- the flag's owner
        (this method) is the only thing allowed to write it, exactly once
        per real reset. When set, marks env._blue_landed=True,
        env._blue_landed_was_free=True (deliberately -- this landing was NOT
        policy-caused, so stopball/softstop's landing_ok gate must still
        treat it as free/ineligible, exactly the anti-farming purpose
        _blue_landed_was_free already exists for; footreach/foot_proximity's
        dense proximity rewards have no such farming concern and DO benefit
        from the unlocked post-landing target).
        """
        region_id = getattr(env, "_region_id", None)
        if not hasattr(env, "_blue_seed_landed_pending"):
            env._blue_seed_landed_pending = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        if region_id is None or len(env_ids) == 0:
            return
        # Clear first -- every env in this reset batch starts a fresh
        # episode, so any leftover pending flag from a past episode must not
        # survive (only the "selected" subset below gets it re-set).
        env._blue_seed_landed_pending[env_ids.long()] = False
        rid = region_id[env_ids]
        is_far = (rid == 1) | (rid == 3)   # left_far=1, right_far=3 (REGION_NAMES order)
        is_left = (rid == 0) | (rid == 1)  # left_near=0, left_far=1
        far_ids = env_ids[is_far]
        if len(far_ids) == 0:
            return
        select_mask = torch.rand(len(far_ids), device=env.device) < fraction
        selected = far_ids[select_mask]
        if len(selected) == 0:
            return
        selected_is_left = is_left[is_far][select_mask]

        robot: Entity = env.scene[asset_cfg.name]
        if not hasattr(env, "_blue_seed_landed_pending"):
            env._blue_seed_landed_pending = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

        for side, side_mask in (("left", selected_is_left), ("right", ~selected_is_left)):
            ids = selected[side_mask]
            if len(ids) == 0:
                continue
            pool = self.pools.get((side, "wide"))
            if pool is None:
                continue
            late = torch.nonzero(pool["frame_frac"] >= 0.5, as_tuple=False).flatten()
            if len(late) == 0:
                continue
            local_idx = torch.randint(0, len(late), (len(ids),), device=env.device)
            frame_ids = late[local_idx]
            self._write_rsi_state(env, ids, pool, robot, frame_ids=frame_ids)
            env._blue_seed_landed_pending[ids] = True

    def reset(
        self,
        env: "ManagerBasedRlEnv",
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
        rsi_fraction: float = 0.8,
        blue_practice_fraction: float = 0.0,
    ) -> None:
        """Literal port of Humanoid-Goalkeeper G1's continue_keep mechanism.

        Source: Humanoid-Goalkeeper/legged_gym/legged_gym/envs/base/legged_robot.py
        _reset_dofs, lines 657-682, with G1's ACTIVE config values from
        legged_gym/legged_gym/envs/g1/g1_29_config.py:279-282
        (continue_keep=True, randomize_initial_joint_pos=True,
        initial_joint_pos_scale=[0.5, 1.5], initial_joint_pos_offset=[-0.1, 0.1]);
        both files are read-only upstream references, not modified:

            dof_upper = self.dof_pos_limits[:, 1].view(1, -1)
            dof_lower = self.dof_pos_limits[:, 0].view(1, -1)
            if self.cfg.domain_rand.continue_keep and torch.rand(1).item() > 0.2:
                self.dof_pos[env_ids] = self.dof_pos[torch.randint(0, self.num_envs, (len(env_ids),), device=...)]
            else:
                init_dos_pos = self.standpos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=...)
                init_dos_pos += torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=...)
                self.dof_pos[env_ids] = torch.clip(init_dos_pos, dof_lower, dof_upper)
            self.dof_vel[env_ids] = 0.

        One coin flip per reset() call decides ALL of env_ids together — this
        is G1's own behavior (`torch.rand(1)`, not one flip per env). On the
        rsi_fraction branch, dof_pos is copied from `torch.randint(0, num_envs, ...)`:
        ANY currently-running env, with no side/tier matching, no exclusion of
        the current reset batch, no minimum-age check, and — matching G1
        exactly — NO clamping: G1's `torch.clip` only appears in the OTHER
        branch, never on the continue_keep copy. This deliberately replaces
        the side/tier-conditioned pool system this project used previously —
        see docs/superpowers/plans/2026-07-01-live-env-rsi.md for the full
        comparison, including a fidelity audit that caught the clamp being
        on the wrong branch and this else-branch's randomization being
        missing entirely in an earlier version of this port.

        dof_vel is always zeroed, matching G1's unconditional
        `self.dof_vel[env_ids] = 0.` (not just on the continue_keep branch).
        Root pose/velocity are untouched here in both branches — reset_base
        handles them, matching G1's separate _reset_root_states (note: G1's
        _reset_root_states also randomizes root velocity ±0.3 unconditionally
        on every reset, which this project's reset_base does not — a known,
        separate divergence outside reset_from_motion_data's scope).

        FEAT 2026-07-10 (blue_practice_fraction, no G1 equivalent -- see
        docs/BugFixes.md "RSI-style episode-start curriculum"): on the
        continue_keep (rsi_fraction) branch, donor selection is normally
        uniform over ALL currently-running envs. For a curriculum-controlled
        fraction of resets, instead bias donor selection toward envs
        CURRENTLY mid-approach on a wide crossing and not yet landed
        (env._blue_wide & ~env._blue_landed) -- giving a freshly-reset
        episode a joint pose that resembles already being partway through
        the hard approach-and-plant behavior, rather than always starting
        cold. Directly manufactures more training exposure to the rare,
        hard-to-reach state that genuine landing requires (JSRL / reverse-
        curriculum-generation pattern), annealed down as training matures
        (see reward_curriculum_ep_len-driven fraction in
        reset_from_motion_data). Falls back to uniform selection if no
        candidate envs currently satisfy the condition (e.g. very early in
        training). Root position/ball state are unaffected -- only the
        donor's joint pose is borrowed, exactly as continue_keep already
        does for any other donor.
        """
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int32)
        if len(env_ids) == 0:
            return

        robot: Entity = env.scene[asset_cfg.name]
        n = len(env_ids)

        if torch.rand(1, device=env.device).item() > (1.0 - rsi_fraction):
            uniform_donor = torch.randint(0, env.num_envs, (n,), device=env.device)
            donor_idx = uniform_donor
            if blue_practice_fraction > 0.0:
                candidate_mask = getattr(env, "_blue_wide", None)
                landed_mask = getattr(env, "_blue_landed", None)
                if candidate_mask is not None and landed_mask is not None:
                    practice_pool = torch.nonzero(candidate_mask & ~landed_mask, as_tuple=False).flatten()
                    if practice_pool.numel() > 0:
                        practice_mask = torch.rand(n, device=env.device) < blue_practice_fraction
                        practice_local_idx = torch.randint(0, practice_pool.numel(), (n,), device=env.device)
                        practice_donor = practice_pool[practice_local_idx]
                        donor_idx = torch.where(practice_mask, practice_donor, uniform_donor)
            joint_pos = robot.data.joint_pos[donor_idx].clone()
        else:
            default_pos = robot.data.default_joint_pos[env_ids]
            num_dof = default_pos.shape[-1]
            scale = sample_uniform(0.5, 1.5, (n, num_dof), env.device)
            offset = sample_uniform(-0.1, 0.1, (n, num_dof), env.device)
            joint_pos = default_pos * scale + offset
            limits = robot.data.joint_pos_limits[env_ids.long()]
            joint_pos = joint_pos.clamp(limits[..., 0], limits[..., 1])

        joint_vel = torch.zeros_like(joint_pos)
        robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        # Root state already correct from reset_base.


def init_motion_loader(env: "ManagerBasedRlEnv", env_ids: torch.Tensor | None) -> None:
    """Startup event: load all NPZ motion files into MotionResetManager."""
    MotionResetManager.get().init(env)


_BLUE_PRACTICE_BASE_FRACTION = 0.4

# FEAT 2026-07-11: fraction of newly-reset FAR-region envs seeded directly
# from a real, late-clip DoubleStep/TripleStep demonstration frame -- see
# MotionResetManager.seed_blue_landed_practice's docstring. Deliberately NOT
# curriculum-annealed like _BLUE_PRACTICE_BASE_FRACTION above: that fraction
# exists to help DISCOVER the approach-and-plant behavior, which becomes
# less necessary as the policy matures; this one exists to give the
# post-landing follow-through phase ongoing training exposure independent of
# whether genuine landing has been solved, which doesn't stop being useful
# once training matures. Applied on top of (independent of) rsi_fraction/
# blue_practice_fraction below -- this seeding happens AFTER mgr.reset()
# writes a baseline joint state, and overrides it (root pose/velocity too)
# only for the selected subset. 0.25 chosen as a middle ground: frequent
# enough to give consistent, dense exposure (roughly 1 in 4 far-region
# resets, far regions being about half of all resets given the region
# split), without dominating the training distribution to the point the
# policy could lean on being handed the hard part for free too often instead
# of learning to reach it. Not tuned against an ablation; flagged for
# revisit if it doesn't move the post-landing behavior.
_BLUE_LANDED_SEED_FRACTION = 0.25


def reset_from_motion_data(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> None:
    """Reset event: continue_keep-style donor copy vs default pose.

    FIX 2026-07-01: this wrapper hardcoded rsi_fraction=0.5 (a silent 50/50
    split), diverging from both G1's actual 80/20 (`torch.rand(1).item() > 0.2`,
    legged_robot.py:669) and this project's own reset()/CLAUDE.md-documented
    80/20 intent. Caught by an independent fidelity audit — see
    docs/superpowers/plans/2026-07-01-live-env-rsi.md.

    FEAT 2026-07-10: computes blue_practice_fraction from the shared
    env._curriculumupdate (0-3, bidirectional as of the curriculum-ratchet
    fix), linearly annealed from _BLUE_PRACTICE_BASE_FRACTION at cu=0 down
    to 0 at cu>=3 -- heaviest early in training when genuinely reaching a
    near-blue state through organic exploration is rarest, tapering off as
    the policy matures and no longer needs the assist. Since
    env._curriculumupdate is itself now bidirectional (can drop back down
    after a regression, not just climb), this fraction will naturally climb
    back up too if training regresses -- consistent with the rest of this
    curriculum system. See mgr.reset's docstring for the mechanism itself.

    FEAT 2026-07-11: after mgr.reset() writes its baseline joint state, calls
    mgr.seed_blue_landed_practice to override a fraction of far-region envs
    with a real post-plant motion-capture frame instead -- see that method's
    docstring and _BLUE_LANDED_SEED_FRACTION above.
    """
    mgr = MotionResetManager.get()
    mgr.init(env)
    cu = getattr(env, "_curriculumupdate", 0)
    blue_practice_fraction = _BLUE_PRACTICE_BASE_FRACTION * max(0.0, 1.0 - cu / 3.0)
    mgr.reset(env, env_ids, asset_cfg, rsi_fraction=0.8, blue_practice_fraction=blue_practice_fraction)

    resolved_ids = env_ids
    if resolved_ids is None:
        resolved_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int32)
    if len(resolved_ids) > 0:
        mgr.seed_blue_landed_practice(env, resolved_ids, asset_cfg, _BLUE_LANDED_SEED_FRACTION)


def reset_ball_local_frame(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    ball_name: str,
    dist_range: tuple[float, float] = (2.0, 4.0),
    y_start_range: tuple[float, float] = (-0.8, 0.8),
    y_end_range: tuple[float, float] = (-0.5, 0.5),
    spawn_height_range: tuple[float, float] = (0.2, 0.7),
    arrive_height_range: tuple[float, float] = (0.1, 0.4),
    speed_range: tuple[float, float] = (3.0, 7.0),
) -> None:
    """Spawn ball aimed at the goal area from a variable lateral position.

    Ball spawns at (dist, y_start) in the robot's local frame and is aimed toward
    (-0.3, y_end) — 0.3 m behind the goal line so the ball retains forward momentum
    at interception (matches ILB). The velocity vector has both X and Y components so
    the ball crosses the goal at a realistic angle.

    Spawning in the robot's local frame means the policy is orientation-invariant:
    the ball always arrives from the robot's front regardless of world yaw. The yaw
    penalty (ang_vel_z) penalises spinning while keeping the coordinate system clean.

    y_start_range ±0.8 m matches ILB (narrowed from ±1.5 m to avoid >50° approach
    angles that produce no gradient signal).

    Args:
        dist_range:          forward (local +X) distance from robot to spawn (m)
        y_start_range:       lateral spawn offset in robot +Y (m)
        y_end_range:         goal target lateral position in robot +Y (m)
        spawn_height_range:  ball height above floor at spawn (m)
        arrive_height_range: target ball height above floor at goal line (m)
        speed_range:         total horizontal approach speed magnitude (m/s)
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    n = len(env_ids)

    # Curriculum difficulty lerps from easy ranges (d=0) to configured params (d=1).
    d = float(getattr(env, "_ball_difficulty", 1.0))
    d = max(0.0, min(1.0, d))

    dist_r    = _lerp_range(_EASY_DIST,    dist_range,          d)
    y_start_r = _lerp_range(_EASY_Y_START, y_start_range,       d)
    y_end_r   = _lerp_range(_EASY_Y_END,   y_end_range,         d)
    z_start_r = _lerp_range(_EASY_Z_START, spawn_height_range,  d)
    z_end_r   = _lerp_range(_EASY_Z_END,   arrive_height_range, d)
    speed_r   = _lerp_range(_EASY_SPEED,   speed_range,         d)

    g = 9.81
    ball: Entity = env.scene[ball_name]
    robot: Entity = env.scene["robot"]

    robot_pos_w  = robot.data.root_link_pos_w[env_ids]      # (n, 3)
    robot_quat_w = robot.data.root_link_quat_w[env_ids]     # (n, 4) wxyz
    yaw_q        = _yaw_only_quat(robot_quat_w)             # pure-yaw quat

    floor_z = env.scene.env_origins[env_ids, 2]             # (n,) floor per env

    x_start = sample_uniform(*dist_r,    (n,), env.device)   # forward distance
    y_start = sample_uniform(*y_start_r, (n,), env.device)   # lateral spawn
    y_end   = sample_uniform(*y_end_r,   (n,), env.device)   # goal target Y
    z_start = floor_z + sample_uniform(*z_start_r, (n,), env.device)
    z_end   = floor_z + sample_uniform(*z_end_r,   (n,), env.device)
    speed_h = sample_uniform(*speed_r,   (n,), env.device)   # total horizontal speed

    # Ball spawn position in local frame: forward x_start, lateral y_start.
    local_spawn = torch.stack([x_start, y_start, torch.zeros_like(x_start)], dim=-1)
    world_spawn_xy = quat_apply(yaw_q, local_spawn)
    ball_pos = torch.empty((n, 3), device=env.device)
    ball_pos[:, 0] = robot_pos_w[:, 0] + world_spawn_xy[:, 0]
    ball_pos[:, 1] = robot_pos_w[:, 1] + world_spawn_xy[:, 1]
    ball_pos[:, 2] = z_start

    # Velocity direction: from spawn (x_start, y_start) toward (-0.3, y_end).
    # Aiming 0.3 m behind the goal line (matches ILB) so the ball still has forward
    # momentum when it reaches the robot — avoids deceleration to zero right at x=0.
    dx_local = -(x_start + 0.3)     # negative: ball moves toward robot then past it
    dy_local = y_end - y_start       # lateral component; nonzero for angled shots
    horiz_dist = torch.sqrt(dx_local ** 2 + dy_local ** 2)
    t_flight = horiz_dist / speed_h

    vx_local = dx_local / t_flight
    vy_local = dy_local / t_flight

    local_vel_h = torch.stack([vx_local, vy_local, torch.zeros_like(vx_local)], dim=-1)
    world_vel_h = quat_apply(yaw_q, local_vel_h)

    # Gravity-compensating vz: z_end = z_start + vz*t - 0.5*g*t^2
    vz = ((z_end - z_start) + 0.5 * g * t_flight ** 2) / t_flight

    ball_vel = torch.empty((n, 3), device=env.device)
    ball_vel[:, 0] = world_vel_h[:, 0]
    ball_vel[:, 1] = world_vel_h[:, 1]
    ball_vel[:, 2] = vz

    ball_quat = torch.zeros((n, 4), device=env.device)
    ball_quat[:, 0] = 1.0
    ball.write_root_link_pose_to_sim(
        torch.cat([ball_pos, ball_quat], dim=-1), env_ids=env_ids
    )
    ball.write_root_link_velocity_to_sim(
        torch.cat([ball_vel, torch.zeros((n, 3), device=env.device)], dim=-1),
        env_ids=env_ids,
    )

    _init_visibility_state(env, env_ids)


def reset_ball_global_frame(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    ball_name: str,
    x_start_range: tuple[float, float] = (2.0, 3.5),
    y_start_range: tuple[float, float] = (-0.5, 0.5),
    y_end_range: tuple[float, float] = (-0.5, 0.5),
    z_start_range: tuple[float, float] = (0.10, 0.20),
    z_end_range: tuple[float, float] = (0.05, 0.10),
    t_flight_range: tuple[float, float] = (0.35, 0.60),
) -> None:
    """Spawn ball aimed at goal from global +X direction (mirrors ILB _reset_ball).

    Ball spawns at (env_origin + x_start, y_start, z_start) in world frame and
    travels toward (env_origin - 0.3, y_end, z_end). Uses t_flight directly (not
    speed) so vz stays bounded regardless of distance — short t_flight keeps the
    arc flat at foot/ankle level.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    n = len(env_ids)

    d = float(getattr(env, "_ball_difficulty", 1.0))
    d = max(0.0, min(1.0, d))

    _EASY_T = (0.30, 0.45)
    x_start_r  = _lerp_range((2.0, 2.5),   x_start_range,  d)
    y_start_r  = _lerp_range((-0.2, 0.2),  y_start_range,  d)
    y_end_r    = _lerp_range((-0.1, 0.1),  y_end_range,    d)
    z_start_r  = _lerp_range((0.05, 0.10), z_start_range,  d)
    z_end_r    = _lerp_range((0.05, 0.08), z_end_range,    d)
    t_range    = _lerp_range(_EASY_T,      t_flight_range, d)

    g = 9.81
    ball: Entity = env.scene[ball_name]
    origins = env.scene.env_origins[env_ids]

    x_start  = sample_uniform(*x_start_r,  (n,), env.device)
    y_start  = sample_uniform(*y_start_r,  (n,), env.device)
    y_end    = sample_uniform(*y_end_r,    (n,), env.device)
    z_start  = origins[:, 2] + sample_uniform(*z_start_r, (n,), env.device)
    z_end    = origins[:, 2] + sample_uniform(*z_end_r,   (n,), env.device)
    t_flight = sample_uniform(*t_range,    (n,), env.device)

    ball_pos = torch.stack([
        origins[:, 0] + x_start,
        origins[:, 1] + y_start,
        z_start,
    ], dim=-1)
    ball_quat = torch.zeros((n, 4), device=env.device)
    ball_quat[:, 0] = 1.0

    dx = -(x_start + 0.3)
    dy = y_end - y_start
    dz = z_end - z_start

    vx = dx / t_flight
    vy = dy / t_flight
    vz = (dz + 0.5 * g * t_flight ** 2) / t_flight

    ball_vel = torch.stack([vx, vy, vz], dim=-1)

    ball.write_root_link_pose_to_sim(
        torch.cat([ball_pos, ball_quat], dim=-1), env_ids=env_ids
    )
    ball.write_root_link_velocity_to_sim(
        torch.cat([ball_vel, torch.zeros((n, 3), device=env.device)], dim=-1),
        env_ids=env_ids,
    )

    _init_visibility_state(env, env_ids)


def _init_visibility_state(env: "ManagerBasedRlEnv", env_ids: torch.Tensor) -> None:
    """Reset per-env visibility counters at episode start."""
    n = env.num_envs

    if not hasattr(env, "_catchstep"):
        env._catchstep = torch.zeros(n, dtype=torch.long, device=env.device)
    env._catchstep[env_ids] = 50

    if not hasattr(env, "_startstep"):
        env._startstep = torch.zeros(n, dtype=torch.long, device=env.device)
    env._startstep[env_ids] = 50 - torch.randint(3, 11, (len(env_ids),), device=env.device)

    if not hasattr(env, "_vanish_step"):
        env._vanish_step = torch.zeros(n, dtype=torch.long, device=env.device)
    env._vanish_step[env_ids] = torch.randint(0, 30, (len(env_ids),), device=env.device)

    if not hasattr(env, "_ball_visible_step"):
        env._ball_visible_step = torch.zeros(n, device=env.device)
    env._ball_visible_step[env_ids] = 0

    if not hasattr(env, "_ball_obs_last_x"):
        env._ball_obs_last_x = torch.zeros(n, device=env.device)
    env._ball_obs_last_x[env_ids] = 0.0


def tick_catchstep(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor,
) -> None:
    """Decrement catchstep warmup counter each step (interval event, interval=(0,0))."""
    if hasattr(env, "_catchstep"):
        env._catchstep = (env._catchstep - 1).clamp(min=0)


_CURRICULUM_EMA_ALPHA = 0.3


def _update_smoothed_ep_len(env: "ManagerBasedRlEnv", mean_ep_len: float) -> float:
    """Exponentially smooth the raw per-window mean_ep_len reading before any
    curriculum term computes cu from it. Shared across reward_curriculum_ep_len
    and ball_difficulty_curriculum (both use the same ep_len_divisor=47) so
    they stay synchronized rather than oscillating independently.

    FIX 2026-07-10: the 2026-07-09 bidirectional-curriculum fix (removing the
    monotonic ratchet) correctly stopped the "stuck forever at a level it
    can't handle" failure, but a live run (rsi_practice_curriculum_2026-07-10)
    showed the opposite overcorrection: with no damping, a single noisy
    500-step window's mean_ep_len is enough to flip cu, and ball_difficulty
    was observed oscillating 0.667<->0.333 repeatedly for the entire run
    (never settling at either level for more than 1-2 updates), with
    mean_episode_length bouncing noisily (78-113) and blue_overshoot_penalty
    flat/oscillating around -0.4 to -0.5 the whole run instead of improving --
    consistent with a policy that never gets a stable enough training
    distribution to consolidate a behavior before the ground shifts again.
    The original (removed) monotonic design's own docstring flagged this
    exact risk ("prevents oscillation near the boundary") but fixed it the
    wrong way (freezing forever) instead of damping the input signal.
    OpenAI's ADR (arXiv 1910.07113) avoids this the right way: difficulty
    boundaries move by small steps based on a rolling performance buffer, not
    an instant snap to the latest noisy reading. This EMA (alpha=0.3, i.e.
    30% weight to the newest window, 70% to history) is the minimal version
    of that idea -- a genuine, sustained shift still moves cu within a few
    updates, but a single noisy window can no longer flip it outright.
    See docs/BugFixes.md.
    """
    if not hasattr(env, "_smoothed_ep_len"):
        env._smoothed_ep_len = mean_ep_len
    else:
        env._smoothed_ep_len = (
            _CURRICULUM_EMA_ALPHA * mean_ep_len + (1.0 - _CURRICULUM_EMA_ALPHA) * env._smoothed_ep_len
        )
    return env._smoothed_ep_len


def track_blue_landing_success(env: "ManagerBasedRlEnv", env_ids: torch.Tensor | None) -> None:
    """Reset event (mode="reset"): accumulates each just-ended episode's
    wide/landed outcome into a windowed rolling success rate,
    env._blue_landing_success_rate, updated every 500 steps (same cadence as
    the other curricula here).

    FEAT 2026-07-11: "decouple task and behavior" idea from the ranked
    research list (docs/BugFixes.md, two dispatched research passes both
    independently ranked this the first thing to try, before a harder
    stage-gate restructure) -- rather than stopball/softstop's existing
    all-or-nothing per-episode landing_ok gate (env._blue_landed, already in
    place), track how RELIABLY the policy reproduces genuine landing across
    recent wide episodes, and let mdp.rewards scale the downstream
    stopball/softstop payoff for wide-crossing saves by that reliability --
    full credit once landing is consistently reproduced, damped credit while
    it's still rare/lucky. Deliberately reads env._blue_wide/env._blue_landed
    (both already latched fresh for the whole episode by _get_reach_target_y)
    on THIS reset event, which runs before _get_reach_target_y's own
    just_reset clears them for the incoming episode -- so this always sees
    the outgoing episode's final, correct outcome, not the reset value.

    Runs at mode="reset" (fires only for envs that just terminated), separate
    from the mode="reset"-driven reward_curriculum_ep_len/ball_difficulty_
    curriculum classes above (which are CurriculumTermCfg, evaluated by the
    curriculum manager on a different cadence/API) -- this is a plain
    EventTermCfg since it only needs to accumulate counts, not return a
    scale itself (mdp.rewards reads env._blue_landing_success_rate directly).
    """
    if env_ids is None or len(env_ids) == 0:
        return
    if not hasattr(env, "_blue_wide") or not hasattr(env, "_blue_landed"):
        return  # narrow-only task variant (e.g. green-ball-baseline) has no blue mechanism

    if not hasattr(env, "_blue_landing_success_rate"):
        env._blue_landing_success_rate = 0.0
        env._blue_success_window_count = 0
        env._blue_wide_window_count = 0
        env._blue_success_last_update = -500

    wide_mask = env._blue_wide[env_ids]
    # 2026-07-11 FIX: exclude RSI-seeded free landings (env._blue_landed_was_free)
    # -- without this, seed_blue_landed_practice's ~25%-of-far-region teleport
    # landings (events.py, _BLUE_LANDED_SEED_FRACTION) pushed the rolling
    # success rate to ~0.25 against the 0.3 target on their own, near-fully
    # unlocking _blue_landing_reward_scale's stopball/softstop payoff boost
    # regardless of true genuine-landing rate (measured 0.1-3%) -- exactly the
    # "damp payoff while landing is still rare/lucky" behavior this rate exists
    # to produce. Found by audit, not yet validated on a live training run.
    landed_mask = env._blue_landed[env_ids] & ~env._blue_landed_was_free[env_ids]
    env._blue_wide_window_count += int(wide_mask.sum().item())
    env._blue_success_window_count += int((wide_mask & landed_mask).sum().item())

    if env.common_step_counter - env._blue_success_last_update >= 500:
        env._blue_success_last_update = env.common_step_counter
        if env._blue_wide_window_count > 0:
            env._blue_landing_success_rate = (
                env._blue_success_window_count / env._blue_wide_window_count
            )
        env._blue_wide_window_count = 0
        env._blue_success_window_count = 0


class reward_curriculum_ep_len:
    """G1-style episode-length-driven reward weight curriculum.

    Mirrors G1 legged_robot.py:325-364 exactly:
        every 500 env steps:
            curriculumupdate = int(mean_episode_length / ep_len_divisor)  # integer 0-3
        then per reward:
            weight = base_weight * (1 + 0.5 * curriculumupdate)

    All instances share env._curriculumupdate so the episode-length sampling
    is identical across all ramped rewards (same as G1's single class variable).

    G1 extremes: ep_len=150 (full episode) → curriculumupdate=3 → weight = 2.5 × base.

    FIX 2026-07-10: cu is now computed from an EMA-smoothed mean_ep_len
    (_update_smoothed_ep_len), not the raw per-window reading -- see that
    function's docstring for why.
    """

    def __init__(self, cfg: "CurriculumTermCfg", env: "ManagerBasedRlEnv") -> None:
        p = cfg.params
        self._reward_name     = p["reward_name"]
        self._base_weight     = p["base_weight"]
        self._update_interval = p.get("update_interval", 500)
        self._ep_len_divisor  = p.get("ep_len_divisor",   50)
        self._last_update     = -(self._update_interval)  # fire immediately on first call
        self._term_cfg        = env.reward_manager.get_term_cfg(self._reward_name)
        if not hasattr(env, "_curriculumupdate"):
            env._curriculumupdate = 0

    def __call__(
        self,
        env: "ManagerBasedRlEnv",
        env_ids: torch.Tensor,
        reward_name: str,
        base_weight: float,
        **kwargs,
    ) -> dict:
        # Update shared curriculumupdate once per window (first term to run wins).
        if env.common_step_counter - self._last_update >= self._update_interval:
            self._last_update = env.common_step_counter
            if len(env_ids) > 0:
                mean_ep_len = env.episode_length_buf[env_ids].float().mean().item()
            else:
                mean_ep_len = 0.0
            smoothed_ep_len = _update_smoothed_ep_len(env, mean_ep_len)
            cu = int(smoothed_ep_len / self._ep_len_divisor)
            # FIX 2026-07-09: was `max(env._curriculumupdate, cu)` (monotonic
            # ratchet, never drops). G1 (legged_robot.py:329) has no such floor
            # -- curriculumupdate is a fresh recomputation every update, so
            # reward/difficulty scales ease back down if performance regresses.
            # The ratchet meant a policy that hit a hard patch after advancing
            # (e.g. a difficulty jump it couldn't yet handle) had no path back
            # to easier practice -- confirmed live: ball_difficulty hit 1.0,
            # shank_height terminations and episode length regressed hard, and
            # never recovered over 2500+ further iterations because nothing
            # could ease back off. See docs/BugFixes.md.
            env._curriculumupdate = cu

        # G1 formula — weight = base * (1 + 0.5 * cu), cu capped at 3 naturally by ep_len.
        new_weight = self._base_weight * (1.0 + 0.5 * env._curriculumupdate)
        self._term_cfg.weight = new_weight
        return {"weight": torch.tensor(float(new_weight))}


class correct_foot_save_curriculum:
    """Step-up curriculum for correct-foot-save quality bonuses.

    Reads the shared env._curriculumupdate (written by reward_curriculum_ep_len
    instances) and doubles the reward weight the moment cu reaches a threshold.

    Formula: weight = base_weight * multiplier
        where multiplier = 1.0 when cu < activate_at_cu
                         = 2.0 when cu >= activate_at_cu

    Intended for one-time quality bonuses that only make sense once the policy
    has already learned to save reliably (cu >= 3 = ep_len ≈ 144 steps):
        single_foot_save, inner_face_orientation_save, cleanstop, airborne_at_save.

    One curriculum entry per reward (same pattern as reward_curriculum_ep_len).
    Shares env._curriculumupdate — no extra episode-length sampling.
    """

    def __init__(self, cfg: "CurriculumTermCfg", env: "ManagerBasedRlEnv") -> None:
        p = cfg.params
        self._reward_name    = p["reward_name"]
        self._base_weight    = p["base_weight"]
        self._activate_at_cu = p.get("activate_at_cu", 3)
        self._term_cfg       = env.reward_manager.get_term_cfg(self._reward_name)
        if not hasattr(env, "_curriculumupdate"):
            env._curriculumupdate = 0

    def __call__(
        self,
        env: "ManagerBasedRlEnv",
        env_ids: torch.Tensor,
        reward_name: str,
        base_weight: float,
        **kwargs,
    ) -> dict:
        multiplier = 2.0 if env._curriculumupdate >= self._activate_at_cu else 1.0
        new_weight = self._base_weight * multiplier
        self._term_cfg.weight = new_weight
        return {"weight": torch.tensor(float(new_weight))}


class ball_difficulty_curriculum:
    """Adaptive difficulty curriculum — direct port of Humanoid-Goalkeeper (G1) approach.

    G1 legged_robot.py:325-336: every `update_interval` per-env steps compute
        curriculumupdate = int(mean_episode_length / ep_len_divisor)
    and expand command ranges by `step_size × curriculumupdate`:
        command_ranges[i] = clip(command_ranges[i] ± 0.3*curriculumupdate, bound_lo, bound_hi)
    This is an ACCUMULATOR, not a fresh recompute -- each update nudges the
    range outward by a small bounded step. Since curriculumupdate = int(...)
    is never negative, this can only grow or hold flat; a bad window just
    means zero growth that cycle, not retreat. Contrast with G1's REWARD
    weights (eereach/success/stopball, compute_reward() lines 359-364),
    which ARE freshly, directly recomputed every step from curriculumupdate
    and genuinely go up and down -- G1 uses two different patterns for two
    different things.

    FIX 2026-07-10: the 2026-07-09 fix ("remove the monotonic ratchet")
    correctly matched G1's reward-weight pattern for reward_curriculum_ep_len,
    but WRONGLY applied that same "fresh, fully bidirectional recompute"
    pattern here too (`difficulty = min(1.0, cu/3.0)` every update) --
    this is G1's reward-weight pattern, not its difficulty pattern. A live
    run (rsi_practice_curriculum_2026-07-10) showed ball_difficulty
    oscillating 0.667<->0.333 repeatedly the entire run, never settling --
    exactly the failure mode G1's accumulator can't have by construction
    (it structurally cannot move backward). Reverted to a step-size-based
    accumulator matching G1's actual command_ranges mechanism: difficulty
    only ever increases (or holds flat), clipped to 1.0, never resets to a
    fresh absolute value. See docs/BugFixes.md.

    Default params mirror G1 exactly:
        ep_len_divisor = 50   (same as G1)
        update_interval = 500 (same as G1, in per-env steps)
        step_size = 0.01      (difficulty units per curriculumupdate per check;
                               at cu=2 sustained, reaches 1.0 in ~25 checks;
                               original project value, unused since the
                               2026-07-09 fix, now restored to actual use)
    """

    def __init__(self, cfg: "CurriculumTermCfg", env: "ManagerBasedRlEnv") -> None:
        p = cfg.params
        self._step_size      = p.get("step_size",      0.01)
        self._update_interval = p.get("update_interval", 500)
        self._ep_len_divisor  = p.get("ep_len_divisor",   50)
        self._last_update     = -(self._update_interval)
        if not hasattr(env, "_ball_difficulty"):
            env._ball_difficulty = 0.0

    def __call__(
        self,
        env: "ManagerBasedRlEnv",
        env_ids: torch.Tensor,
        **kwargs,
    ) -> dict:
        # Only update every `update_interval` per-env steps (mirrors G1's 500-step cadence).
        if env.common_step_counter - self._last_update < self._update_interval:
            return {"ball_difficulty": torch.tensor(env._ball_difficulty)}

        self._last_update = env.common_step_counter

        # G1: curriculumupdate = int(mean_episode_length / 50)
        # Uses lengths of just-completed episodes (env_ids are resetting envs).
        # FIX 2026-07-10: EMA-smoothed via the same shared _update_smoothed_ep_len
        # as reward_curriculum_ep_len, so both curricula stay synchronized and
        # neither oscillates independently on single-window noise. See that
        # function's docstring.
        if len(env_ids) > 0:
            mean_ep_len = env.episode_length_buf[env_ids].float().mean().item()
        else:
            mean_ep_len = 0.0
        smoothed_ep_len = _update_smoothed_ep_len(env, mean_ep_len)
        curriculumupdate = int(smoothed_ep_len / self._ep_len_divisor)

        # FIX 2026-07-10: accumulator, matching G1's ACTUAL command_ranges
        # mechanism (legged_robot.py:329-335: range = clip(range ± 0.3*cu,
        # bound_lo, bound_hi) -- an incremental nudge each update, never a
        # fresh absolute recompute). curriculumupdate is never negative, so
        # this can only grow or hold flat -- structurally cannot oscillate,
        # unlike the 2026-07-09 version's `difficulty = min(1.0, cu/3.0)`
        # direct recompute (that formula matches G1's REWARD weights, not
        # its difficulty mechanism -- see class docstring). No ratchet
        # needed here: G1's own mechanism has no floor/ceiling logic beyond
        # the clip, because monotonic-non-decreasing is its natural,
        # structural behavior, not a bolted-on guard.
        env._ball_difficulty = min(1.0, env._ball_difficulty + self._step_size * curriculumupdate)
        return {"ball_difficulty": torch.tensor(env._ball_difficulty)}


def sharpforce_termination(
    env: "ManagerBasedRlEnv",
    max_contact_force: float = 1500.0,
) -> torch.Tensor:
    """Terminate when mean foot contact force exceeds threshold.

    Mirrors upstream Humanoid-Goalkeeper sharpforce_buf termination.
    Requires feet_contact ContactSensor in the scene.
    Geom layout (sorted by name): left_foot_1, left_foot_2, right_foot_1, right_foot_2.
    Returns [B] bool tensor: True → terminate.
    """
    sensor: ContactSensor = env.scene["feet_contact"]
    force_per_geom = sensor.data.force.norm(dim=-1)       # [B, 8]
    left_max  = force_per_geom[:, :4].max(dim=-1).values  # left_foot1-4
    right_max = force_per_geom[:, 4:].max(dim=-1).values  # right_foot1-4
    mean_force = (left_max + right_max) / 2.0
    return mean_force > max_contact_force


def shank_height_termination(
    env: "ManagerBasedRlEnv",
    min_height: float = 0.24,
    asset_cfg: "SceneEntityCfg" = SceneEntityCfg("robot", body_names=("Shank_Left", "Shank_Right")),
) -> torch.Tensor:
    """Terminate when either knee (shank body origin) drops below min_height above floor.

    Catches deep single-step lunges that the kneeheight penalty alone cannot prevent.
    Fires when the worst-case shank across left/right is below threshold.
    """
    robot: Entity = env.scene[asset_cfg.name]
    shank_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids, :]  # (N, 2, 3)
    floor_z = env.scene.env_origins[:, 2]
    shank_z = shank_pos_w[:, :, 2] - floor_z[:, None]                  # (N, 2)
    return shank_z.min(dim=-1).values < min_height


def reset_ball_rolling(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    ball_name: str,
    dist_range: tuple[float, float] = (1.5, 2.5),
    y_start_range: tuple[float, float] = (-0.5, 0.5),
    y_end_range: tuple[float, float] = (-0.5, 0.5),
    t_flight_range: tuple[float, float] = (0.7, 1.1),
    spawn_z: float = 0.10,
    y_end_outer_frac: float | None = None,
) -> None:
    """Spawn ball at ground level in world (global) frame — rolling ground pass.

    Spawns at (env_origin_x + x_start, env_origin_y + y_start) in world frame so
    the ball always approaches along world -X regardless of robot yaw. This keeps
    all world-frame reward terms (stopball uses world-X velocity, stayonline uses
    world-X position, ball_exit_termination uses world-X) perfectly aligned with
    the ball trajectory.

    vz=0: ball drops to ground in <0.05 s then rolls at foot/ankle height.
    Curriculum lerps from easy to hard via env._ball_difficulty.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    n = len(env_ids)

    d = float(getattr(env, "_ball_difficulty", 1.0))
    d = max(0.0, min(1.0, d))

    _EASY_DIST_R      = (1.5, 2.0)
    _EASY_Y_ROLL      = (-0.05, 0.05)
    _EASY_T_FLIGHT_R  = (0.9, 1.3)   # easy: long flight → slow balls; hard: 0.7–1.1 s

    dist_r      = _lerp_range(_EASY_DIST_R,     dist_range,     d)
    y_start_r   = _lerp_range(_EASY_Y_ROLL,     y_start_range,  d)
    t_flight_r  = _lerp_range(_EASY_T_FLIGHT_R, t_flight_range, d)

    ball: Entity = env.scene[ball_name]
    origins = env.scene.env_origins[env_ids]   # world goal-line origin per env
    floor_z = origins[:, 2]

    x_start  = sample_uniform(*dist_r,     (n,), env.device)
    y_start  = sample_uniform(*y_start_r,  (n,), env.device)
    t_flight = sample_uniform(*t_flight_r, (n,), env.device)

    if y_end_range[0] * y_end_range[1] > 0:
        # One-sided range (region-conditioned calls via reset_ball_rolling_by_region,
        # e.g. left_near=(0.15,0.5), right_far=(-0.9,-0.5)): the sign is the whole
        # point of the region label (left vs right) -- it must never be randomized.
        # BUG (fixed 2026-07-06): this branch used to fall through to the two-sided
        # dead-zone logic below, which (a) re-randomizes side with a 50/50 coin flip
        # regardless of y_end_range's actual sign, and (b) derives its magnitude
        # bounds from a fixed generic constant (_Y_OUTER=0.35) and max_half, ignoring
        # the region's own inner bound entirely -- so e.g. a "far" region
        # (0.5,0.9) sampled magnitudes as low as 0.1, spilling deep into "near"
        # territory. Verified empirically: ~50% wrong sign, and "far" regions landed
        # in their own intended [0.5,0.9] band only ~25% of the time. This silently
        # decorrelated the region_estimator's ground-truth label (env._region_id)
        # from the ball's actual observable trajectory, and fed the actor a
        # region_arg conditioning signal that didn't match the real shot ~50-75%
        # of the time -- the direct cause of the region_estimator's persistent
        # left/right confusion and the failure to converge on far-side
        # double/triple-step motions. Fix: derive lo/hi from this range's own
        # bounds and lerp the sampled magnitude within [lo,hi] only, sign fixed.
        lo = min(abs(y_end_range[0]), abs(y_end_range[1]))
        hi = max(abs(y_end_range[0]), abs(y_end_range[1]))
        sign_val = 1.0 if y_end_range[1] > 0 else -1.0
        if y_end_outer_frac is not None:
            inner = lo + (hi - lo) * max(0.0, min(1.0, y_end_outer_frac))
            outer = hi
        else:
            inner = lo
            outer = lo + (hi - lo) * d
        mag = sample_uniform(inner, outer, (n,), env.device)
        y_end = sign_val * mag
    else:
        # Two-sided dead zone for y_end (mirrors G1's ±0.2 m minimum offset).
        # At d=0: ball always arrives 0.15–0.35 m left or right of center — never
        # through the robot's legs regardless of RSI stance width.
        # At d=1: magnitude range opens to [0.0, max_half] covering the full range.
        # The dead zone shrinks linearly so there is no curriculum cliff at any d.
        _Y_INNER = 0.15   # minimum offset from center at d=0
        _Y_OUTER = 0.35   # maximum offset from center at d=0
        max_half = max(abs(y_end_range[0]), abs(y_end_range[1]))
        if y_end_outer_frac is not None:
            # Testing/play override: ignore the difficulty curriculum entirely for
            # this dimension and pin to the outer band of the true (max-difficulty)
            # range -- e.g. y_end_outer_frac=0.8 samples only the top 20% of
            # max_half, regardless of env._ball_difficulty's current value.
            inner = max_half * max(0.0, min(1.0, y_end_outer_frac))
            outer = max_half
        else:
            inner = max(_Y_INNER * (1.0 - d), 0.1)
            outer = _Y_OUTER + (max_half - _Y_OUTER) * d
        mag   = sample_uniform(inner, outer, (n,), env.device)
        side  = torch.where(torch.rand(n, device=env.device) > 0.5,
                            torch.ones(n, device=env.device),
                            -torch.ones(n, device=env.device))
        y_end = side * mag

    # Ball spawns in world frame: x_start metres in front of goal line, y_start to the side.
    ball_pos = torch.stack([
        origins[:, 0] + x_start,
        origins[:, 1] + y_start,
        floor_z + spawn_z,
    ], dim=-1)

    # World-frame velocity: aimed from (x_start, y_start) → goal-line target (-0.3, y_end).
    # Speed is derived from geometry and flight time (mirrors G1's assign_ball_states).
    dx = -(x_start + 0.3)          # world -X: from spawn toward 0.3 m behind goal line
    dy = y_end - y_start            # world Y: lateral sweep to target

    ball_vel = torch.stack([
        dx / t_flight,              # world vx (negative = approaching)
        dy / t_flight,              # world vy
        torch.zeros_like(x_start),  # vz = 0: drops instantly, rolls at foot level
    ], dim=-1)

    ball_quat = torch.zeros((n, 4), device=env.device)
    ball_quat[:, 0] = 1.0
    ball.write_root_link_pose_to_sim(
        torch.cat([ball_pos, ball_quat], dim=-1), env_ids=env_ids
    )
    # Pure-rolling angular velocity: no-slip condition for ball rolling in XY plane.
    # Without this the ball starts sliding (zero ω, nonzero v) and loses 2/7*v0 to
    # sliding friction before reaching rolling — at max spawn 3.5 m/s that is exactly
    # 1.0 m/s, enough to fire stopball from floor friction alone (false positive).
    # ωy = vx/r, ωx = -vy/r eliminates the sliding phase entirely.
    _BALL_RADIUS = 0.11
    ball_ang_vel = torch.stack([
        -ball_vel[:, 1] / _BALL_RADIUS,
         ball_vel[:, 0] / _BALL_RADIUS,
        torch.zeros(n, device=env.device),
    ], dim=-1)
    ball.write_root_link_velocity_to_sim(
        torch.cat([ball_vel, ball_ang_vel], dim=-1),
        env_ids=env_ids,
    )

    # Store predicted goal-line crossing Y (relative to env origin) for RSI pool selection.
    # Ball travels from (x_start, y_start) to (-0.3, y_end) along a straight line, linearly
    # parametrized by fraction f in [0,1]: X(f) = x_start - f*(x_start+0.3). The goal line is
    # at X=0 (NOT at the aim point X=-0.3), so solving X(f)=0 gives f = x_start/(x_start+0.3).
    # cross_y = y_start + (y_end - y_start) * f.
    # FIX 2026-07-07: previously used f = (x_start+0.3)/horiz_dist (horiz_dist = the Euclidean
    # spawn-to-aim-point distance) -- that's cos(trajectory angle), not the goal-line crossing
    # fraction, and it silently overestimated f for every diagonal shot (e.g. x_start=2.0,
    # dy=0.5: buggy f=0.977 vs correct f=0.870). Ported from the same bug in SimpleGoalKeeper's
    # reset_ball_rolling (fixed there same day) -- see docs/BugFixes.md.
    if not hasattr(env, "_rsi_cross_y"):
        env._rsi_cross_y = torch.zeros(env.num_envs, device=env.device)
    env._rsi_cross_y[env_ids] = y_start + (y_end - y_start) * (x_start / (x_start + 0.3))

    _init_visibility_state(env, env_ids)


def ball_exit_termination(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    behind_threshold: float = -0.5,
) -> torch.Tensor:
    """Terminate when ball has clearly passed the goal line.

    Previously also terminated on deflection (sb_flag & ball_x_vel > 0.5), which
    ended episodes immediately after any body contact — before feet could reach the
    ball. Now only the goal-line crossing terminates, giving feet time to contact
    the ball even after a torso deflection.
    """
    ball: Entity = env.scene[ball_name]
    ball_x_local = ball.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
    return ball_x_local < behind_threshold
