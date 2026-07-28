"""Play a trained goalkeeper policy on Booster T1 (mjlab backend).

Usage:
    # Zero-action sanity check:
    uv run sgk_play Mjlab-BeyondAMP-Goalkeeper-T1 --agent zero --num-envs 1

    # Play trained checkpoint:
    uv run sgk_play Mjlab-BeyondAMP-Goalkeeper-T1 \\
        --checkpoint-file logs/rsl_rl/simple_goalkeeper/<run>/model_500.pt

    # Play with ghost overlay (cycles 1st motion by default):
    uv run sgk_play Mjlab-BeyondAMP-Goalkeeper-T1-WithOverlay \\
        --checkpoint-file logs/rsl_rl/simple_goalkeeper/<run>/model_500.pt

    # Ghost overlay with specific motion file:
    uv run sgk_play Mjlab-BeyondAMP-Goalkeeper-T1-WithOverlay \\
        --checkpoint-file <ckpt> \\
        --motion-file src/simple_goalkeeper/motions/data/3-1_booster_t1.npz

    # Visualise reference motions only (no policy, ghost overlay):
    uv run sgk_play Mjlab-BeyondAMP-Goalkeeper-T1-WithOverlay \\
        --agent zero --no-terminations True

    # Multi-disc (intercept) checkpoint, with ghost overlay cycling through
    # this task's own 4-motion AMP dataset (REGION_MOTION_FILES):
    uv run sgk_play Mjlab-BeyondAMP-Goalkeeper-T1-MultiDisc-WithOverlay \\
        --checkpoint-file logs/rsl_rl/intercept_simple_goalkeeper_multidisc/<run>/model_5250.pt
"""
from __future__ import annotations

import os
import sys
import types
from collections import deque
from sys import stderr
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.lab_api.math import quat_apply
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

from beyondAMP.mjlab.rsl_rl import AMPEnvWrapper, AMPRunnerCfg
from rsl_rl_amp.runners.amp_on_policy_runner import AMPOnPolicyRunner
from simple_goalkeeper.rsl_rl_multi.him_amp_on_policy_runner import (
    _get_actor_current_obs,
)


@dataclass(frozen=True)
class PlayConfig:
    agent: Literal["zero", "random", "trained"] = "trained"
    checkpoint_file: str | None = None
    motion_file: str | None = None
    """Optional NPZ motion file for the WithOverlay task (overrides default)."""
    num_envs: int | None = None
    device: str | None = None
    no_terminations: bool = False
    viewer: Literal["auto", "native", "viser"] = "auto"
    difficulty: float | None = None
    """Override ball difficulty (0.0 = easiest, 1.0 = hardest). Default: use curriculum value."""
    difficulty_outer_only_frac: float | None = None
    """Ignore the difficulty curriculum for the ball's lateral target offset (y_end
    magnitude) and pin it to the outer band of each region's own range instead --
    e.g. 0.8 samples only the top 20% of the max offset, regardless of --difficulty.
    Only affects the region-conditioned reset (Mjlab-BeyondAMP-Goalkeeper-T1-MultiDisc*)."""
    rsi: bool = False
    """Enable Random State Initialization in play (default: off — starts from standing keyframe)."""
    force_region: Literal["left_near", "left_far", "right_near", "right_far"] | None = None
    """Pin every episode to a single region instead of the default random 0-3 cycling
    (--num-envs 1's randomize_region_on_reset). Lets you watch only e.g. right_far
    episodes back to back instead of getting 3/4 episodes from other regions.
    Only affects the region-conditioned reset (Mjlab-BeyondAMP-Goalkeeper-T1-MultiDisc*)."""
    analytics: bool = True
    """Print ball velocity, delta_vx, foot heights, and stopball/softstop state to stdout each step.
    Also toggleable at runtime with the V key."""


class AnalyticsPolicy:
    """Wraps a policy to print per-step ball/foot/reward analytics to stdout.

    Toggle with the V key in the viewer or pass --analytics True on the CLI.
    Prints a single overwriting line while active; outputs a separator on episode end.
    """

    def __init__(self, policy, env, enabled: bool = False) -> None:
        self._policy = policy
        self._env = env
        self.enabled = enabled
        self._step = 0
        self._ep = 0
        self._prev_ep_buf: "torch.Tensor | None" = None
        self._wf_flash = 0.0
        self._head_flash = 0.0
        # FIX 2026-07-22 (research: shank_height vs base_height termination
        # tuning): per-episode running minimums, so a deep-lunge episode's
        # worst height is captured even though the live status line
        # (below) scrolls past it. Reset at each episode boundary.
        self._min_base_h = float("inf")
        self._min_lsh_h = float("inf")
        self._min_rsh_h = float("inf")
        # DEBUG 2026-07-23 (TEMPORARY, remove after landing-gate investigation):
        # per-episode closest approach to the blue target + highest settle
        # count reached + whether landed ever fired, across many episodes at
        # once, instead of eyeballing single scrolling lines.
        self._min_blue_dist = float("inf")
        self._max_settle = 0
        self._ep_was_wide = False
        self._ep_landed = False

    def toggle(self) -> None:
        self.enabled = not self.enabled
        status = "ON" if self.enabled else "OFF"
        print(f"\n[Analytics] {status}", file=stderr)

    def __call__(self, obs: "torch.Tensor") -> "torch.Tensor":
        import torch as _torch
        actions = self._policy(obs)
        self._step += 1

        env = self._env.unwrapped
        ep_buf = env.episode_length_buf

        # Detect episode reset (any env reset → print separator for env 0).
        if self._prev_ep_buf is not None and (ep_buf[0] < self._prev_ep_buf[0]).item():
            if self.enabled:
                # FIX 2026-07-22 (research: shank_height vs base_height
                # termination tuning). env.termination_manager's per-term
                # `_term_dones` buffer is only overwritten by compute()
                # (called once per env.step()) and NOT cleared by reset() --
                # see termination_manager.py -- so it still reflects the
                # step that just caused THIS reset, even though we're now
                # one policy call past it. Read it here, before anything
                # else this call does, to report which termination(s)
                # fired for env 0's episode that just ended.
                term_mgr = getattr(env, "termination_manager", None)
                fired = []
                if term_mgr is not None:
                    for name in term_mgr.active_terms:
                        if bool(term_mgr.get_term(name)[0].item()):
                            fired.append(name)
                blue_summary = (
                    f" | BLUE min_dist={self._min_blue_dist:.2f} "
                    f"max_settle={self._max_settle}/3 landed={self._ep_landed}"
                    if self._ep_was_wide else " | BLUE narrow-crossing"
                )
                print(
                    f"\n[EpEnd] terminated_by={','.join(fired) or 'none'} | "
                    f"min_base={self._min_base_h:.2f} "
                    f"min_Lsh={self._min_lsh_h:.2f} min_Rsh={self._min_rsh_h:.2f}"
                    f"{blue_summary}",
                    file=stderr,
                )
            self._ep += 1
            self._min_base_h = float("inf")
            self._min_lsh_h = float("inf")
            self._min_rsh_h = float("inf")
            self._min_blue_dist = float("inf")
            self._max_settle = 0
            self._ep_was_wide = False
            self._ep_landed = False
        self._prev_ep_buf = ep_buf.clone()

        # DEBUG 2026-07-28: called unconditionally (before the enabled-gate
        # below, and independent of the viewer's _show_plots/_is_paused
        # state) so the wrong-foot/head contact latch keeps advancing every
        # step regardless of whether analytics printing or the P-panel
        # plots happen to be toggled on -- the viewer-only version of this
        # call (_patch_viewer_wrong_foot_contact_plot) only runs while
        # _show_plots is True, which made it an unreliable ground-truth
        # source for confirming whether contact is actually being detected.
        self._wf_flash, self._head_flash = _compute_wrong_foot_contact_flash(env, 0)

        if not self.enabled:
            return actions

        ball = env.scene["ball"]
        robot = env.scene["robot"]

        bv = ball.data.root_link_lin_vel_w[0]        # ball velocity world frame
        bp = ball.data.root_link_pos_w[0]            # ball position world frame
        bx_local = (bp[0] - env.scene.env_origins[0, 0]).item()

        init_vx = getattr(env, "_sb_init_vx", None)
        sb_flag  = getattr(env, "_sb_flag", None)
        ss_flag  = getattr(env, "_softstop_flag", None)
        cs_flag  = getattr(env, "_cleanstop_flag", None)

        delta_vx = (bv[0] - init_vx[0]).item() if init_vx is not None else float("nan")
        stopball_fired  = sb_flag[0].item()  if sb_flag  is not None else False
        softstop_fired  = ss_flag[0].item()  if ss_flag  is not None else False
        cleanstop_fired = cs_flag[0].item()  if cs_flag  is not None else False

        # Foot heights + slip velocities.
        from simple_goalkeeper.robots.t1_constants import HOME_KEYFRAME  # noqa: F401 (unused val)
        foot_ids = robot.find_bodies(["left_foot_link", "right_foot_link"])[0]
        foot_pos_w  = robot.data.body_link_pos_w[0, foot_ids, :]   # [2, 3]
        foot_vel_w  = robot.data.body_link_lin_vel_w[0, foot_ids, :]  # [2, 3]
        floor_z = env.scene.env_origins[0, 2]
        lf_h = (foot_pos_w[0, 2] - floor_z).item()
        rf_h = (foot_pos_w[1, 2] - floor_z).item()
        _CONTACT_H = 0.06  # foot height threshold to consider "in contact"
        lf_contact = lf_h < _CONTACT_H
        rf_contact = rf_h < _CONTACT_H
        lf_slip = foot_vel_w[0, :2].norm().item() if lf_contact else 0.0
        rf_slip = foot_vel_w[1, :2].norm().item() if rf_contact else 0.0

        # Base (Trunk root) and shank heights above floor.
        base_h = (robot.data.root_link_pos_w[0, 2] - floor_z).item()
        shank_ids = robot.find_bodies(["Shank_Left", "Shank_Right"])[0]
        shank_pos_w = robot.data.body_link_pos_w[0, shank_ids, :]  # [2, 3]
        lsh_h = (shank_pos_w[0, 2] - floor_z).item()
        rsh_h = (shank_pos_w[1, 2] - floor_z).item()
        self._min_base_h = min(self._min_base_h, base_h)
        self._min_lsh_h = min(self._min_lsh_h, lsh_h)
        self._min_rsh_h = min(self._min_rsh_h, rsh_h)

        # DEBUG 2026-07-23 (TEMPORARY, remove after landing-gate investigation):
        # per-episode blue-landing accumulators (see __init__ + [EpEnd] print).
        wide_t = getattr(env, "_blue_wide", None)
        if wide_t is not None and bool(wide_t[0].item()):
            self._ep_was_wide = True
            dist_t = getattr(env, "_blue_dbg_dist", None)
            settle_t = getattr(env, "_blue_dbg_settle", None)
            if dist_t is not None:
                self._min_blue_dist = min(self._min_blue_dist, dist_t[0].item())
            if settle_t is not None:
                self._max_settle = max(self._max_settle, int(settle_t[0].item()))
        landed_t = getattr(env, "_blue_landed", None)
        if landed_t is not None and bool(landed_t[0].item()):
            self._ep_landed = True

        ball_speed = bv.norm().item()

        # Interception point in robot's local frame.
        # crossing_y is world-frame Y; express it as robot-local lateral offset.
        cross_y_w = getattr(env, "_ball_crossing_y", None)
        if cross_y_w is not None:
            from mjlab.utils.lab_api.math import quat_apply_inverse
            origin = env.scene.env_origins[0]
            cross_pt_w = torch.tensor(
                [origin[0].item(), cross_y_w[0].item(), origin[2].item() + 0.1],
                device=env.device,
            )
            robot_pos_w0 = robot.data.root_link_pos_w[0]
            robot_quat_w0 = robot.data.root_link_quat_w[0]
            cross_local = quat_apply_inverse(robot_quat_w0.unsqueeze(0),
                                             (cross_pt_w - robot_pos_w0).unsqueeze(0))[0]
            int_x = cross_local[0].item()   # forward in robot frame (+= in front)
            int_y = cross_local[1].item()   # lateral in robot frame (+= right)
        else:
            int_x = float("nan")
            int_y = float("nan")

        # Per-step reward breakdown (unscaled rate = raw_value * weight, dt-scaling
        # divided back out -- see mjlab RewardManager._step_reward docstring), read
        # the same way termination_manager.get_term() is read above.
        rew_mgr = getattr(env, "reward_manager", None)
        if rew_mgr is not None:
            reward_terms = dict(rew_mgr.get_active_iterable_terms(0))
            reward_total = sum(v[0] for v in reward_terms.values())
            rew_dbg = " | REW total={:.2f} [{}]".format(
                reward_total,
                " ".join(f"{name}={v[0]:.2f}" for name, v in reward_terms.items()),
            )
        else:
            rew_dbg = ""

        flags = (
            f"{'SB✓' if stopball_fired else 'SB·'} "
            f"{'SS✓' if softstop_fired else 'SS·'} "
            f"{'CS✓' if cleanstop_fired else 'CS·'} "
            f"{'WF✓' if self._wf_flash else 'WF·'} "
            f"{'HD✓' if self._head_flash else 'HD·'}"
        )
        lf_tag = f"{'G' if lf_contact else 'A'}{lf_slip:.2f}"
        rf_tag = f"{'G' if rf_contact else 'A'}{rf_slip:.2f}"

        # DEBUG 2026-07-23 (TEMPORARY, remove after landing-gate investigation):
        # blue-ball landing-gate internals, cached onto env by rewards.py's
        # _get_reach_target_y (env._blue_dbg_*). Only meaningful once wide=True.
        blue_dbg = ""
        wide_t = getattr(env, "_blue_wide", None)
        if wide_t is not None and bool(wide_t[0].item()):
            dist_t = getattr(env, "_blue_dbg_dist", None)
            speed_t = getattr(env, "_blue_dbg_speed", None)
            contact_t = getattr(env, "_blue_dbg_contact", None)
            settle_t = getattr(env, "_blue_dbg_settle", None)
            foot_idx_t = getattr(env, "_blue_dbg_foot_idx", None)
            radius = getattr(env, "_blue_dbg_radius", float("nan"))
            speed_th = getattr(env, "_blue_dbg_speed_th", float("nan"))
            landed_t = getattr(env, "_blue_landed", None)
            half_off_t = getattr(env, "_blue_dbg_half_off", None)
            full_off_t = getattr(env, "_blue_dbg_full_off", None)
            foot_off_t = getattr(env, "_blue_dbg_foot_off", None)
            wide_dist_t = getattr(env, "_blue_dbg_wide_by_dist", None)
            airborne_t = getattr(env, "_blue_dbg_was_airborne", None)
            touch_ball_t = getattr(env, "_blue_dbg_touching_ball", None)
            candidate_t = getattr(env, "_blue_dbg_candidate", None)
            first_call_t = getattr(env, "_blue_dbg_first_call", None)
            gate_wide_t = getattr(env, "_blue_dbg_wide", None)
            ep_len_t = getattr(env, "_blue_dbg_ep_len", None)
            last_settle_before_t = getattr(env, "_blue_dbg_last_settle_step_before", None)
            blue_dbg = (
                f" | BLUE dist={dist_t[0].item() if dist_t is not None else float('nan'):.2f}"
                f"(<{radius:.2f}) "
                f"spd={speed_t[0].item() if speed_t is not None else float('nan'):.2f}"
                f"(<{speed_th:.2f}) "
                f"contact={bool(contact_t[0].item()) if contact_t is not None else None} "
                f"foot={foot_idx_t[0].item() if foot_idx_t is not None else None} "
                f"settle={settle_t[0].item() if settle_t is not None else None}/3 "
                f"landed={bool(landed_t[0].item()) if landed_t is not None else None} || "
                f"halfOff={half_off_t[0].item() if half_off_t is not None else float('nan'):+.2f} "
                f"fullOff={full_off_t[0].item() if full_off_t is not None else float('nan'):+.2f} "
                f"footOff={foot_off_t[0].item() if foot_off_t is not None else float('nan'):+.2f} "
                f"wideByDist={bool(wide_dist_t[0].item()) if wide_dist_t is not None else None} "
                f"wasAirborne={bool(airborne_t[0].item()) if airborne_t is not None else None} "
                f"touchingBall={bool(touch_ball_t[0].item()) if touch_ball_t is not None else None} || "
                f"candidate={bool(candidate_t[0].item()) if candidate_t is not None else None} "
                f"firstCallThisTick={bool(first_call_t[0].item()) if first_call_t is not None else None} "
                f"gateWide={bool(gate_wide_t[0].item()) if gate_wide_t is not None else None} || "
                f"epLen={ep_len_t[0].item() if ep_len_t is not None else None} "
                f"lastSettleStepBefore={last_settle_before_t[0].item() if last_settle_before_t is not None else None}"
            )

        # DEBUG 2026-07-23 (TEMPORARY): epLen/settle/candidate moved to the
        # FRONT of the line -- terminal column-width truncation of this \r
        # overwriting line was hiding them on all but the last print of a
        # burst, making it impossible to see settle's progression tick to
        # tick across a whole pasted trace.
        ep_len_front = getattr(env, "_blue_dbg_ep_len", None)
        settle_front = getattr(env, "_blue_dbg_settle", None)
        candidate_front = getattr(env, "_blue_dbg_candidate", None)
        front_dbg = (
            f"epLen={ep_len_front[0].item() if ep_len_front is not None else None} "
            f"settle={settle_front[0].item() if settle_front is not None else None} "
            f"cand={bool(candidate_front[0].item()) if candidate_front is not None else None} | "
        )
        print(
            f"\rEp{self._ep:3d} | {front_dbg}"
            f"bvx={bv[0].item():+6.2f} bvy={bv[1].item():+5.2f} spd={ball_speed:.2f} "
            f"bx={bx_local:+5.2f} | "
            f"int(x={int_x:+5.2f} y={int_y:+5.2f}) | "
            f"dvx={delta_vx:+5.2f} | "
            f"LF={lf_h:.2f}({lf_tag}) RF={rf_h:.2f}({rf_tag}) | "
            f"base={base_h:.2f} Lsh={lsh_h:.2f} Rsh={rsh_h:.2f} | "
            f"{flags}{blue_dbg}{rew_dbg}",
            end="",
            flush=True,
            file=stderr,
        )
        return actions

    # Allow duck-typing with plain callables (reset hook used by runner).
    def reset(self) -> None:
        reset_fn = getattr(self._policy, "reset", None)
        if reset_fn is not None:
            reset_fn()


def _patch_viewer_intercept_vis(native_viewer: "NativeMujocoViewer", env) -> None:
    """Monkey-patch NativeMujocoViewer to draw the predicted interception sphere.

    Adds a sphere at the current reach target (env 0) and a vertical line from
    floor to sphere so it's visible from any camera angle. The sphere updates
    each render frame — it moves when a new episode starts and the crossing_y
    changes.

    Two-stage wide-crossing visualization (v2 reimplementation, 2026-07-23, of
    the blue-ball-waypoint branch's mechanism -- see rewards.py's
    _get_reach_target_y). When |crossing_y - start_y| > wide_threshold (0.65
    as of the 2026-07-23 widening -- kept symbolic here rather than
    hardcoded so this comment can't drift out of sync with rewards.py's
    actual default again) or the region is a far region, and the assigned
    foot has not yet landed at the midpoint,
    draws a BLUE sphere there instead of the usual green one. Once landing has
    occurred (or the crossing is narrow), draws the usual GREEN sphere at the
    full crossing point. Lets a human watching sgk_play confirm landing
    timing visually. Reads env._blue_wide/_blue_landed directly -- cached
    every step by _get_reach_target_y -- rather than recomputing the
    wide/region check here, so the marker can't drift out of sync with what
    footreach/foot_proximity/stopball/softstop are actually gating on.
    """
    orig_update = native_viewer._update_debug_visualizers

    def _patched_update(viewer_handle: "mujoco.viewer.Handle") -> None:
        orig_update(viewer_handle)  # run normal debug vis (resets ngeom=0)

        raw_env = env.unwrapped if hasattr(env, "unwrapped") else env
        cross_y_t = getattr(raw_env, "_ball_crossing_y", None)
        if cross_y_t is None:
            return

        scn = viewer_handle.user_scn
        origins = raw_env.scene.env_origins[0].cpu().numpy()
        goal_x = float(origins[0])
        cross_y = float(cross_y_t[0].item())
        floor_z = float(origins[2])
        sphere_z = floor_z + 0.12

        def _add_sphere(x: float, y: float, z: float, r: float, rgba) -> None:
            if scn.ngeom >= scn.maxgeom:
                return
            scn.ngeom += 1
            g = scn.geoms[scn.ngeom - 1]
            g.category = mujoco.mjtCatBit.mjCAT_DECOR
            mujoco.mjv_initGeom(
                geom=g,
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=np.array([r, r, r], dtype=np.float64),
                pos=np.array([x, y, z], dtype=np.float64),
                mat=np.eye(3, dtype=np.float64).flatten(),
                rgba=np.array(rgba, dtype=np.float32),
            )

        def _add_line(from_: np.ndarray, to: np.ndarray, width: float, rgba) -> None:
            if scn.ngeom >= scn.maxgeom:
                return
            scn.ngeom += 1
            g = scn.geoms[scn.ngeom - 1]
            g.category = mujoco.mjtCatBit.mjCAT_DECOR
            mujoco.mjv_initGeom(
                geom=g,
                type=mujoco.mjtGeom.mjGEOM_LINE,
                size=np.zeros(3, dtype=np.float64),
                pos=np.zeros(3, dtype=np.float64),
                mat=np.zeros(9, dtype=np.float64),
                rgba=np.array(rgba, dtype=np.float32),
            )
            mujoco.mjv_connector(
                geom=g,
                type=mujoco.mjtGeom.mjGEOM_LINE,
                width=width,
                from_=from_,
                to=to,
            )

        # Two-stage schedule -- read the cached state _get_reach_target_y
        # (rewards.py) sets every step, rather than recomputing wide/region
        # here (see docstring).
        wide_t = getattr(raw_env, "_blue_wide", None)
        landed_t = getattr(raw_env, "_blue_landed", None)
        wide = bool(wide_t[0].item()) if wide_t is not None else False
        landed = bool(landed_t[0].item()) if landed_t is not None else False

        if wide and not landed:
            # Phase 1: BLUE sphere at the midpoint — half the distance, half as far.
            start_y = float(origins[1])
            mid_y = start_y + (cross_y - start_y) / 2.0
            _add_sphere(goal_x, mid_y, sphere_z, 0.08, [0.15, 0.4, 1.0, 0.75])
            _add_line(
                np.array([goal_x, mid_y, floor_z], dtype=np.float64),
                np.array([goal_x, mid_y, sphere_z], dtype=np.float64),
                0.008, [0.15, 0.4, 1.0, 0.6],
            )
        else:
            # Phase 2 (or narrow crossing): GREEN sphere at the full crossing point.
            _add_sphere(goal_x, cross_y, sphere_z, 0.08, [0.1, 1.0, 0.2, 0.75])
            _add_line(
                np.array([goal_x, cross_y, floor_z], dtype=np.float64),
                np.array([goal_x, cross_y, sphere_z], dtype=np.float64),
                0.008, [0.1, 1.0, 0.2, 0.6],
            )

    native_viewer._update_debug_visualizers = _patched_update


def _compute_foot_slip(env, env_idx: int) -> tuple[float, float]:
    """Per-foot horizontal slip speed (m/s), 0.0 when that foot is not in
    genuine GROUND contact.

    Contact test mirrors the real `feet_slippage` reward function exactly
    (rewards.py:2258-2270, FIX 2026-07-27): "in contact" is `feet_contact`
    (any contact) with genuine foot-ball contact (`ball_contact` sensor)
    excluded, NOT a height threshold -- a height proxy still reports "in
    contact" for a few frames into a fast push-off (foot below the height
    cutoff but already swinging at step speed), producing a false slip spike
    the actual sensor-based reward never sees. Using the exact same contact
    logic here keeps this plot directly comparable to the feet_slippage
    reward plot next to it -- without the ball exclusion, this plot still
    spikes on every foot-ball touch even though the reward (post-fix) no
    longer penalizes it, which looks like the fix isn't working when it is.
    """
    raw_env = env.unwrapped if hasattr(env, "unwrapped") else env
    robot = raw_env.scene["robot"]
    foot_ids = robot.find_bodies(["left_foot_link", "right_foot_link"])[0]
    foot_vel_w = robot.data.body_link_lin_vel_w[env_idx, foot_ids, :]  # [2, 3]
    found = raw_env.scene["feet_contact"].data.found[env_idx]  # [8]
    ball_found = raw_env.scene["ball_contact"].data.found[env_idx]  # [8], same layout
    lf_contact = bool((found[:4] > 0).any().item()) and not bool((ball_found[:4] > 0).any().item())
    rf_contact = bool((found[4:] > 0).any().item()) and not bool((ball_found[4:] > 0).any().item())
    lf_slip = foot_vel_w[0, :2].norm().item() if lf_contact else 0.0
    rf_slip = foot_vel_w[1, :2].norm().item() if rf_contact else 0.0
    return lf_slip, rf_slip


def _patch_viewer_feet_slip_plots(native_viewer: "NativeMujocoViewer", env) -> None:
    """Add two extra P-panel plots ("left_foot_slip"/"right_foot_slip") showing
    raw per-foot horizontal slip speed in m/s, alongside the existing per-reward-term
    plots. The stock feet_slippage reward plot shows exp(-10*slip) squashed into
    [0,1] -- these show the underlying speed directly, which is easier to read.
    """
    orig_setup = native_viewer.setup
    orig_update_reward_figures = native_viewer._update_reward_figures

    _SLIP_TERM_NAMES = ("left_foot_slip", "right_foot_slip")
    # Existing reward-term plot(s) to also force into the visible slots, so
    # the raw slip plots above have the actual reward to compare against.
    _PROMOTED_REWARD_TERMS = ("feet_slippage",)

    def _patched_setup() -> None:
        orig_setup()
        from mjlab.viewer.native.viewer import make_empty_figure
        cfg = native_viewer._plot_cfg
        for name in _SLIP_TERM_NAMES:
            native_viewer._figures[name] = make_empty_figure(
                name, cfg.grid_size, cfg.init_yrange, cfg.history, cfg.background_alpha,
            )
            native_viewer._histories[name] = deque(maxlen=cfg.history)
            native_viewer._yrange[name] = cfg.init_yrange
            native_viewer._scale[name] = 1.0
        # Reorder (not append): _update_reward_figures only renders
        # _term_names[:max_viewports] (default 12) -- this task registers 41
        # active reward terms, so anything past index 12 is silently never
        # rendered no matter what _show_plots/_is_paused say. feet_slippage
        # itself sits at position ~30 in declaration order and was NEVER
        # visible even before this patch. Move the two slip plots plus
        # feet_slippage to the front so all three are always in view.
        rest = [n for n in native_viewer._term_names if n not in _PROMOTED_REWARD_TERMS]
        promoted = [n for n in _PROMOTED_REWARD_TERMS if n in native_viewer._term_names]
        native_viewer._term_names = list(_SLIP_TERM_NAMES) + promoted + rest

    def _patched_update_reward_figures(viewer_handle: "mujoco.viewer.Handle") -> None:
        if native_viewer._show_plots and native_viewer._term_names and not native_viewer._is_paused:
            lf_slip, rf_slip = _compute_foot_slip(env, native_viewer.env_idx)
            native_viewer._append_point("left_foot_slip", lf_slip)
            native_viewer._append_point("right_foot_slip", rf_slip)
            native_viewer._write_history_to_figure("left_foot_slip")
            native_viewer._write_history_to_figure("right_foot_slip")
        orig_update_reward_figures(viewer_handle)

    native_viewer.setup = _patched_setup
    native_viewer._update_reward_figures = _patched_update_reward_figures


_WRONG_FOOT_FLASH_STEPS = 15
"""Steps to hold the wrong-foot-contact plot signal high after a touch (~0.3s
at dt=0.02s). A genuine ball-vs-foot contact often lasts only 1-3 physics
steps, well under the ~7-sample (2.3% of the 300-sample history window)
floor _write_history_to_figure's percentile-based autoscale (viewer.py,
p_lo=2/p_hi=98) needs to widen the y-range at all -- below that floor the
spike is computed correctly but clipped out of the plotted range entirely,
looking like no reaction. 15 steps clears that floor with margin."""


def _compute_wrong_foot_contact_flash(env, env_idx: int) -> tuple[float, float]:
    """Raw "bad ball contact" signals for the viewer's P-panel, each latched
    high for _WRONG_FOOT_FLASH_STEPS steps after a detected touch: (1) the
    WRONG (non-assigned) foot touching the ball, (2) the head/chin touching
    the ball at all.

    (1) mirrors penalize_wrong_foot_ball_contact's own detection exactly (same
    "ball_contact" sensor + _get_correct_foot_idx) -- does NOT read the reward
    term itself, since promoting that raw reward value (tried first) plots a
    signal too sparse for the viewer's autoscale to ever show (see
    docs/BugFixes.md).

    (2) DEBUG 2026-07-28: added after the wrong-foot flash (1) still showed no
    reaction and the user asked whether the ball might actually be hitting the
    chin/head instead of a foot -- "ball_contact"/"feet_contact" only match
    foot geoms, so a head touch is invisible to both and to (1) above. Reads
    the new "head_ball_contact" sensor (goalkeeper_env_cfg.py) directly.

    (1) EXTENDED 2026-07-28 (same day, later): user then spotted an orange
    MuJoCo contact-point dot between the trailing leg and the ball while (1)
    still read 0 -- confirmed via a real-checkpoint replay that the shin/knee
    (not covered by ball_contact's foot[1-4]_collision pattern) genuinely
    touches the ball, sometimes on the wrong side. Now also ORs in
    "leg_ball_contact" (goalkeeper_env_cfg.py), mirroring
    penalize_wrong_foot_ball_contact's own fix exactly so this flash can
    never drift out of sync with what the reward actually penalizes.

    Both are viewer-only display latches with no effect on training/reward.
    """
    raw_env = env.unwrapped if hasattr(env, "unwrapped") else env
    from simple_goalkeeper.mdp.rewards import _get_correct_foot_idx

    foot_idx = _get_correct_foot_idx(raw_env, "ball")  # (N,) 0=left, 1=right
    found = raw_env.scene["ball_contact"].data.found  # [B, 8]
    left_touch = (found[:, :4] > 0).any(dim=-1)
    right_touch = (found[:, 4:] > 0).any(dim=-1)

    leg_found = raw_env.scene["leg_ball_contact"].data.found  # [B, 4]: 0-1=left, 2-3=right
    left_leg_touch = (leg_found[:, :2] > 0).any(dim=-1)
    right_leg_touch = (leg_found[:, 2:] > 0).any(dim=-1)

    foot_touch = torch.stack(
        [left_touch | left_leg_touch, right_touch | right_leg_touch], dim=-1
    )  # (B, 2)
    wrong_foot_idx = 1 - foot_idx
    wrong_touch = foot_touch[torch.arange(raw_env.num_envs, device=raw_env.device), wrong_foot_idx]

    head_found = raw_env.scene["head_ball_contact"].data.found  # [B, N]
    head_touch = (head_found > 0).any(dim=-1)

    if not hasattr(raw_env, "_wrong_foot_flash_counter"):
        raw_env._wrong_foot_flash_counter = torch.zeros(
            raw_env.num_envs, dtype=torch.long, device=raw_env.device
        )
    if not hasattr(raw_env, "_head_contact_flash_counter"):
        raw_env._head_contact_flash_counter = torch.zeros(
            raw_env.num_envs, dtype=torch.long, device=raw_env.device
        )
    just_reset = raw_env.episode_length_buf <= 1

    counter = raw_env._wrong_foot_flash_counter
    counter[just_reset] = 0
    counter[wrong_touch] = _WRONG_FOOT_FLASH_STEPS
    counter[~wrong_touch & ~just_reset] = torch.clamp(counter[~wrong_touch & ~just_reset] - 1, min=0)

    head_counter = raw_env._head_contact_flash_counter
    head_counter[just_reset] = 0
    head_counter[head_touch] = _WRONG_FOOT_FLASH_STEPS
    head_counter[~head_touch & ~just_reset] = torch.clamp(head_counter[~head_touch & ~just_reset] - 1, min=0)

    return float(counter[env_idx].item() > 0), float(head_counter[env_idx].item() > 0)


def _patch_viewer_wrong_foot_contact_plot(native_viewer: "NativeMujocoViewer", env) -> None:
    """Add two P-panel plots ("wrong_foot_ball_contact"/"head_ball_contact")
    showing the latched raw bad-contact signals (see
    _compute_wrong_foot_contact_flash), alongside the existing per-reward-term
    plots. Mirrors _patch_viewer_feet_slip_plots' structure exactly.
    """
    orig_setup = native_viewer.setup
    orig_update_reward_figures = native_viewer._update_reward_figures

    _TERM_NAMES = ("wrong_foot_ball_contact", "head_ball_contact")

    def _patched_setup() -> None:
        orig_setup()
        from mjlab.viewer.native.viewer import make_empty_figure
        cfg = native_viewer._plot_cfg
        for name in _TERM_NAMES:
            native_viewer._figures[name] = make_empty_figure(
                name, cfg.grid_size, cfg.init_yrange, cfg.history, cfg.background_alpha,
            )
            native_viewer._histories[name] = deque(maxlen=cfg.history)
            native_viewer._yrange[name] = cfg.init_yrange
            native_viewer._scale[name] = 1.0
        # Front of the list -- same reasoning as the other promotions above:
        # this task's 41 active reward terms exceed max_viewports (12), so
        # anything not moved to the front is silently never rendered.
        rest = [n for n in native_viewer._term_names]
        native_viewer._term_names = list(_TERM_NAMES) + rest

    def _patched_update_reward_figures(viewer_handle: "mujoco.viewer.Handle") -> None:
        if native_viewer._show_plots and native_viewer._term_names and not native_viewer._is_paused:
            wrong_foot_flash, head_flash = _compute_wrong_foot_contact_flash(env, native_viewer.env_idx)
            native_viewer._append_point("wrong_foot_ball_contact", wrong_foot_flash)
            native_viewer._write_history_to_figure("wrong_foot_ball_contact")
            native_viewer._append_point("head_ball_contact", head_flash)
            native_viewer._write_history_to_figure("head_ball_contact")
        orig_update_reward_figures(viewer_handle)

    native_viewer.setup = _patched_setup
    native_viewer._update_reward_figures = _patched_update_reward_figures


def _compute_foot_yaw_error(env, env_idx: int) -> tuple[float, float]:
    """Signed yaw deviation (degrees) of each foot's forward axis from world +X.

    0 = foot's toe points along world +X -- the "facing the field" direction
    this whole task assumes the robot holds (ang_vel_z's -0.5 weight, and the
    ball's -X approach direction, see CLAUDE.md's Frame Convention section).
    Positive = rotated counter-clockwise from forward (toe swings toward
    +Y/left), negative = clockwise (toward -Y/right). Foot-local +X is the
    toe-forward axis -- same convention feet_slippage's toe-tip contact point
    uses (_TOE_X_LOCAL).
    """
    raw_env = env.unwrapped if hasattr(env, "unwrapped") else env
    robot = raw_env.scene["robot"]
    foot_ids = robot.find_bodies(["left_foot_link", "right_foot_link"])[0]
    foot_quat_w = robot.data.body_link_quat_w[env_idx, foot_ids, :]  # [2, 4]
    forward_local = torch.tensor(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], device=foot_quat_w.device, dtype=foot_quat_w.dtype,
    )
    forward_world = quat_apply(foot_quat_w, forward_local)  # [2, 3]
    yaw_deg = torch.rad2deg(torch.atan2(forward_world[:, 1], forward_world[:, 0]))  # [2]
    return yaw_deg[0].item(), yaw_deg[1].item()


def _patch_viewer_foot_orientation_plots(native_viewer: "NativeMujocoViewer", env) -> None:
    """Add two P-panel plots ("left_foot_yaw_deg"/"right_foot_yaw_deg") showing
    each foot's signed yaw deviation from forward (world +X) in degrees.

    Added per user report: the non-leading (trailing/non-assigned) foot ends
    up pointing in visually odd directions after a save. No dedicated reward
    currently targets foot HEADING directly -- the closest existing term is
    `postlegdofpos` (rewards.py:1846), which pulls all 12 leg joints
    (including Hip_Yaw -- T1 has no ankle-yaw actuator, so hip yaw is what
    actually controls where a foot points) back toward their default values,
    but only post-save and only in joint space, not as a direct world-frame
    heading target. Promoted here alongside the two new plots so the two can
    be compared -- if yaw error stays large while postlegdofpos is near its
    ceiling, that's evidence the indirect joint-space pull isn't sufficient
    and a dedicated heading reward may be warranted (a separate, bigger
    change, not implemented here).
    """
    orig_setup = native_viewer.setup
    orig_update_reward_figures = native_viewer._update_reward_figures

    _YAW_TERM_NAMES = ("left_foot_yaw_deg", "right_foot_yaw_deg")
    _PROMOTED_REWARD_TERMS = ("postlegdofpos",)

    def _patched_setup() -> None:
        orig_setup()
        from mjlab.viewer.native.viewer import make_empty_figure
        cfg = native_viewer._plot_cfg
        for name in _YAW_TERM_NAMES:
            native_viewer._figures[name] = make_empty_figure(
                name, cfg.grid_size, cfg.init_yrange, cfg.history, cfg.background_alpha,
            )
            native_viewer._histories[name] = deque(maxlen=cfg.history)
            native_viewer._yrange[name] = cfg.init_yrange
            native_viewer._scale[name] = 1.0
        rest = [n for n in native_viewer._term_names if n not in _PROMOTED_REWARD_TERMS]
        promoted = [n for n in _PROMOTED_REWARD_TERMS if n in native_viewer._term_names]
        native_viewer._term_names = list(_YAW_TERM_NAMES) + promoted + rest

    def _patched_update_reward_figures(viewer_handle: "mujoco.viewer.Handle") -> None:
        if native_viewer._show_plots and native_viewer._term_names and not native_viewer._is_paused:
            lf_yaw, rf_yaw = _compute_foot_yaw_error(env, native_viewer.env_idx)
            native_viewer._append_point("left_foot_yaw_deg", lf_yaw)
            native_viewer._append_point("right_foot_yaw_deg", rf_yaw)
            native_viewer._write_history_to_figure("left_foot_yaw_deg")
            native_viewer._write_history_to_figure("right_foot_yaw_deg")
        orig_update_reward_figures(viewer_handle)

    native_viewer.setup = _patched_setup
    native_viewer._update_reward_figures = _patched_update_reward_figures


def run_play(task_id: str, cfg: PlayConfig) -> None:
    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    env_cfg = load_env_cfg(task_id, play=True)
    agent_cfg = load_rl_cfg(task_id)
    assert isinstance(agent_cfg, (AMPRunnerCfg, dict)), (
        f"Task '{task_id}' is not an AMP task — got {type(agent_cfg).__name__}."
    )
    # Multi-disc tasks (e.g. Mjlab-BeyondAMP-Goalkeeper-T1-MultiDisc) register a
    # plain dict rl_cfg consumed by HimAMPOnPolicyRunner instead of the stock
    # AMPOnPolicyRunner -- mirrors train.py's TrainConfig.from_task dispatch.
    is_multidisc = isinstance(agent_cfg, dict)

    if cfg.num_envs is not None:
        env_cfg.scene.num_envs = cfg.num_envs
    if cfg.no_terminations:
        env_cfg.terminations = {}
        print("[INFO]: Terminations disabled")
    if cfg.rsi:
        # RSI already popped in play mode by default; re-add it when requested.
        from simple_goalkeeper.mdp.events import reset_from_motion_data as _rsi_fn
        from mjlab.managers.event_manager import EventTermCfg as _EvtCfg
        env_cfg.events["reset_from_motion_data"] = _EvtCfg(
            func=_rsi_fn, mode="reset",
        )
        print("[INFO]: RSI enabled — episodes start from random motion frames")

    if cfg.difficulty_outer_only_frac is not None:
        if "reset_ball" not in env_cfg.events:
            raise RuntimeError(
                "--difficulty-outer-only-frac requires a 'reset_ball' event "
                f"(task '{task_id}' has none)."
            )
        env_cfg.events["reset_ball"].params["y_end_outer_frac"] = cfg.difficulty_outer_only_frac
        print(
            f"[INFO]: Outer-only difficulty enabled — ball's lateral target offset "
            f"pinned to the top {100 * (1 - cfg.difficulty_outer_only_frac):.0f}% "
            f"of each region's range"
        )

    if cfg.force_region is not None:
        if "assign_static_regions" not in env_cfg.events:
            raise RuntimeError(
                "--force-region requires an 'assign_static_regions' event "
                f"(task '{task_id}' has none — not a region-conditioned task)."
            )
        from simple_goalkeeper.mdp.regions import REGION_NAMES, pin_region_on_reset
        from mjlab.managers.event_manager import EventTermCfg as _EvtCfg
        region_id = REGION_NAMES.index(cfg.force_region)
        env_cfg.events["assign_static_regions"] = _EvtCfg(
            func=pin_region_on_reset, mode="reset", params={"region_id": region_id},
        )
        print(f"[INFO]: Region pinned to '{cfg.force_region}' — every episode will be this region only")

    # Override motion file for WithOverlay task if specified on CLI.
    if cfg.motion_file is not None and "motion_ghost" in env_cfg.commands:
        from simple_goalkeeper.tasks.goalkeeper_env_cfg import (
            _MOTIONS_DATA_DIR,
            _T1_HEADLESS_BODY_NAMES,
        )
        from simple_goalkeeper.mdp.commands import GhostMotionCommandCfg
        from mjlab.tasks.tracking.mdp.commands import MotionCommandCfg

        motion_path = Path(cfg.motion_file)
        if not motion_path.exists():
            raise FileNotFoundError(f"Motion file not found: {motion_path}")
        print(f"[INFO]: Using motion file: {motion_path.name}")
        env_cfg.commands["motion_ghost"] = GhostMotionCommandCfg(
            motion_file=str(motion_path),
            anchor_body_name="Trunk",
            body_names=_T1_HEADLESS_BODY_NAMES,
            entity_name="robot",
            debug_vis=True,
            resampling_time_range=(10.0, 10.0),
            viz=MotionCommandCfg.VizCfg(mode="ghost", ghost_color=(0.3, 0.8, 0.4, 0.45)),
        )

    os.environ.setdefault("MUJOCO_GL", "egl")
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    if cfg.difficulty is not None:
        env._ball_difficulty = float(cfg.difficulty)
        print(f"[INFO]: Ball difficulty overridden to {cfg.difficulty} (0=easy, 1=hard)")

    import math as _math
    d = float(getattr(env, "_ball_difficulty", 1.0))
    _EASY_DIST = (2.0, 2.0); _HARD_DIST = (2.0, 3.5)
    _EASY_T    = (0.9, 1.3); _HARD_T    = (0.7, 1.1)
    _Y_INNER = 0.15; _Y_OUTER = 0.35; _Y_MAX = 1.0
    dist_lo = _EASY_DIST[0] + d * (_HARD_DIST[0] - _EASY_DIST[0])
    dist_hi = _EASY_DIST[1] + d * (_HARD_DIST[1] - _EASY_DIST[1])
    t_lo    = _EASY_T[0]    + d * (_HARD_T[0]    - _EASY_T[0])
    t_hi    = _EASY_T[1]    + d * (_HARD_T[1]    - _EASY_T[1])
    y_inner = max(_Y_INNER * (1.0 - d), 0.1)
    y_outer = _Y_OUTER + (_Y_MAX - _Y_OUTER) * d
    spd_min = (dist_lo + 0.3) / t_hi
    spd_max = _math.sqrt((dist_hi + 0.3)**2 + (y_outer + 0.3)**2) / t_lo
    print(
        f"[INFO]: Ball spawn at d={d:.2f} — "
        f"dist {dist_lo:.1f}–{dist_hi:.1f} m | "
        f"t_flight {t_lo:.2f}–{t_hi:.2f} s | "
        f"y_end ±{y_inner:.2f}–{y_outer:.2f} m | "
        f"speed ~{spd_min:.1f}–{spd_max:.1f} m/s"
    )

    if is_multidisc:
        # Multi-disc RSI/motion loading never reads AMPEnvWrapper.motion_dataset,
        # and its amp_data is a dict of 4 per-region cfgs the single-dataset
        # wrapper path cannot consume -- mirrors train.py's run_train. clip_actions
        # has no override in the dict, so it keeps mjlab's own default (None),
        # same value used at training time.
        env = AMPEnvWrapper(env, clip_actions=None, motion_dataset=None)
    else:
        env = AMPEnvWrapper(env, clip_actions=agent_cfg.clip_actions, motion_dataset=agent_cfg.amp_data)

    DUMMY_MODE = cfg.agent in {"zero", "random"}
    if DUMMY_MODE:
        action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
        if cfg.agent == "zero":
            def policy(obs: torch.Tensor) -> torch.Tensor:
                return torch.zeros(action_shape, device=device)
        else:
            def policy(obs: torch.Tensor) -> torch.Tensor:
                return 2 * torch.rand(action_shape, device=device) - 1
    else:
        if cfg.checkpoint_file is None:
            raise ValueError("--checkpoint-file is required for --agent trained")
        resume_path = Path(cfg.checkpoint_file)
        if not resume_path.is_absolute() and not resume_path.exists():
            # Walk up directory tree to find the checkpoint relative to a parent dir.
            # Allows passing paths like "BEPImitationLearning/SimpleGoalKeeper/logs/..."
            # from inside SimpleGoalKeeper/ — resolves from /home/robocup/IsaakB/.
            parent = Path.cwd()
            for _ in range(6):
                parent = parent.parent
                candidate = parent / resume_path
                if candidate.exists():
                    resume_path = candidate
                    break
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
        print(f"[INFO] Loading checkpoint: {resume_path.name}")
        if is_multidisc:
            runner_cls = load_runner_cls(task_id)
            runner = runner_cls(env, agent_cfg, log_dir=None, device=device)
            runner.load(str(resume_path), load_optimizer=False)
            act_inference = runner.get_inference_policy(device=device)

            # HimActorCritic.act_inference needs (obs_current, obs_history), not
            # the single obs tensor the viewer's loop calls policy(obs) with.
            # env.get_observations() (what the viewer passes as `obs`) is the
            # same "actor" history-stacked group as obs_history, so it's reused
            # directly -- only obs_current needs a separate fetch.
            def policy(obs_history: torch.Tensor) -> torch.Tensor:
                obs_current = _get_actor_current_obs(env)
                return act_inference(obs_current, obs_history)
        else:
            runner = AMPOnPolicyRunner(env, asdict(agent_cfg), log_dir=None, device=device)
            runner.load(str(resume_path), load_optimizer=False)
            policy = runner.get_inference_policy(device=device)

    # Wrap policy with analytics printer (always constructed, enabled by flag).
    analytics = AnalyticsPolicy(policy, env, enabled=cfg.analytics)
    final_policy = analytics

    if cfg.viewer == "auto":
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        resolved_viewer = "native" if has_display else "viser"
    else:
        resolved_viewer = cfg.viewer

    if resolved_viewer == "native":
        from mjlab.viewer.native.keys import KEY_V
        def _key_cb(key: int) -> None:
            if key == KEY_V:
                analytics.toggle()
        native_viewer = NativeMujocoViewer(env, final_policy, key_callback=_key_cb)
        _patch_viewer_intercept_vis(native_viewer, env)
        _patch_viewer_feet_slip_plots(native_viewer, env)
        _patch_viewer_foot_orientation_plots(native_viewer, env)
        _patch_viewer_wrong_foot_contact_plot(native_viewer, env)
        native_viewer.run()
    elif resolved_viewer == "viser":
        ViserPlayViewer(env, final_policy).run()
    else:
        raise RuntimeError(f"Unsupported viewer: {resolved_viewer}")

    env.close()


def main() -> None:
    import mjlab.tasks  # noqa: F401
    import simple_goalkeeper.tasks  # noqa: F401

    import mjlab

    amp_tasks = [t for t in list_tasks() if isinstance(load_rl_cfg(t), (AMPRunnerCfg, dict))]
    if not amp_tasks:
        raise RuntimeError("No AMP tasks registered.")

    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(amp_tasks),
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )

    args = tyro.cli(
        PlayConfig,
        args=remaining_args,
        default=PlayConfig(),
        prog=sys.argv[0] + f" {chosen_task}",
        config=mjlab.TYRO_FLAGS,
    )

    run_play(chosen_task, args)


if __name__ == "__main__":
    main()
