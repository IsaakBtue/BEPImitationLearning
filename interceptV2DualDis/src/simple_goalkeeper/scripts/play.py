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

    # Same, but rendered as AMP's discriminator actually perceives it
    # (re-anchored to whichever foot is planted each frame, root orientation
    # cancelled out -- reveals whether an edit to a reference clip is real
    # training signal or purely a root-orientation visual):
    uv run sgk_play Mjlab-BeyondAMP-Goalkeeper-T1-WithOverlay \\
        --agent zero --no-terminations True --amp-eye-view True \\
        --motion-file src/simple_goalkeeper/motions/data/<clip>.npz

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
    amp_eye_view: bool = False
    """When set with --motion-file: re-anchor the ghost, every frame, to
    whichever foot is currently planted (canonical: flat, fixed position)
    before rendering, instead of the file's own root frame. This cancels out
    root orientation entirely (a rigid re-expression relative to a body
    already known to be flat/grounded), so what's left is driven purely by
    relative joint angles -- exactly what AMP's discriminator perceives
    (joint_pos/joint_vel only; it never observes root orientation). Without
    this flag, the ghost faithfully replays the file's real root+joint data,
    which is correct for judging what a clip LOOKS like but is easy to
    mistake for what AMP is actually trained on whenever the clip's root
    orientation itself carries information (see docs/BugFixes.md,
    2026-08-01, and the Visualization Honesty Rule in CLAUDE.md). Expect a
    visible jump in the render each time stance switches feet -- that's
    expected, not a bug. Never use the derived file this produces as
    training data -- see scripts/amp_eye_view.py's module docstring."""
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
    # REMOVED 2026-08-07 (user request: "ignore the whole head thing, even
    # remove the whole head touch penalty for now"): force_head_contact_test
    # existed solely to verify the "HD" console indicator / head_ball_contact
    # P-panel plot, both of which were removed the same day alongside the
    # reward's head/chin sub-condition itself (rewards.py:
    # penalize_wrong_foot_ball_contact). See docs/BugFixes.md.


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
        self._knee_flash = 0.0
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
        # NEW 2026-08-08 (user request, final-review follow-up): same
        # per-episode accumulator pattern for the orange (trailing-foot)
        # mechanism -- shares blue's own _ep_was_wide (orange is only ever
        # active when env._blue_wide is true, see rewards.py's
        # _get_orange_reach_target_y), so only its own min-dist/max-settle/
        # landed fields are separate.
        self._min_orange_dist = float("inf")
        self._max_orange_settle = 0
        self._ep_orange_landed = False

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
                # NEW 2026-08-08: orange (trailing-foot) summary, same shape
                # as blue's -- shares self._ep_was_wide since orange is only
                # ever active on the same wide crossings blue is.
                orange_summary = (
                    f" | ORANGE min_dist={self._min_orange_dist:.2f} "
                    f"max_settle={self._max_orange_settle}/3 landed={self._ep_orange_landed}"
                    if self._ep_was_wide else " | ORANGE narrow-crossing"
                )
                print(
                    f"\n[EpEnd] terminated_by={','.join(fired) or 'none'} | "
                    f"min_base={self._min_base_h:.2f} "
                    f"min_Lsh={self._min_lsh_h:.2f} min_Rsh={self._min_rsh_h:.2f}"
                    f"{blue_summary}{orange_summary}",
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
            self._min_orange_dist = float("inf")
            self._max_orange_settle = 0
            self._ep_orange_landed = False
        self._prev_ep_buf = ep_buf.clone()

        # DEBUG 2026-07-28: called unconditionally (before the enabled-gate
        # below, and independent of the viewer's _show_plots/_is_paused
        # state) so the wrong-foot contact latch keeps advancing every
        # step regardless of whether analytics printing or the P-panel
        # plots happen to be toggled on -- the viewer-only version of this
        # call (_patch_viewer_wrong_foot_contact_plot) only runs while
        # _show_plots is True, which made it an unreliable ground-truth
        # source for confirming whether contact is actually being detected.
        self._wf_flash, self._knee_flash = _compute_wrong_foot_contact_flash(env, 0)

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
            # NEW 2026-08-08: orange (trailing-foot) accumulator, same
            # wide-gate as blue's above (rewards.py's env._orange_dbg_* is
            # only ever set inside the same env._blue_wide-gated branch).
            orange_dist_t = getattr(env, "_orange_dbg_dist", None)
            orange_settle_t = getattr(env, "_orange_dbg_settle", None)
            if orange_dist_t is not None:
                self._min_orange_dist = min(self._min_orange_dist, orange_dist_t[0].item())
            if orange_settle_t is not None:
                self._max_orange_settle = max(self._max_orange_settle, int(orange_settle_t[0].item()))
        landed_t = getattr(env, "_blue_landed", None)
        if landed_t is not None and bool(landed_t[0].item()):
            self._ep_landed = True
        orange_landed_t = getattr(env, "_orange_landed", None)
        if orange_landed_t is not None and bool(orange_landed_t[0].item()):
            self._ep_orange_landed = True

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
            f"{'KN✓' if self._knee_flash else 'KN·'}"
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
    _get_reach_target_y). When |crossing_y - start_y| > wide_threshold (0.5
    as of the 2026-08-01 revert -- kept symbolic here rather than
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

        # NEW 2026-08-08: orange sphere -- trailing-foot ("orange") mirror of
        # the blue midpoint above, recomputed inline the same way blue's own
        # mid_y is (not read from a cached env attribute) so this marker can't
        # drift out of sync with what orange_foot_proximity/orange_ball_landed/
        # orange_overshoot_penalty/orange_stick_landing (rewards.py) actually
        # target. See rewards.py:_get_orange_reach_target_y and
        # docs/superpowers/specs/2026-08-08-orange-ball-trailing-foot-design.md.
        # No color-switch-on-landed (unlike blue->green): the landing-focused
        # subset has no live-ball-tracking phase for the trailing foot to
        # graduate into, so a single static orange sphere is shown for the
        # whole wide-crossing window regardless of env._orange_landed.
        if wide:
            start_y = float(origins[1])
            delta = cross_y - start_y
            sign = 1.0 if delta >= 0 else -1.0
            shrunk = sign * max(abs(delta) - 0.50, 0.0)  # FIX 2026-08-08 (user request): 0.30 -> 0.60 -> 0.50
            orange_y = start_y + shrunk / 2.0
            _add_sphere(goal_x, orange_y, sphere_z, 0.08, [1.0, 0.55, 0.0, 0.75])
            _add_line(
                np.array([goal_x, orange_y, floor_z], dtype=np.float64),
                np.array([goal_x, orange_y, sphere_z], dtype=np.float64),
                0.008, [1.0, 0.55, 0.0, 0.6],
            )

    native_viewer._update_debug_visualizers = _patched_update


def _patch_viewer_new_reward_plots(native_viewer: "NativeMujocoViewer", env) -> None:
    """Promote the 2026-07-28 cleanstop/arm-penalty reward-term plots into the
    always-visible front slots, replacing the 3 feet-slippage plots that used
    to occupy them (2 custom raw slip-speed plots + the promoted
    `feet_slippage` reward term -- removed by this same change; see
    docs/BugFixes.md).

    No new custom raw plots needed here (unlike the removed feet-slip patch):
    `cleanstop`, `arm_torque_limits`, `arm_action_rate_l2`, `arm_action_acc_l2`
    are already ordinary reward terms with their own auto-created figures --
    they just need to be moved to the front of `_term_names`, same mechanism
    `postupperdofpos`'s promotion in `_patch_viewer_postupperdofpos_plot` uses.
    Without this, they're silently never rendered: this task registers ~48
    active reward terms against `max_viewports` (12), so anything past index
    12 in declaration order never appears no matter what `_show_plots` says.
    """
    orig_setup = native_viewer.setup

    _PROMOTED_REWARD_TERMS = (
        "cleanstop", "arm_torque_limits", "arm_action_rate_l2", "arm_action_acc_l2",
    )

    def _patched_setup() -> None:
        orig_setup()
        rest = [n for n in native_viewer._term_names if n not in _PROMOTED_REWARD_TERMS]
        promoted = [n for n in _PROMOTED_REWARD_TERMS if n in native_viewer._term_names]
        native_viewer._term_names = promoted + rest

    native_viewer.setup = _patched_setup


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
    high for _WRONG_FOOT_FLASH_STEPS steps after a detected touch: (1) any
    ball contact penalize_wrong_foot_ball_contact treats as bad (wrong-side
    foot sole OR either side's knee/shin OR leading knee proximity), (2) NEW
    2026-08-07 (user request) -- JUST the leading-knee proximity
    sub-condition of (1), isolated on its own so it can be verified
    independently of the wrong-side sole/leg conditions it's normally OR'd
    with in (1). This is the exact "knee contact with a distance calculator"
    mechanism the user asked to re-find in git history -- confirmed via
    `git log`: rewards.py:penalize_wrong_foot_ball_contact's leading_knee_touch,
    introduced in commit c55786c (2026-07-30), the SAME commit that tried
    and reverted a chin/head penalty. No separate "knee distance" reward
    function exists or ever existed -- it has always been this one
    sub-condition inside penalize_wrong_foot_ball_contact.

    (1) mirrors penalize_wrong_foot_ball_contact's own detection exactly (same
    "ball_contact"/"leg_ball_contact" sensors + _get_correct_foot_idx) -- does
    NOT read the reward term itself, since promoting that raw reward value
    (tried first) plots a signal too sparse for the viewer's autoscale to
    ever show (see docs/BugFixes.md).

    (1) EXTENDED 2026-07-28: user spotted an orange MuJoCo contact-point dot
    between the trailing leg and the ball while (1) still read 0 -- confirmed
    via a real-checkpoint replay that the shin/knee (not covered by
    ball_contact's foot[1-4]_collision pattern) genuinely touches the ball,
    sometimes on the wrong side. Added "leg_ball_contact" (goalkeeper_env_cfg.py).

    (1) REVISED 2026-07-30 (user request, after a live-checkpoint probe found
    the "knee/shin" label was misleading -- nearly every firing was genuinely
    the SHIN, never the knee sphere itself, see docs/BugFixes.md): now mirrors
    penalize_wrong_foot_ball_contact's final asymmetric shape exactly --
    WRONG side: whole leg (sole OR shin OR knee, contact-sensor based).
    CORRECT/leading side: KNEE PROXIMITY (distance-based, not contact-sensor
    "found" -- see rewards.py's docstring for why: the contact flag needs
    real geometric overlap and under-fires relative to visible near-misses).

    leg_ball_contact geom-index order: 0=left_shin, 1=left_knee, 2=right_knee,
    3=right_shin (same as penalize_wrong_foot_ball_contact, rewards.py).

    FIX 2026-08-07 (user request: "i only want the treshold of the chin so
    revert back again, and ignore the whole head thing, even remove the
    whole head touch penalty for now"): the head/chin sub-condition (a
    distance check against the H2 body's head_collision sphere, wired into
    the return value earlier this same session) is removed entirely,
    matching the same-day removal from rewards.py:penalize_wrong_foot_ball_
    contact -- this function must keep mirroring that reward exactly. The
    "head_ball_contact" P-panel plot / "HD" console flag / "HDdist" readout
    that read this function's now-removed 3rd return value are removed the
    same way (see _patch_viewer_wrong_foot_contact_plot and AnalyticsPolicy).

    FIX 2026-08-07 (user request: "implement it in the wrong foot ball
    contact" -- the new left_shin/right_shin sites, t1_headless.xml/t1.xml):
    folded in the same leading-shin proximity sub-condition added to
    rewards.py:penalize_wrong_foot_ball_contact the same day -- this
    function must keep mirroring that reward exactly (see the FIX above).
    Not split into its own isolated plot/flash counter (unlike the knee)
    since it wasn't requested -- only rolled into the combined
    wrong_touch/"wrong_foot_ball_contact" signal.

    Both remaining signals are viewer-only display latches with no effect
    on training/reward.
    """
    raw_env = env.unwrapped if hasattr(env, "unwrapped") else env
    from simple_goalkeeper.mdp.rewards import _get_correct_foot_idx

    _KNEE_GEOM_RADIUS = 0.06
    _SHIN_GEOM_RADIUS = 0.035  # FIX 2026-08-07: must match rewards.py's matching constant
    _BALL_GEOM_RADIUS = 0.10
    # FIX 2026-08-07: was 0.05, never updated through rewards.py's/
    # goalkeeper_env_cfg.py's earlier same-day increases -- silently out of
    # sync with the actual registered penalty for a while, likely why the
    # plot never visibly spiked before that was caught. Went 0.05->0.10->
    # 0.15->1.0 (DIAGNOSTIC-ONLY, confirmed the plot pipeline works) -> 0.15
    # (the real tuned value) -> back to 0.05 (user request: this is the
    # ORIGINAL default that was actually in effect, unversioned, for the
    # entire model_10250.pt training run -- watching that checkpoint under
    # its own true training-time threshold). Must match the value in
    # goalkeeper_env_cfg.py's penalize_wrong_foot_ball_contact params, not
    # rewards.py's function default (they can differ; params always wins).
    _KNEE_PROXIMITY_MARGIN = 0.05

    foot_idx = _get_correct_foot_idx(raw_env, "ball")  # (N,) 0=left, 1=right
    wrong_foot_idx = 1 - foot_idx
    env_ar = torch.arange(raw_env.num_envs, device=raw_env.device)

    found = raw_env.scene["ball_contact"].data.found  # [B, 8]
    left_sole = (found[:, :4] > 0).any(dim=-1)
    right_sole = (found[:, 4:] > 0).any(dim=-1)
    sole_touch = torch.stack([left_sole, right_sole], dim=-1)  # (B, 2)
    wrong_sole_touch = sole_touch[env_ar, wrong_foot_idx]

    leg_found = raw_env.scene["leg_ball_contact"].data.found  # [B, 4]
    left_leg_touch = (leg_found[:, :2] > 0).any(dim=-1)
    right_leg_touch = (leg_found[:, 2:] > 0).any(dim=-1)
    leg_touch = torch.stack([left_leg_touch, right_leg_touch], dim=-1)  # (B, 2)
    wrong_leg_touch = leg_touch[env_ar, wrong_foot_idx]

    ball_pos_w = raw_env.scene["ball"].data.root_link_pos_w  # (B, 3)
    robot = raw_env.scene["robot"]
    shank_ids = robot.find_bodies(["Shank_Left", "Shank_Right"])[0]
    knee_pos_w = robot.data.body_link_pos_w[:, shank_ids, :]  # (B, 2, 3)
    dist_to_knee = (ball_pos_w.unsqueeze(1) - knee_pos_w).norm(dim=-1)  # (B, 2)
    knee_threshold = _KNEE_GEOM_RADIUS + _BALL_GEOM_RADIUS + _KNEE_PROXIMITY_MARGIN
    knee_near = dist_to_knee < knee_threshold  # (B, 2)
    leading_knee_touch = knee_near[env_ar, foot_idx]

    shin_site_ids = robot.find_sites(["left_shin", "right_shin"])[0]
    shin_pos_w = robot.data.site_pos_w[:, shin_site_ids, :]  # (B, 2, 3)
    dist_to_shin = (ball_pos_w.unsqueeze(1) - shin_pos_w).norm(dim=-1)  # (B, 2)
    shin_threshold = _SHIN_GEOM_RADIUS + _BALL_GEOM_RADIUS + _KNEE_PROXIMITY_MARGIN
    shin_near = dist_to_shin < shin_threshold  # (B, 2)
    leading_shin_touch = shin_near[env_ar, foot_idx]

    wrong_touch = wrong_sole_touch | wrong_leg_touch | leading_knee_touch | leading_shin_touch

    if not hasattr(raw_env, "_wrong_foot_flash_counter"):
        raw_env._wrong_foot_flash_counter = torch.zeros(
            raw_env.num_envs, dtype=torch.long, device=raw_env.device
        )
    # NEW 2026-08-07 (user request): isolated latch for JUST the leading-knee
    # distance check (leading_knee_touch) -- the combined wrong_touch/counter
    # above ORs it together with wrong-side sole/leg conditions, so it alone
    # can't tell the user whether the SPECIFIC knee-distance mechanism
    # (rewards.py:penalize_wrong_foot_ball_contact's distance-based leading-
    # knee sub-check, git c55786c 2026-07-30) is the one firing. Same git
    # history the user asked to re-confirm: chin was tried and reverted in
    # that same commit; only the knee-distance check remains live in the
    # actual reward.
    if not hasattr(raw_env, "_knee_distance_flash_counter"):
        raw_env._knee_distance_flash_counter = torch.zeros(
            raw_env.num_envs, dtype=torch.long, device=raw_env.device
        )
    just_reset = raw_env.episode_length_buf <= 1

    counter = raw_env._wrong_foot_flash_counter
    counter[just_reset] = 0
    counter[wrong_touch] = _WRONG_FOOT_FLASH_STEPS
    counter[~wrong_touch & ~just_reset] = torch.clamp(counter[~wrong_touch & ~just_reset] - 1, min=0)

    knee_counter = raw_env._knee_distance_flash_counter
    knee_counter[just_reset] = 0
    knee_counter[leading_knee_touch] = _WRONG_FOOT_FLASH_STEPS
    knee_counter[~leading_knee_touch & ~just_reset] = torch.clamp(
        knee_counter[~leading_knee_touch & ~just_reset] - 1, min=0
    )

    return (
        float(counter[env_idx].item() > 0),
        float(knee_counter[env_idx].item() > 0),
    )


def _patch_viewer_wrong_foot_contact_plot(native_viewer: "NativeMujocoViewer", env) -> None:
    """Add two P-panel plots ("wrong_foot_ball_contact"/"knee_distance_contact")
    showing the latched raw bad-contact signals (see
    _compute_wrong_foot_contact_flash), alongside the existing per-reward-term
    plots. Mirrors the removed feet-slip patch's structure exactly (see
    _patch_viewer_new_reward_plots and docs/BugFixes.md for its replacement).

    NEW 2026-08-07 (user request): added "knee_distance_contact", isolating
    JUST the leading-knee proximity check (rewards.py:
    penalize_wrong_foot_ball_contact's distance-based sub-condition, git
    c55786c 2026-07-30 -- same commit chin/head was tried and reverted in)
    from "wrong_foot_ball_contact"'s combined OR of wrong-side-sole/leg/
    leading-knee -- the combined signal can't tell you WHICH condition
    fired. See docs/BugFixes.md.

    FIX 2026-08-07 (user request, same day: "remove the whole head touch
    penalty for now"): dropped the "head_ball_contact" plot -- the reward's
    head/chin sub-condition it tracked was removed the same way from
    rewards.py, so this plot no longer corresponds to anything the policy is
    penalized for.
    """
    orig_setup = native_viewer.setup
    orig_update_reward_figures = native_viewer._update_reward_figures

    _TERM_NAMES = ("wrong_foot_ball_contact", "knee_distance_contact")

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
            wrong_foot_flash, knee_flash = _compute_wrong_foot_contact_flash(env, native_viewer.env_idx)
            native_viewer._append_point("wrong_foot_ball_contact", wrong_foot_flash)
            native_viewer._write_history_to_figure("wrong_foot_ball_contact")
            native_viewer._append_point("knee_distance_contact", knee_flash)
            native_viewer._write_history_to_figure("knee_distance_contact")
        orig_update_reward_figures(viewer_handle)

    native_viewer.setup = _patched_setup
    native_viewer._update_reward_figures = _patched_update_reward_figures


_FOOT_TARGET_ANGLE_DEG = 75.0
"""Must match rewards.py's _FOOT_TARGET_ANGLE_DEG (leading-foot block-posture
target, 2026-08-01, retargeted 60->45 2026-08-07, reverted 45->60 2026-08-08
per user request -- back to the original 2026-08-01 value; retargeted again
60->75 2026-08-10 per user request, "15 degrees from the y axis", after a
model_11250.pt replay found the foot almost not rotating toward the target
at all). Kept as a separate literal here (viewer-only display, no runtime
dependency on rewards.py's private constant) purely for the plot title --
verify against rewards.py if that constant ever changes."""

_FOOT_TARGET_TOLERANCE_DEG = 31.79
"""degrees(acos(0.85)) -- inner_face_orientation_save's alignment_threshold
(0.7->0.85, 2026-08-07, tightened alongside the target-angle change above).
FIX 2026-08-08: the target angle reverted 45->60 the same day but
alignment_threshold was deliberately LEFT at 0.85 per explicit user request
(not reverted to the 0.7 that was originally calibrated for 60deg) -- this
tolerance value is therefore no longer the "matching" calibration for the
current target the way it was on 2026-08-07, a known, requested divergence.
Not itself read by any reward function."""


def _compute_assigned_foot_angle_deg(env, env_idx: int) -> float:
    """Assigned/leading foot's long-axis angle off robot-forward, in degrees,
    for the single env at env_idx.

    NEW 2026-08-01 (user request): visual verification for the
    inner_face_orientation_save/foot_inner_face_continuous retarget (was 90
    deg/parallel-to-Y, now _FOOT_TARGET_ANGLE_DEG deg off forward) -- lets a
    human watching sgk_play confirm the foot is actually settling near the
    new target instead of trusting the reward-curve shape alone. Reuses the
    exact foot_inner_face_continuous geometry (rewards.py) so the plotted
    number matches what the reward itself sees, not a fresh reimplementation.
    Unsigned (0-180 deg): the assigned foot is always evaluated against its
    own side's target, so magnitude alone is meaningful here.
    """
    raw_env = env.unwrapped if hasattr(env, "unwrapped") else env
    from mjlab.utils.lab_api.math import quat_apply
    from simple_goalkeeper.mdp.rewards import _get_correct_foot_idx

    robot = raw_env.scene["robot"]
    foot_ids = robot.find_bodies(["left_foot_link", "right_foot_link"])[0]
    foot_quat_w = robot.data.body_link_quat_w[:, foot_ids, :]  # (N, 2, 4)

    foot_idx = _get_correct_foot_idx(raw_env, "ball")  # (N,) 0=left, 1=right

    foot_long_local = torch.zeros(raw_env.num_envs, 3, device=raw_env.device)
    foot_long_local[:, 0] = 1.0
    left_long_w  = quat_apply(foot_quat_w[:, 0, :], foot_long_local)
    right_long_w = quat_apply(foot_quat_w[:, 1, :], foot_long_local)
    foot_long_w  = torch.where((foot_idx == 0)[:, None], left_long_w, right_long_w)  # (N, 3)

    forward_local = torch.zeros(raw_env.num_envs, 3, device=raw_env.device)
    forward_local[:, 0] = 1.0
    forward_w = quat_apply(robot.data.root_link_quat_w, forward_local)  # (N, 3)

    cos_angle = (foot_long_w * forward_w).sum(dim=-1).clamp(-1.0, 1.0)  # (N,)
    angle_deg = torch.rad2deg(torch.acos(cos_angle))  # (N,) in [0, 180]

    return float(angle_deg[env_idx].item())


def _compute_blue_overshoot_margin(env, env_idx: int, landing_radius: float = 0.08) -> float:
    """Raw, UNCLAMPED margin behind `blue_overshoot_penalty`'s own hinge
    (rewards.py) for the single env at env_idx: `signed_progress -
    landing_radius`. Negative = short of the blue midpoint (by that many
    meters, past the landing-radius deadband); 0 = right at the hinge; the
    reward itself only reads the clamped `max(margin, 0)` part.

    NEW 2026-08-12 (user request): `blue_overshoot_penalty`'s own plot
    reads flat 0 the whole time other terms are visibly moving. That's
    ambiguous by construction -- unlike a continuous term (footreach,
    blue_stick_landing), this reward is `clamp(x, min=0)`: it is
    EXACTLY, identically 0.0 for the entire region short of the hinge, not
    just small -- so "flat 0" is indistinguishable on the reward plot alone
    between "genuinely never approaches blue" (expected if the policy is
    risk-averse) and "the position computation itself is stuck/broken."
    This plot exposes the pre-clamp value so real (if small) approach
    progress is visible as movement even while the reward stays at 0.

    Must be kept in sync with `blue_overshoot_penalty`'s own geometry
    (rewards.py) -- same desync risk as any other duplicated-math diagnostic
    in this file (see `_compute_assigned_foot_angle_deg`'s docstring for the
    established pattern this follows).
    """
    raw_env = env.unwrapped if hasattr(env, "unwrapped") else env
    from simple_goalkeeper.mdp.rewards import _get_ball_crossing_y, _get_correct_foot_idx

    full_y = _get_ball_crossing_y(raw_env, "ball")                 # (N,) world Y
    start_y = raw_env.scene.env_origins[:, 1]                      # (N,) world Y
    half_y = start_y + (full_y - start_y) / 2.0
    direction = torch.sign(full_y - start_y)

    robot = raw_env.scene["robot"]
    foot_ids = robot.find_bodies(["left_foot_link", "right_foot_link"])[0]
    foot_pos_w = robot.data.body_link_pos_w[:, foot_ids, :]        # (N, 2, 3)
    foot_idx = _get_correct_foot_idx(raw_env, "ball")              # (N,)
    assigned_foot_y = foot_pos_w[env_idx, int(foot_idx[env_idx].item()), 1]

    signed_progress = direction[env_idx] * (assigned_foot_y - half_y[env_idx])
    margin = signed_progress - landing_radius
    return float(margin.item())


def _patch_viewer_blue_overshoot_margin_plot(native_viewer: "NativeMujocoViewer", env) -> None:
    """Add a "blue_overshoot_margin_m" P-panel plot (see
    _compute_blue_overshoot_margin), promoted to an always-visible front
    slot. Same structural pattern as _patch_viewer_foot_orientation_plot.
    NEW 2026-08-12 (user request) -- diagnostic for the flat-0
    blue_overshoot_penalty plot investigation.
    """
    orig_setup = native_viewer.setup
    orig_update_reward_figures = native_viewer._update_reward_figures

    _NAME = "blue_overshoot_margin_m"

    def _patched_setup() -> None:
        orig_setup()
        from mjlab.viewer.native.viewer import make_empty_figure
        cfg = native_viewer._plot_cfg
        native_viewer._figures[_NAME] = make_empty_figure(
            f"{_NAME} (0=hinge, neg=short of blue)",
            cfg.grid_size, cfg.init_yrange, cfg.history, cfg.background_alpha,
        )
        native_viewer._histories[_NAME] = deque(maxlen=cfg.history)
        native_viewer._yrange[_NAME] = cfg.init_yrange
        native_viewer._scale[_NAME] = 1.0
        rest = [n for n in native_viewer._term_names]
        native_viewer._term_names = [_NAME] + rest

    def _patched_update_reward_figures(viewer_handle: "mujoco.viewer.Handle") -> None:
        if native_viewer._show_plots and native_viewer._term_names and not native_viewer._is_paused:
            margin = _compute_blue_overshoot_margin(env, native_viewer.env_idx)
            native_viewer._append_point(_NAME, margin)
            native_viewer._write_history_to_figure(_NAME)
        orig_update_reward_figures(viewer_handle)

    native_viewer.setup = _patched_setup
    native_viewer._update_reward_figures = _patched_update_reward_figures


def _patch_viewer_blue_landed_flags_plot(native_viewer: "NativeMujocoViewer", env) -> None:
    """Add "blue_landed" and "blue_landed_was_free" P-panel plots (0.0/1.0
    step functions), both always-visible front slots.

    NEW 2026-08-12 (user request) -- direct evidence for whether the
    blue-overshoot-still-negative-after-the-viz-switched-green mystery is
    actually the `_blue_landed_was_free` desync (viz reads the loose
    `_blue_landed`, every reward gate reads the strict `_blue_landed_genuine
    = _blue_landed & ~_blue_landed_was_free`) or something else. NOTE: play
    mode disables RSI by default (`goalkeeper_env_cfg.py`, "RSI disabled in
    play mode... Always starting from standing" -- popped unless `--no-rsi
    False`), so `_blue_landed_was_free`'s `episode_length_buf < 10` early-
    landing trigger, which was calibrated against RSI's mid-motion donor
    starts, may rarely or never fire from a standing-pose reset -- these two
    plots will show directly whether it's firing at all in this session,
    rather than assuming.
    """
    orig_setup = native_viewer.setup
    orig_update_reward_figures = native_viewer._update_reward_figures

    _NAMES = ("blue_landed", "blue_landed_was_free")

    def _patched_setup() -> None:
        orig_setup()
        from mjlab.viewer.native.viewer import make_empty_figure
        cfg = native_viewer._plot_cfg
        for name in _NAMES:
            native_viewer._figures[name] = make_empty_figure(
                name, cfg.grid_size, cfg.init_yrange, cfg.history, cfg.background_alpha,
            )
            native_viewer._histories[name] = deque(maxlen=cfg.history)
            native_viewer._yrange[name] = cfg.init_yrange
            native_viewer._scale[name] = 1.0
        rest = [n for n in native_viewer._term_names]
        native_viewer._term_names = list(_NAMES) + rest

    def _patched_update_reward_figures(viewer_handle: "mujoco.viewer.Handle") -> None:
        if native_viewer._show_plots and native_viewer._term_names and not native_viewer._is_paused:
            raw_env = env.unwrapped if hasattr(env, "unwrapped") else env
            idx = native_viewer.env_idx
            landed_t = getattr(raw_env, "_blue_landed", None)
            was_free_t = getattr(raw_env, "_blue_landed_was_free", None)
            landed_v = float(landed_t[idx].item()) if landed_t is not None else 0.0
            was_free_v = float(was_free_t[idx].item()) if was_free_t is not None else 0.0
            native_viewer._append_point("blue_landed", landed_v)
            native_viewer._write_history_to_figure("blue_landed")
            native_viewer._append_point("blue_landed_was_free", was_free_v)
            native_viewer._write_history_to_figure("blue_landed_was_free")
        orig_update_reward_figures(viewer_handle)

    native_viewer.setup = _patched_setup
    native_viewer._update_reward_figures = _patched_update_reward_figures


def _patch_viewer_foot_orientation_plot(native_viewer: "NativeMujocoViewer", env) -> None:
    """Add an "assigned_foot_angle_deg" P-panel plot (see
    _compute_assigned_foot_angle_deg), promoted to an always-visible front
    slot. NEW 2026-08-01 (user request), same structural pattern as
    _patch_viewer_wrong_foot_contact_plot (raw custom plot, not an ordinary
    reward-term figure).
    """
    orig_setup = native_viewer.setup
    orig_update_reward_figures = native_viewer._update_reward_figures

    _NAME = "assigned_foot_angle_deg"

    def _patched_setup() -> None:
        orig_setup()
        from mjlab.viewer.native.viewer import make_empty_figure
        cfg = native_viewer._plot_cfg
        native_viewer._figures[_NAME] = make_empty_figure(
            f"{_NAME} (target={_FOOT_TARGET_ANGLE_DEG:.0f}+/-{_FOOT_TARGET_TOLERANCE_DEG:.0f})",
            cfg.grid_size, cfg.init_yrange, cfg.history, cfg.background_alpha,
        )
        native_viewer._histories[_NAME] = deque(maxlen=cfg.history)
        native_viewer._yrange[_NAME] = cfg.init_yrange
        native_viewer._scale[_NAME] = 1.0
        rest = [n for n in native_viewer._term_names]
        native_viewer._term_names = [_NAME] + rest

    def _patched_update_reward_figures(viewer_handle: "mujoco.viewer.Handle") -> None:
        if native_viewer._show_plots and native_viewer._term_names and not native_viewer._is_paused:
            angle_deg = _compute_assigned_foot_angle_deg(env, native_viewer.env_idx)
            native_viewer._append_point(_NAME, angle_deg)
            native_viewer._write_history_to_figure(_NAME)
        orig_update_reward_figures(viewer_handle)

    native_viewer.setup = _patched_setup
    native_viewer._update_reward_figures = _patched_update_reward_figures


def _patch_viewer_postupperdofpos_plot(native_viewer: "NativeMujocoViewer", env) -> None:
    """Promote `postupperdofpos` (arm post-save recovery reward) into the
    always-visible front P-panel slots.

    FIX 2026-07-30 (user request): replaces this function's prior content
    (`left_foot_yaw_deg`/`right_foot_yaw_deg` custom plots + `postlegdofpos`
    promotion) -- removed outright, not just deprioritized, per user request.
    `postupperdofpos` is an ordinary reward term with its own auto-created
    figure, so this only needs the promotion mechanism (no custom raw plot),
    same pattern as `_patch_viewer_new_reward_plots`.

    FIX 2026-07-30 (2nd follow-up, user request): also promoted
    `penalize_arm_above_shoulder` (the new shoulder-height penalty, same
    date, see rewards.py) and `trailing_foot_forward_continuous` (existing
    term, previously never promoted so silently unrendered past the top-12
    viewport cap) alongside it -- all three are arm/posture-adjacent terms
    useful to watch together while diagnosing the "arms flying" behavior.
    """
    orig_setup = native_viewer.setup

    _PROMOTED_REWARD_TERMS = (
        "postupperdofpos", "penalize_arm_above_shoulder", "trailing_foot_forward_continuous",
    )

    def _patched_setup() -> None:
        orig_setup()
        rest = [n for n in native_viewer._term_names if n not in _PROMOTED_REWARD_TERMS]
        promoted = [n for n in _PROMOTED_REWARD_TERMS if n in native_viewer._term_names]
        native_viewer._term_names = promoted + rest

    native_viewer.setup = _patched_setup


def _patch_viewer_postsave_airtime_plot(native_viewer: "NativeMujocoViewer", env) -> None:
    """Swap the `stopball` P-panel plot for `postsave_foot_airtime` (2026-08-01
    airborne-hangtime bonus, rewards.py).

    NEW 2026-08-03 (user request): user wants to watch postsave_foot_airtime
    live while replaying a checkpoint in the native viewer instead of
    stopball -- stopball reliably fires already and doesn't need dedicated
    screen space. Same promotion mechanism as
    _patch_viewer_postupperdofpos_plot, but also explicitly demotes
    `stopball` to the back of `_term_names` (this task registers far more
    reward terms than `max_viewports` (12), so anything not in the front 12
    is silently unrendered) -- without this, promoting postsave_foot_airtime
    alone wouldn't guarantee stopball drops off, only that something at the
    tail of the list does.
    """
    orig_setup = native_viewer.setup

    _PROMOTED = ("postsave_foot_airtime",)
    _DEMOTED = ("stopball",)

    def _patched_setup() -> None:
        orig_setup()
        rest = [
            n for n in native_viewer._term_names
            if n not in _PROMOTED and n not in _DEMOTED
        ]
        promoted = [n for n in _PROMOTED if n in native_viewer._term_names]
        demoted = [n for n in _DEMOTED if n in native_viewer._term_names]
        native_viewer._term_names = promoted + rest + demoted

    native_viewer.setup = _patched_setup


def _patch_viewer_post_recovery_plots(native_viewer: "NativeMujocoViewer", env) -> None:
    """Make the P-panel (KEY_P already toggles its visibility natively --
    `_safe_key_callback`'s `TOGGLE_PLOTS` action, mjlab's own binding, not a
    new one added here) show the full post-recovery reward family in the
    front 12 visible slots, superseding whatever the earlier promotion
    patches in this file left there.

    NEW 2026-08-03 (user request): "put all the rewards that are on at post
    recovery" when pressing P. Built directly from this file's own
    `_ball_is_behind`-gated reward functions (grepped `rewards.py` for every
    function containing a `_ball_is_behind(env` call, excluding the
    APPROACH-phase terms that instead gate OFF once behind --
    `footreach`/`foot_proximity`/`foot_inner_face_continuous`/
    `foot_clearance`/`blue_trunk_drive` -- those are the opposite family):
    `postangvel`, `postlinvel`, `postupperdofpos`, `postwaistdofpos`,
    `postlegdofpos`, `postleadfootorientation`, `postsave_foot_airtime`,
    `postheadingorientation`, `penalize_arm_above_shoulder` (steady-post-save
    gated). Plus three always-active terms from the same "recovery" family
    by naming/purpose, included since they're exactly what's under
    investigation for the post-save arm-drift bug: `postorientation`
    (upright recovery, always active per its own docstring -- AMP has no
    root-orientation signal to supply this itself), `angular_momentum_penalty`
    (2026-08-03, whole-body momentum). Exactly 12 terms -- fits `max_viewports`
    (12) with no overflow, so all twelve are guaranteed visible with no need
    to also demote anything else.

    FIX 2026-08-06 (user request): swapped `arm_dof_vel` (12th slot) for
    `inner_face_orientation_save` -- the one-time save-moment leading-foot
    "block posture" reward, retargeted 2026-08-01 (commit 036222f) from 90deg
    off forward (parallel to world Y) to 60deg off forward (30deg off Y).
    It's an APPROACH-phase term by this function's own earlier classification
    (fires at the softstop moment, not gated by `_ball_is_behind` like the
    other 11), included here anyway so the new save-angle target can be
    watched directly in the panel.
    """
    orig_setup = native_viewer.setup

    # FIX 2026-08-06 (user request): swapped `arm_dof_vel` out for
    # `inner_face_orientation_save` -- the save-moment leading-foot "block
    # posture" reward retargeted 2026-08-01 (commit 036222f) to 60deg off
    # forward / 30deg off world Y (`_FOOT_TARGET_ANGLE_DEG`, rewards.py).
    # It's an APPROACH-phase term (fires at the softstop moment, not
    # `~behind`-gated like the other 11), included here anyway so the foot's
    # save-angle can be watched directly; `arm_dof_vel` was judged the least
    # foot-orientation-relevant of the original 12.
    #
    # FIX 2026-08-06 (user request): dropped `postwaistdofpos` and
    # `angular_momentum_penalty` to make room for the 4 terms touched by
    # today's foot/leg orientation rework (airborne-latch fix affecting
    # `postlegdofpos`/`postleadfootorientation`, the simplified
    # `foot_inner_face_continuous`, and the save-time-captured
    # `postheadingorientation`) -- added `foot_inner_face_continuous`
    # (the only one of the 4 not already in this list). 11 terms now, not
    # backfilled to 12 -- the display cap, not a requirement to hit exactly.
    #
    # FIX 2026-08-06 (later same day, user request): swapped
    # `penalize_arm_above_shoulder` out for `penalize_sharpcontact` --
    # user just retuned that term's force threshold (1700->1800N) and
    # wants to watch it directly in the panel.
    #
    # FIX 2026-08-07 (user request): swapped `postlinvel` out for
    # `penalize_arm_above_shoulder` -- brought back into the panel (was
    # bumped out same-day 2026-08-06 for `penalize_sharpcontact`, which
    # stays). `postlinvel` was judged the least relevant of the 11 to the
    # current investigation (foot-yaw/post-save-pose instability).
    #
    # FIX 2026-08-07 (later same day, user request): added `postshoulderdofpos`
    # -- the shoulder-scoped sibling of `postupperdofpos` (added 2026-08-06,
    # never previously promoted to this panel). 12 terms now, exactly at the
    # display cap -- nothing else needed removing since the list was at 11.
    #
    # FIX 2026-08-07 (later same day, user request): swapped
    # `foot_inner_face_continuous` out for `head_ball_contact` -- this panel
    # is registered LAST in run_play (after _patch_viewer_wrong_foot_contact_
    # plot), so its own term_names reordering runs last and wins; since
    # "head_ball_contact" isn't in _POST_RECOVERY_REWARD_TERMS it was
    # silently pushed past the 12-slot max_viewports cutoff by this exact
    # panel, despite _patch_viewer_wrong_foot_contact_plot having genuinely
    # registered and computed it -- the plot existed and updated every step,
    # it just was never actually visible. This surfaces the raw latched
    # chin/head-contact signal (see _compute_wrong_foot_contact_flash) after
    # user review of model_10250.pt found the policy converged on chin/head
    # contact as a save method. `foot_inner_face_continuous`'s own target
    # angle is still separately visible via the "assigned_foot_angle_deg"
    # custom plot (_patch_viewer_foot_orientation_plot), and its one-shot
    # sibling `inner_face_orientation_save` stays in this panel -- judged
    # the least-redundant term to demote. Viewer-only change, no effect on
    # training/rewards.
    # FIX 2026-08-07 (user request): same visibility bug as the head_ball_contact
    # fix above -- "wrong_foot_ball_contact" and "knee_distance_contact" were
    # ALSO being silently pushed past the 12-slot cutoff by this same panel,
    # the whole time head_ball_contact was (they were never in this list
    # either, until now). Demoted `penalize_sharpcontact` (unrelated to the
    # current wrong-foot/knee/chin-contact investigation) and
    # `postsave_foot_airtime` (peripheral to it) to make room for both.
    #
    # FIX 2026-08-07 (later same day, user request: "ignore the whole head
    # thing, even remove the whole head touch penalty for now"): dropped
    # "head_ball_contact" -- the reward's head/chin sub-condition it tracked
    # was removed the same day from rewards.py:penalize_wrong_foot_ball_
    # contact. 11 terms now, not backfilled to 12 -- the display cap, not a
    # requirement to hit exactly (same reasoning as the 2026-08-06 entry
    # above).
    # FIX 2026-08-08 (user request): "turn on rewards i need to watch out for"
    # for the (re-opened, still-unresolved per a fresh model_17250.pt replay
    # this session) leading-foot yaw-spin investigation. Swapped out
    # "wrong_foot_ball_contact"/"knee_distance_contact" (a separate, orthogonal
    # investigation from an earlier session) for the three reward terms that
    # directly measure/damp the spin itself and were never in this panel
    # despite being exactly what the current investigation needs to watch:
    # "foot_ang_vel_xy"/"foot_ang_vel_z" (the two rate-damping penalties added
    # 2026-08-07 specifically for this issue) and "ang_vel_z" (whole-body yaw
    # rate). "postleadfootorientation"/"postheadingorientation"/
    # "inner_face_orientation_save" were already present and stay. See
    # docs/BugFixes.md for the fresh replay evidence this responds to.
    # FIX 2026-08-10 (user request, "enable all the leading foot orientation
    # rewards"): swapped "ang_vel_z" (whole-body yaw rate, least foot-specific
    # of the 12, and redundant with foot_ang_vel_z which already covers foot-
    # level yaw) for "foot_inner_face_continuous" -- the pre-save continuous
    # sibling of inner_face_orientation_save/postleadfootorientation, bumped
    # out of this panel on 2026-08-07 and never restored. This completes the
    # full leading-foot-orientation reward family in one panel:
    # foot_inner_face_continuous (pre-save), inner_face_orientation_save
    # (save-moment one-shot), postleadfootorientation (post-save), plus the
    # two rotation-rate penalties foot_ang_vel_xy/foot_ang_vel_z. Still 12
    # terms, still fits max_viewports with no overflow.
    _POST_RECOVERY_REWARD_TERMS = (
        "postorientation", "postangvel", "penalize_arm_above_shoulder",
        "postupperdofpos", "postshoulderdofpos", "postlegdofpos",
        "postleadfootorientation", "postheadingorientation",
        "inner_face_orientation_save", "foot_inner_face_continuous",
        "foot_ang_vel_xy", "foot_ang_vel_z",
    )

    def _patched_setup() -> None:
        orig_setup()
        rest = [n for n in native_viewer._term_names if n not in _POST_RECOVERY_REWARD_TERMS]
        promoted = [n for n in _POST_RECOVERY_REWARD_TERMS if n in native_viewer._term_names]
        native_viewer._term_names = promoted + rest

    native_viewer.setup = _patched_setup


def _patch_viewer_blue_overshoot_plot(native_viewer: "NativeMujocoViewer", env) -> None:
    """Promote `blue_overshoot_penalty` (rewards.py) into the always-visible
    front slot. NEW 2026-08-12 (user request) -- investigating a suspected
    false-positive source: this term has no RSI-freshness guard (unlike its
    sibling `_blue_landed_was_free`, which excludes suspiciously-early
    landings from paying out landing bonuses), so a mid-episode RSI reset
    that happens to place the assigned foot already past the blue midpoint
    can fire a negative reward on the very first step(s) of an episode,
    before the policy has taken any action. Watching this term live is the
    first step to confirming that.

    Same mechanism as `_patch_viewer_new_reward_plots` (ordinary reward-term
    figure, just reordered to the front) -- `blue_overshoot_penalty` already
    has its own auto-created figure, no custom raw plot needed. Registered
    AFTER `_patch_viewer_post_recovery_plots` in run_play so this promotion
    wins final ordering; this pushes the last (12th) post-recovery term
    (`foot_ang_vel_z`) out of the visible cap.
    """
    orig_setup = native_viewer.setup

    _NAME = "blue_overshoot_penalty"

    def _patched_setup() -> None:
        orig_setup()
        rest = [n for n in native_viewer._term_names if n != _NAME]
        native_viewer._term_names = [_NAME] + rest

    native_viewer.setup = _patched_setup


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

        if cfg.amp_eye_view:
            from simple_goalkeeper.scripts.amp_eye_view import make_amp_eye_view
            import tempfile
            eye_path = Path(tempfile.gettempdir()) / f"amp_eye_view_{motion_path.stem}.npz"
            make_amp_eye_view(motion_path, eye_path)
            print(
                f"[AMP-EYE-VIEW]: {motion_path.name} re-anchored to the currently-planted "
                f"foot each frame (root transform cancelled out) -> {eye_path.name}. This is "
                f"what the AMP discriminator actually observes (joint_pos/joint_vel only, no "
                f"root term) -- NOT a training-data file, diagnostic rendering only. Expect a "
                f"visible jump each time stance switches feet."
            )
            motion_path = eye_path
        else:
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
        _patch_viewer_new_reward_plots(native_viewer, env)
        _patch_viewer_postupperdofpos_plot(native_viewer, env)
        _patch_viewer_wrong_foot_contact_plot(native_viewer, env)
        _patch_viewer_foot_orientation_plot(native_viewer, env)
        _patch_viewer_postsave_airtime_plot(native_viewer, env)
        _patch_viewer_post_recovery_plots(native_viewer, env)
        _patch_viewer_blue_overshoot_plot(native_viewer, env)
        _patch_viewer_blue_overshoot_margin_plot(native_viewer, env)
        _patch_viewer_blue_landed_flags_plot(native_viewer, env)
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
