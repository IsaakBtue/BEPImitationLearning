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
"""
from __future__ import annotations

import os
import sys
import types
from sys import stderr
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

from beyondAMP.mjlab.rsl_rl import AMPEnvWrapper, AMPRunnerCfg
from rsl_rl_amp.runners.amp_on_policy_runner import AMPOnPolicyRunner


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
    rsi: bool = False
    """Enable Random State Initialization in play (default: off — starts from standing keyframe)."""
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
                print(file=stderr)  # newline after overwriting line
            self._ep += 1
        self._prev_ep_buf = ep_buf.clone()

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

        flags = (
            f"{'SB✓' if stopball_fired else 'SB·'} "
            f"{'SS✓' if softstop_fired else 'SS·'} "
            f"{'CS✓' if cleanstop_fired else 'CS·'}"
        )
        lf_tag = f"{'G' if lf_contact else 'A'}{lf_slip:.2f}"
        rf_tag = f"{'G' if rf_contact else 'A'}{rf_slip:.2f}"
        print(
            f"\rEp{self._ep:3d} | "
            f"bvx={bv[0].item():+6.2f} bvy={bv[1].item():+5.2f} spd={ball_speed:.2f} "
            f"bx={bx_local:+5.2f} | "
            f"int(x={int_x:+5.2f} y={int_y:+5.2f}) | "
            f"dvx={delta_vx:+5.2f} | "
            f"LF={lf_h:.3f}({lf_tag}) RF={rf_h:.3f}({rf_tag}) | "
            f"{flags}",
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

    Adds a green sphere at the frozen goal-line crossing point (env 0) and a
    vertical line from floor to sphere so it's visible from any camera angle.
    The sphere updates each render frame — it moves when a new episode starts
    and the crossing_y changes.
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

        # Sphere at intercept point.
        _add_sphere(goal_x, cross_y, sphere_z, 0.08, [0.1, 1.0, 0.2, 0.75])
        # Vertical line from floor to sphere.
        _add_line(
            np.array([goal_x, cross_y, floor_z], dtype=np.float64),
            np.array([goal_x, cross_y, sphere_z], dtype=np.float64),
            0.008, [0.1, 1.0, 0.2, 0.6],
        )

    native_viewer._update_debug_visualizers = _patched_update


def run_play(task_id: str, cfg: PlayConfig) -> None:
    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    env_cfg = load_env_cfg(task_id, play=True)
    agent_cfg = load_rl_cfg(task_id)
    assert isinstance(agent_cfg, AMPRunnerCfg), (
        f"Task '{task_id}' is not an AMP task — got {type(agent_cfg).__name__}."
    )

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

    # Print expected ball spawn ranges at the current difficulty (matches training).
    d = float(getattr(env, "_ball_difficulty", 1.0))
    _EASY_DIST = (2.0, 2.0); _HARD_DIST = (2.0, 3.5)
    _EASY_T    = (0.9, 1.3); _HARD_T    = (0.7, 1.1)
    _Y_INNER = 0.15; _Y_OUTER = 0.35; _Y_MAX = 1.0
    import math as _math
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

    amp_tasks = [t for t in list_tasks() if isinstance(load_rl_cfg(t), AMPRunnerCfg)]
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
