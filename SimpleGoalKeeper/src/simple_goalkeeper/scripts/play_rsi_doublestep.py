"""Play a trained checkpoint with a chosen RSI motion pool.

Motion pools map to these NPZ files:
  wide   (default) — LeftDoubleStep + LeftTripleStep + RightDoubleStep + RightTripleStep
  double           — LeftSafeMedium + RightSafeMedium  (medium-range shots)
  triple           — LeftSafeFar + RightSafeFar         (far-range shots)
  all              — every NPZ file combined (same as normal training RSI)

Usage:
    uv run sgk_play_rsi

    # Different pool or checkpoint:
    uv run sgk_play_rsi --motion-pool double
    uv run sgk_play_rsi --motion-pool all
    uv run sgk_play_rsi --checkpoint-file logs/rsl_rl/simple_goalkeeper/<run>/model_9000.pt

    # All normal play flags also work:
    uv run sgk_play_rsi --difficulty 0.5 --viewer viser --num-envs 1
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.utils.torch import configure_torch_backends

from beyondAMP.mjlab.rsl_rl import AMPEnvWrapper, AMPRunnerCfg
from rsl_rl_amp.runners.amp_on_policy_runner import AMPOnPolicyRunner

from simple_goalkeeper.scripts.play import AnalyticsPolicy, _patch_viewer_intercept_vis

_DEFAULT_CHECKPOINT = (
    "logs/rsl_rl/simple_goalkeeper/2026-06-29_20-36-28_phase1/model_13500.pt"
)
_TASK = "Mjlab-BeyondAMP-Goalkeeper-T1"

# Maps CLI pool name → internal (side, tier) key used by MotionResetManager.
# Side is filled in at runtime from ball crossing_y sign.
_POOL_TIERS: dict[str, str | None] = {
    "wide":   "wide",    # DoubleStep + TripleStep files
    "double": "double",  # SafeMedium files
    "triple": "triple",  # SafeFar files
    "all":    None,      # combined pool (all motions)
}


@dataclass(frozen=True)
class RsiPlayConfig:
    checkpoint_file: str = _DEFAULT_CHECKPOINT
    """Path to .pt checkpoint (absolute or relative to SimpleGoalKeeper/)."""

    motion_pool: Literal["wide", "double", "triple", "all"] = "wide"
    """RSI motion pool. 'wide' = DoubleStep+TripleStep, 'double' = SafeMedium,
    'triple' = SafeFar, 'all' = every motion combined."""

    rsi_fraction: float = 0.8
    """Fraction of resets that use RSI (remainder start from standing keyframe)."""

    num_envs: int | None = None
    device: str | None = None
    no_terminations: bool = False
    difficulty: float | None = None
    viewer: Literal["auto", "native", "viser"] = "auto"
    analytics: bool = True


def _make_rsi_event(pool_name: str, rsi_fraction: float):
    """Return an RSI reset function locked to the chosen pool tier."""
    tier = _POOL_TIERS[pool_name]

    def _rsi_event(env, env_ids):
        from simple_goalkeeper.mdp.events import MotionResetManager, _DEFAULT_ROBOT_CFG

        mgr = MotionResetManager.get()
        mgr.init(env)

        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int32)
        if len(env_ids) == 0:
            return

        robot = env.scene[_DEFAULT_ROBOT_CFG.name]
        n = len(env_ids)

        use_rsi = torch.rand(n, device=env.device) < rsi_fraction
        rsi_ids   = env_ids[use_rsi.nonzero(as_tuple=True)[0]]
        stand_ids = env_ids[(~use_rsi).nonzero(as_tuple=True)[0]]

        if len(rsi_ids) > 0:
            if tier is None:
                # "all" pool: no side split needed
                mgr._write_rsi_state(env, rsi_ids, mgr.frames, robot)
            else:
                # Pick side from predicted ball crossing_y; fall back to 50/50
                cross_y = getattr(env, "_rsi_cross_y", None)
                if cross_y is not None:
                    cy = cross_y[rsi_ids]
                    is_left = cy > 0
                else:
                    is_left = torch.rand(len(rsi_ids), device=env.device) > 0.5

                for side, mask in [("left", is_left), ("right", ~is_left)]:
                    ids = rsi_ids[mask.nonzero(as_tuple=True)[0]]
                    if len(ids) == 0:
                        continue
                    pool = mgr.pools.get((side, tier), mgr.frames)
                    mgr._write_rsi_state(env, ids, pool, robot)

        if len(stand_ids) > 0:
            home_pos = robot.data.default_joint_pos[stand_ids]
            home_vel = torch.zeros(len(stand_ids), home_pos.shape[-1], device=env.device)
            robot.write_joint_state_to_sim(home_pos, home_vel, env_ids=stand_ids)

    return _rsi_event


def run() -> None:
    import mjlab.tasks  # noqa: F401
    import simple_goalkeeper.tasks  # noqa: F401

    configure_torch_backends()

    cfg = tyro.cli(
        RsiPlayConfig,
        default=RsiPlayConfig(),
        config=mjlab.TYRO_FLAGS,
    )

    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    env_cfg = load_env_cfg(_TASK, play=True)
    agent_cfg = load_rl_cfg(_TASK)
    assert isinstance(agent_cfg, AMPRunnerCfg)

    if cfg.num_envs is not None:
        env_cfg.scene.num_envs = cfg.num_envs
    if cfg.no_terminations:
        env_cfg.terminations = {}

    rsi_event = _make_rsi_event(cfg.motion_pool, cfg.rsi_fraction)
    env_cfg.events["reset_from_motion_data"] = EventTermCfg(
        func=rsi_event, mode="reset",
    )

    # 2026-07-05: corrected to match mdp.events._SINGLE/_DOUBLE/_TRIPLE_THRESH
    # (this description had drifted out of sync with the actual thresholds).
    tier_desc = {
        "wide":   "DoubleStep + TripleStep  (wide pool, |cross_y| ≥ 0.50)",
        "double": "SafeMedium               (double pool, 0.20–0.40 m)",
        "triple": "SafeFar                  (triple pool, 0.40–0.50 m)",
        "all":    "all motions combined",
    }
    print(f"[INFO]: RSI pool = {cfg.motion_pool}  ({tier_desc[cfg.motion_pool]})")
    print(f"[INFO]: RSI fraction = {cfg.rsi_fraction:.0%}")

    os.environ.setdefault("MUJOCO_GL", "egl")
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)

    if cfg.difficulty is not None:
        env._ball_difficulty = float(cfg.difficulty)
        print(f"[INFO]: Difficulty overridden to {cfg.difficulty}")

    env = AMPEnvWrapper(env, clip_actions=agent_cfg.clip_actions, motion_dataset=agent_cfg.amp_data)

    # Resolve checkpoint path
    ckpt = Path(cfg.checkpoint_file)
    if not ckpt.is_absolute() and not ckpt.exists():
        parent = Path.cwd()
        for _ in range(6):
            parent = parent.parent
            candidate = parent / ckpt
            if candidate.exists():
                ckpt = candidate
                break
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    print(f"[INFO]: Checkpoint = {ckpt.name}")

    runner = AMPOnPolicyRunner(env, asdict(agent_cfg), log_dir=None, device=device)
    runner.load(str(ckpt), load_optimizer=False)
    policy = runner.get_inference_policy(device=device)

    analytics = AnalyticsPolicy(policy, env, enabled=cfg.analytics)

    resolved = cfg.viewer
    if resolved == "auto":
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        resolved = "native" if has_display else "viser"

    if resolved == "native":
        from mjlab.viewer import NativeMujocoViewer
        from mjlab.viewer.native.keys import KEY_V
        def _key_cb(key: int) -> None:
            if key == KEY_V:
                analytics.toggle()
        viewer = NativeMujocoViewer(env, analytics, key_callback=_key_cb)
        _patch_viewer_intercept_vis(viewer, env)
        viewer.run()
    else:
        from mjlab.viewer import ViserPlayViewer
        ViserPlayViewer(env, analytics).run()

    env.close()


if __name__ == "__main__":
    run()
