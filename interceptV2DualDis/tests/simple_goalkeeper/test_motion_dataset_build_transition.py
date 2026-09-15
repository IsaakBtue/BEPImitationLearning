"""Regression tests for MotionDataset.build_transition()'s randomized-
playback-speed interpolation -- specifically the per-trajectory boundary
clamp (beyondAMP/source/beyondAMP/beyondAMP/motion/motion_dataset.py).

Written as part of the 2026-09-15 AMP-discriminator-collapse investigation's
independent verification pass: the trajectory-boundary clamp was read and
judged correct by code inspection (see docs/superpowers/
isaacgym_vs_mjlab_flow.md, Phase 12), but had no direct test exercising the
actual numeric behavior -- these confirm it empirically instead of trusting
the read.
"""
import numpy as np
import torch


class _FakeRobot:
    def __init__(self, name_to_index):
        self._name_to_index = name_to_index

    def find_bodies(self, names, preserve_order=True):
        if isinstance(names, str):
            names = [names]
        indices = [self._name_to_index[n] for n in names]
        return indices, names


class _FakeScene:
    def __init__(self, robot):
        self._robot = robot

    def __getitem__(self, name):
        return self._robot


class _FakeEnv:
    def __init__(self, robot, device="cpu", step_dt=0.02):
        self.scene = _FakeScene(robot)
        self.device = device
        self.step_dt = step_dt


def _write_motion_npz(path, num_frames, joint_pos_base, fps=50.0, num_bodies=1, ndof=1):
    # joint_pos[i] = joint_pos_base + i, distinctly identifiable per-trajectory
    # (e.g. base=0 -> values 0..N-1; base=1000 -> values 1000..1000+N-1) so a
    # cross-trajectory blend would produce an out-of-range value, not just a
    # subtly-wrong one.
    joint_pos = (joint_pos_base + np.arange(num_frames)).astype(np.float32).reshape(-1, 1)
    np.savez(
        path,
        fps=np.array(fps),
        joint_pos=joint_pos,
        joint_vel=np.zeros((num_frames, ndof), dtype=np.float32),
        body_pos_w=np.zeros((num_frames, num_bodies, 3), dtype=np.float32),
        body_quat_w=np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (num_frames, num_bodies, 1)),
        body_lin_vel_w=np.zeros((num_frames, num_bodies, 3), dtype=np.float32),
        body_ang_vel_w=np.zeros((num_frames, num_bodies, 3), dtype=np.float32),
    )


def _make_dataset(tmp_path, frames_a=6, frames_b=6, base_a=0, base_b=1000, fps=50.0):
    from beyondAMP.motion.motion_dataset import MotionDataset, MotionDatasetCfg

    motion_a = tmp_path / "a.npz"
    motion_b = tmp_path / "b.npz"
    _write_motion_npz(motion_a, frames_a, base_a, fps=fps)
    _write_motion_npz(motion_b, frames_b, base_b, fps=fps)

    robot = _FakeRobot({"b0": 0})
    env = _FakeEnv(robot, step_dt=1.0 / fps)  # env_fps == fps, matches this project's real config

    cfg = MotionDatasetCfg(
        motion_files=[str(motion_a), str(motion_b)],
        body_names=["b0"],
        amp_obs_terms=["joint_pos"],
        anchor_name="b0",
    )
    return MotionDataset(cfg, env, device="cpu"), frames_a, frames_b, base_a, base_b


def test_build_transition_never_blends_across_trajectory_boundary(tmp_path):
    """Sampling near the TAIL of trajectory A must never pull a frame from
    trajectory B, even at the max ratio (1.25x)."""
    dataset, frames_a, frames_b, base_a, base_b = _make_dataset(tmp_path)

    torch.manual_seed(0)
    # Sample every valid t in trajectory A repeatedly, many draws each, to
    # exercise the full U(0.25, 1.25) ratio range including its max.
    t = torch.arange(0, frames_a - 1).repeat(500)
    tp1 = t + 1  # unused by build_transition itself, required by the call signature
    state_t, state_tp1 = dataset.build_transition(t, tp1)

    # Trajectory A's values live in [base_a, base_a + frames_a - 1].
    # Trajectory B's values live in [base_b, base_b + frames_b - 1].
    # A genuine cross-trajectory blend would produce a value strictly between
    # the two ranges (or inside B's range) -- assert it never happens.
    assert state_tp1.max().item() <= base_a + frames_a - 1 + 1e-4, (
        f"build_transition produced a next-state value ({state_tp1.max().item()}) "
        f"beyond trajectory A's own range [{base_a}, {base_a + frames_a - 1}] -- "
        "cross-trajectory blend into B"
    )
    assert state_tp1.min().item() >= base_a - 1e-4


def test_build_transition_state_t_is_exact_unblended_frame(tmp_path):
    """The 's_t' half must be the literal stored frame, never interpolated
    (matches G1's get_expert_obs -- only t+1 is a randomized-speed blend)."""
    dataset, frames_a, _, base_a, _ = _make_dataset(tmp_path)

    t = torch.arange(0, frames_a - 1)
    tp1 = t + 1
    state_t, _ = dataset.build_transition(t, tp1)

    expected = (base_a + t).to(torch.float32).unsqueeze(-1)
    assert torch.equal(state_t, expected)


def test_build_transition_produces_genuine_ratio_diversity(tmp_path):
    """The randomized playback ratio should actually vary the sampled
    next-state position across draws, not collapse to a fixed offset --
    this is the entire stated purpose of the mechanism (velocity/displacement
    diversity, ported from G1's get_expert_obs)."""
    dataset, frames_a, _, base_a, _ = _make_dataset(tmp_path, frames_a=10)

    torch.manual_seed(1)
    t = torch.full((2000,), 2)  # fixed base frame, repeated many times
    tp1 = t + 1
    _, state_tp1 = dataset.build_transition(t, tp1)

    displacement = state_tp1.squeeze(-1) - (base_a + 2)
    # ratio ~ U(0.25, 1.25) at fps==env_fps, so displacement should spread
    # roughly across that same range, not sit at one fixed value.
    assert displacement.std().item() > 0.15, (
        f"displacement std ({displacement.std().item()}) too low -- ratio "
        "sampling may not be genuinely randomized"
    )
    assert 0.15 < displacement.min().item() < 0.4
    assert 1.1 < displacement.max().item() < 1.4
