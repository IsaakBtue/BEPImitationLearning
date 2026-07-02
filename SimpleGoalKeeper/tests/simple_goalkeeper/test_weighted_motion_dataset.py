import numpy as np
import pytest


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
    def __init__(self, robot, device="cpu"):
        self.scene = _FakeScene(robot)
        self.device = device


def _write_motion_npz(path, num_frames, num_bodies=2, ndof=3, fps=30.0):
    np.savez(
        path,
        fps=np.array(fps),
        joint_pos=np.zeros((num_frames, ndof), dtype=np.float32),
        joint_vel=np.zeros((num_frames, ndof), dtype=np.float32),
        body_pos_w=np.zeros((num_frames, num_bodies, 3), dtype=np.float32),
        body_quat_w=np.tile(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (num_frames, num_bodies, 1)
        ),
        body_lin_vel_w=np.zeros((num_frames, num_bodies, 3), dtype=np.float32),
        body_ang_vel_w=np.zeros((num_frames, num_bodies, 3), dtype=np.float32),
    )


def _make_dataset(tmp_path, traj_weights):
    from beyondAMP.motion.motion_dataset import MotionDatasetCfg
    from beyondAMP.motion.weighted_motion_dataset import WeightedMotionDataset

    motion_a = tmp_path / "a.npz"
    motion_b = tmp_path / "b.npz"
    _write_motion_npz(motion_a, num_frames=5)  # 4 transitions
    _write_motion_npz(motion_b, num_frames=5)  # 4 transitions

    robot = _FakeRobot({"b0": 0, "b1": 1})
    env = _FakeEnv(robot)

    cfg = MotionDatasetCfg(
        motion_files=[str(motion_a), str(motion_b)],
        body_names=["b0", "b1"],
        amp_obs_terms=["body_pos_w"],
        anchor_name="b0",
    )

    return WeightedMotionDataset(cfg, env, device="cpu", traj_weights=traj_weights)


def test_traj_weights_scale_per_transition_sampling_weight(tmp_path):
    dataset = _make_dataset(tmp_path, traj_weights=[1.0, 3.0])

    weights = dataset.weights
    a_weight = weights[:4].mean().item()
    b_weight = weights[4:].mean().item()
    assert b_weight / a_weight == pytest.approx(3.0)


def test_no_traj_weights_defaults_to_uniform(tmp_path):
    dataset = _make_dataset(tmp_path, traj_weights=None)

    weights = dataset.weights
    assert weights[:4].mean().item() == pytest.approx(weights[4:].mean().item())
