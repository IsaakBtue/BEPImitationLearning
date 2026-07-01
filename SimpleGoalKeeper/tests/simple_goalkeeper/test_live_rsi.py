import torch

from simple_goalkeeper.mdp.events import _POOL_KEYS, _POOL_ID, _select_live_donors


def test_pool_id_covers_six_side_tier_combinations():
    assert len(_POOL_KEYS) == 6
    assert set(_POOL_KEYS) == {
        ("left", "double"), ("left", "triple"), ("left", "wide"),
        ("right", "double"), ("right", "triple"), ("right", "wide"),
    }
    assert set(_POOL_ID.values()) == {0, 1, 2, 3, 4, 5}
    assert set(_POOL_ID.keys()) == set(_POOL_KEYS)


def test_select_live_donors_matches_pool_and_maturity():
    # 6 envs: pool assignment per env, episode age per env.
    pool_id = torch.tensor([0, 0, 1, 0, -1, 0])
    episode_steps = torch.tensor([50, 3, 50, 50, 50, 50])
    exclude_ids = torch.tensor([], dtype=torch.long)

    donors = _select_live_donors(
        pool_id, episode_steps, exclude_ids,
        target_pool=0, min_maturity_steps=10,
    )
    # env 0: pool matches, mature -> eligible
    # env 1: pool matches, but only 3 steps old -> excluded (immature)
    # env 2: wrong pool -> excluded
    # env 3: pool matches, mature -> eligible
    # env 4: pool -1 (standing) -> excluded
    # env 5: pool matches, mature -> eligible
    assert sorted(donors.tolist()) == [0, 3, 5]


def test_select_live_donors_excludes_current_reset_batch():
    pool_id = torch.tensor([0, 0, 0])
    episode_steps = torch.tensor([50, 50, 50])
    exclude_ids = torch.tensor([1], dtype=torch.long)

    donors = _select_live_donors(
        pool_id, episode_steps, exclude_ids,
        target_pool=0, min_maturity_steps=10,
    )
    assert sorted(donors.tolist()) == [0, 2]


def test_select_live_donors_returns_empty_when_no_match():
    pool_id = torch.tensor([1, 2, 3])
    episode_steps = torch.tensor([50, 50, 50])
    exclude_ids = torch.tensor([], dtype=torch.long)

    donors = _select_live_donors(
        pool_id, episode_steps, exclude_ids,
        target_pool=0, min_maturity_steps=10,
    )
    assert donors.numel() == 0


class _FakeRobotData:
    def __init__(self, joint_pos, soft_limits):
        self.joint_pos = joint_pos
        self.soft_joint_pos_limits = soft_limits


class _FakeRobot:
    def __init__(self, joint_pos, soft_limits):
        self.data = _FakeRobotData(joint_pos, soft_limits)
        self.written_pos = None
        self.written_vel = None
        self.written_ids = None

    def write_joint_state_to_sim(self, joint_pos, joint_vel, env_ids):
        self.written_pos = joint_pos
        self.written_vel = joint_vel
        self.written_ids = env_ids


class _FakeEnv:
    def __init__(self, num_envs, device="cpu"):
        self.num_envs = num_envs
        self.device = device


def test_write_live_donor_state_copies_joint_pos_only_and_clamps():
    from simple_goalkeeper.mdp.events import MotionResetManager

    num_dof = 4
    joint_pos = torch.tensor([
        [0.0, 0.0, 0.0, 0.0],   # env 0 (will be reset — irrelevant source)
        [1.0, 1.0, 1.0, 1.0],   # env 1 (donor)
        [2.0, 2.0, 2.0, 2.0],   # env 2 (donor)
        [5.0, 5.0, 5.0, 5.0],   # env 3 (donor, out of joint limits — must clamp)
    ])
    soft_limits = torch.tensor([[-3.0, 3.0]] * num_dof).unsqueeze(0).repeat(4, 1, 1)
    robot = _FakeRobot(joint_pos, soft_limits)
    env = _FakeEnv(num_envs=4)

    mgr = MotionResetManager()
    ids = torch.tensor([0], dtype=torch.long)
    donor_idx = torch.tensor([3], dtype=torch.long)  # force-pick the out-of-range donor

    torch.manual_seed(0)
    mgr._write_live_donor_state(env, ids, donor_idx, robot)

    assert robot.written_ids.tolist() == [0]
    # Clamped to [-3, 3] even though donor env 3 had joint_pos == 5.0.
    assert torch.allclose(robot.written_pos, torch.tensor([[3.0, 3.0, 3.0, 3.0]]))
    # Velocities are always zero for live-donor resets (never copied from the
    # donor) — this is the exact fix for the yaw/velocity-drift bug from the
    # prior attempt (SimpleGoalKeeperObsHis, 2026-06-18).
    assert torch.allclose(robot.written_vel, torch.zeros(1, num_dof))


def test_reset_prefers_live_donor_when_pool_is_large_and_warm(monkeypatch):
    from simple_goalkeeper.mdp import events as events_mod
    from simple_goalkeeper.mdp.events import MotionResetManager

    monkeypatch.setattr(events_mod, "_MIN_DONOR_POOL_SIZE", 2)
    monkeypatch.setattr(events_mod, "_LIVE_RSI_WARMUP_STEPS", 0)
    monkeypatch.setattr(events_mod, "_MIN_MATURITY_STEPS", 0)

    num_envs = 4
    num_dof = 2
    joint_pos = torch.zeros(num_envs, num_dof)
    soft_limits = torch.tensor([[-3.0, 3.0]] * num_dof).unsqueeze(0).repeat(num_envs, 1, 1)
    robot = _FakeRobot(joint_pos, soft_limits)
    robot.data.default_joint_pos = torch.zeros(num_envs, num_dof)
    robot.write_root_link_pose_to_sim = lambda *a, **k: None
    robot.write_root_link_velocity_to_sim = lambda *a, **k: None

    class _Scene:
        def __init__(self, robot):
            self._robot = robot
            self.env_origins = torch.zeros(num_envs, 3)

        def __getitem__(self, name):
            return self._robot

    env = _FakeEnv(num_envs=num_envs)
    env.scene = _Scene(robot)
    env.common_step_counter = 999
    env.episode_length_buf = torch.tensor([0, 100, 100, 100])  # env 0 is resetting now
    env._rsi_cross_y = torch.tensor([0.9, 0.0, 0.0, 0.0])  # env 0 wants a "wide" target
    env._rsi_pool_id = torch.tensor([-1, 2, 2, -1])  # envs 1,2 already tagged "left wide" (pool 2)

    mgr = MotionResetManager()
    mgr.frames = {"joint_pos": torch.zeros(1, num_dof), "joint_vel": torch.zeros(1, num_dof),
                  "root_pos": torch.zeros(1, 3), "root_quat": torch.zeros(1, 4),
                  "root_lin_vel": torch.zeros(1, 3), "root_ang_vel": torch.zeros(1, 3)}

    env_ids = torch.tensor([0], dtype=torch.int32)
    torch.manual_seed(0)
    mgr.reset(env, env_ids, rsi_fraction=1.0)

    assert robot.written_ids is not None
    assert robot.written_ids.tolist() == [0]
    assert mgr.live_rsi_hits == 1
    assert mgr.live_rsi_total == 1
    # env 0 gets tagged into the pool it was just assigned (left, wide) = 2.
    assert env._rsi_pool_id[0].item() == 2


def test_live_rsi_usage_curriculum_reports_and_resets_counters():
    from simple_goalkeeper.mdp.events import MotionResetManager, live_rsi_usage_curriculum

    mgr = MotionResetManager.get()
    mgr.live_rsi_hits = 3
    mgr.live_rsi_total = 4

    result = live_rsi_usage_curriculum(env=None, env_ids=None)

    assert torch.isclose(result["live_rsi_fraction"], torch.tensor(0.75))
    assert mgr.live_rsi_hits == 0
    assert mgr.live_rsi_total == 0


def test_live_rsi_usage_curriculum_handles_zero_total():
    from simple_goalkeeper.mdp.events import MotionResetManager, live_rsi_usage_curriculum

    mgr = MotionResetManager.get()
    mgr.live_rsi_hits = 0
    mgr.live_rsi_total = 0

    result = live_rsi_usage_curriculum(env=None, env_ids=None)

    assert torch.isclose(result["live_rsi_fraction"], torch.tensor(0.0))
