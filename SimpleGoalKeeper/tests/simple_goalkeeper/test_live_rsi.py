import torch


class _FakeRobotData:
    def __init__(self, joint_pos, default_joint_pos, soft_limits):
        self.joint_pos = joint_pos
        self.default_joint_pos = default_joint_pos
        self.soft_joint_pos_limits = soft_limits


class _FakeRobot:
    def __init__(self, joint_pos, default_joint_pos, soft_limits):
        self.data = _FakeRobotData(joint_pos, default_joint_pos, soft_limits)
        self.written_pos = None
        self.written_vel = None
        self.written_ids = None
        self.root_pose_calls = 0
        self.root_velocity_calls = 0

    def write_joint_state_to_sim(self, joint_pos, joint_vel, env_ids):
        self.written_pos = joint_pos
        self.written_vel = joint_vel
        self.written_ids = env_ids

    def write_root_link_pose_to_sim(self, *a, **k):
        self.root_pose_calls += 1

    def write_root_link_velocity_to_sim(self, *a, **k):
        self.root_velocity_calls += 1


class _Scene:
    def __init__(self, robot):
        self._robot = robot

    def __getitem__(self, name):
        return self._robot


class _FakeEnv:
    def __init__(self, num_envs, robot, device="cpu"):
        self.num_envs = num_envs
        self.device = device
        self.scene = _Scene(robot)


def _make_env(num_envs, num_dof, joint_values):
    """joint_values: (num_envs, num_dof) tensor of each env's CURRENT live dof_pos."""
    default_joint_pos = torch.zeros(num_envs, num_dof)
    soft_limits = torch.tensor([[-3.0, 3.0]] * num_dof).unsqueeze(0).repeat(num_envs, 1, 1)
    robot = _FakeRobot(joint_values, default_joint_pos, soft_limits)
    env = _FakeEnv(num_envs=num_envs, robot=robot)
    return env, robot


def test_reset_root_state_is_never_touched():
    from simple_goalkeeper.mdp.events import MotionResetManager

    num_envs, num_dof = 4, 2
    joint_values = torch.arange(num_envs * num_dof, dtype=torch.float32).reshape(num_envs, num_dof)
    env, robot = _make_env(num_envs, num_dof, joint_values)

    mgr = MotionResetManager()
    env_ids = torch.tensor([0], dtype=torch.int32)
    mgr.reset(env, env_ids, rsi_fraction=1.0)

    # Root pose/velocity are handled by the separate reset_base event, matching
    # G1's continue_keep, which only ever touches dof_pos/dof_vel.
    assert robot.root_pose_calls == 0
    assert robot.root_velocity_calls == 0


def test_reset_dof_vel_always_zero_on_both_branches():
    from simple_goalkeeper.mdp.events import MotionResetManager

    num_envs, num_dof = 4, 2
    joint_values = torch.ones(num_envs, num_dof) * 2.5
    env, robot = _make_env(num_envs, num_dof, joint_values)
    mgr = MotionResetManager()
    env_ids = torch.tensor([0], dtype=torch.int32)

    mgr.reset(env, env_ids, rsi_fraction=1.0)   # forces the donor-copy branch
    assert torch.allclose(robot.written_vel, torch.zeros(1, num_dof))

    mgr.reset(env, env_ids, rsi_fraction=0.0)   # forces the default-pose branch
    assert torch.allclose(robot.written_vel, torch.zeros(1, num_dof))


def test_reset_rsi_fraction_1_copies_dof_pos_from_another_live_env_and_clamps(monkeypatch):
    from simple_goalkeeper.mdp.events import MotionResetManager

    num_envs, num_dof = 4, 3
    # Env 1 (the donor torch.randint is forced to pick) is out of joint range.
    joint_values = torch.tensor([
        [0.0, 0.0, 0.0],
        [5.0, 5.0, 5.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    env, robot = _make_env(num_envs, num_dof, joint_values)
    mgr = MotionResetManager()
    env_ids = torch.tensor([0], dtype=torch.int32)

    monkeypatch.setattr(torch, "randint", lambda *a, **k: torch.ones(1, dtype=torch.long))
    mgr.reset(env, env_ids, rsi_fraction=1.0)  # rsi_fraction=1.0 -> always the donor branch

    assert robot.written_ids.tolist() == [0]
    # Donor (env 1) had joint_pos == 5.0, clamped to the soft limit [-3, 3].
    assert torch.allclose(robot.written_pos, torch.tensor([[3.0, 3.0, 3.0]]))


def test_reset_rsi_fraction_0_uses_default_pose_not_a_donor():
    from simple_goalkeeper.mdp.events import MotionResetManager

    num_envs, num_dof = 4, 2
    joint_values = torch.ones(num_envs, num_dof) * 9.0  # would be obviously wrong if ever copied
    env, robot = _make_env(num_envs, num_dof, joint_values)
    mgr = MotionResetManager()
    env_ids = torch.tensor([0], dtype=torch.int32)

    mgr.reset(env, env_ids, rsi_fraction=0.0)  # rsi_fraction=0.0 -> always the default-pose branch

    assert robot.written_ids.tolist() == [0]
    assert torch.allclose(robot.written_pos, torch.zeros(1, num_dof))


def test_reset_donor_pool_is_not_tier_restricted_and_not_batch_excluded(monkeypatch):
    """No side/tier matching (G1 has none) and no exclusion of the current
    reset batch (G1 samples torch.randint(0, num_envs, ...) unconditionally,
    so a resetting env CAN be sampled as its own or another resetting env's
    donor — this test locks in that this project matches that, deliberately,
    rather than silently reintroducing tier/exclusion logic)."""
    from simple_goalkeeper.mdp.events import MotionResetManager

    num_envs, num_dof = 3, 1
    joint_values = torch.tensor([[1.0], [2.0], [3.0]])
    env, robot = _make_env(num_envs, num_dof, joint_values)
    mgr = MotionResetManager()
    env_ids = torch.tensor([0, 1, 2], dtype=torch.int32)  # the WHOLE population resets together

    torch.manual_seed(0)
    monkeypatch.setattr(torch, "randint", lambda *a, **k: torch.zeros(len(env_ids), dtype=torch.long))
    mgr.reset(env, env_ids, rsi_fraction=1.0)

    # torch.randint patched to always return index 0 -> every env, including
    # env 0 reading from itself, copies joint_values[0] == 1.0. No exclusion,
    # no tier filter — the donor pool is genuinely env.num_envs wide.
    assert torch.allclose(robot.written_pos, torch.full((3, 1), 1.0))


def test_reset_single_coin_flip_covers_the_whole_batch_at_once(monkeypatch):
    """G1 draws ONE torch.rand(1) per reset() call, not one per env — so
    every env in this call must land on the SAME branch."""
    from simple_goalkeeper.mdp.events import MotionResetManager

    num_envs, num_dof = 5, 1
    joint_values = torch.arange(num_envs, dtype=torch.float32).reshape(num_envs, 1)
    env, robot = _make_env(num_envs, num_dof, joint_values)
    mgr = MotionResetManager()
    env_ids = torch.tensor([0, 1, 2, 3], dtype=torch.int32)

    monkeypatch.setattr(torch, "rand", lambda *a, **k: torch.tensor([0.99]))  # forces donor branch
    mgr.reset(env, env_ids, rsi_fraction=0.8)
    assert robot.written_ids.tolist() == [0, 1, 2, 3]
    assert not torch.allclose(robot.written_pos, torch.zeros(4, 1))  # not the default pose

    monkeypatch.setattr(torch, "rand", lambda *a, **k: torch.tensor([0.01]))  # forces default branch
    mgr.reset(env, env_ids, rsi_fraction=0.8)
    assert robot.written_ids.tolist() == [0, 1, 2, 3]
    assert torch.allclose(robot.written_pos, torch.zeros(4, 1))
