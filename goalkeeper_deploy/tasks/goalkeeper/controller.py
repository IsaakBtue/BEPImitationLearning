"""Custom MuJoCo controller for goalkeeper sim2sim.

Extends BaseController directly (rather than MujocoController) because the
goalkeeper scene contains a ball freejoint whose qpos/qvel must be kept
separate from the robot's 23 DOF joints.

Scene layout in qpos:
  [0:7]  robot freejoint (pos xyz + quat wxyz)
  [7:30] robot joints (23)
  [30:37] ball freejoint (pos xyz + quat wxyz)

Scene layout in qvel:
  [0:6]  robot freejoint: qvel[0:3]=world-frame lin vel, qvel[3:6]=body-frame ang vel
  [6:29] robot joints vel (23)
  [29:35] ball freejoint vel
"""
from __future__ import annotations

import math
from pathlib import Path
from time import sleep
from typing import Optional

import mujoco
import mujoco.viewer
import numpy as np
import torch

from booster_deploy.controllers.base_controller import BaseController
from booster_deploy.controllers.controller_cfg import ControllerCfg


def _rot_world_to_body(q_wxyz: np.ndarray, v_w: np.ndarray) -> np.ndarray:
    """Rotate world-frame vector into body frame. q = [w, x, y, z]."""
    w, x, y, z = q_wxyz
    v0, v1, v2 = v_w
    r0 = (1-2*(y*y+z*z))*v0 + 2*(x*y+w*z)*v1 + 2*(x*z-w*y)*v2
    r1 = 2*(x*y-w*z)*v0 + (1-2*(x*x+z*z))*v1 + 2*(y*z+w*x)*v2
    r2 = 2*(x*z+w*y)*v0 + 2*(y*z-w*x)*v1 + (1-2*(x*x+y*y))*v2
    return np.array([r0, r1, r2], dtype=np.float32)

# Absolute path to training T1 MJCF
_T1_XML = str(
    Path(__file__).resolve().parent.parent.parent.parent
    / "my_mjlab_project_booster_t1"
    / "src" / "my_mjlab_project_booster_t1"
    / "assets" / "booster_t1" / "T1_serial_clean.xml"
)

_JOINT_NAMES = [
    "AAHead_yaw", "Head_pitch",
    "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
    "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
    "Waist",
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw",
    "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw",
    "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll",
]
_EFFORT = [7, 7, 36, 36, 36, 36, 36, 36, 36, 36, 40, 55, 40, 40, 65, 50, 50, 55, 40, 40, 65, 50, 50]

# Training initial orientation: +90° yaw so robot faces +Y (ball approach direction)
_INIT_QUAT = [0.7071068, 0.0, 0.0, 0.7071068]   # [w, x, y, z]
_INIT_POS = [0.0, 0.0, 0.78]


def _build_scene(physics_dt: float) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Build goalkeeper MuJoCo model: T1 robot + floor + ball + motor actuators."""
    spec = mujoco.MjSpec.from_file(_T1_XML)
    spec.option.timestep = physics_dt

    # Floor
    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [20.0, 20.0, 0.01]
    floor.rgba = [0.55, 0.55, 0.55, 1.0]
    floor.pos = [0.0, 0.0, 0.0]

    # Ball (soccer: radius=0.11 m, mass=0.42 kg)
    ball_body = spec.worldbody.add_body()
    ball_body.name = "ball"
    ball_body.pos = [0.0, 5.0, 1.0]
    ball_body.add_freejoint().name = "ball_joint"
    bg = ball_body.add_geom()
    bg.name = "ball_geom"
    bg.type = mujoco.mjtGeom.mjGEOM_SPHERE
    bg.size = [0.11, 0.0, 0.0]
    bg.mass = 0.42
    bg.rgba = [1.0, 1.0, 0.0, 1.0]
    bg.friction = [0.4, 0.005, 0.0001]

    # Motor actuators (PD torques applied externally in ctrl_step)
    for i, jname in enumerate(_JOINT_NAMES):
        act = spec.add_actuator()
        act.set_to_motor()
        act.name = f"motor_{jname}"
        act.target = jname
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.forcerange = [-float(_EFFORT[i]), float(_EFFORT[i])]
        act.gainprm[0] = 1.0

    model = spec.compile()
    data = mujoco.MjData(model)
    return model, data


class GoalkeeperMujocoController(BaseController):
    """Sim2sim MuJoCo controller for goalkeeper policy.

    Differences from the standard MujocoController:
    - Builds scene dynamically (T1 + floor + ball + motor actuators).
    - qpos[7:30] and qvel[6:29] are robot joints; ball occupies [30:] / [29:].
    - Exposes ball_pos_w, ball_vel_w, left_hand_pos_w, right_hand_pos_w for the policy.
    - Auto-relaunches ball periodically.
    """

    ball_pos_w: torch.Tensor     # (3,) world frame
    ball_vel_w: torch.Tensor     # (3,) world frame
    left_hand_pos_w: torch.Tensor   # (3,) world frame
    right_hand_pos_w: torch.Tensor  # (3,) world frame

    def __init__(self, cfg: ControllerCfg) -> None:
        super().__init__(cfg)
        physics_dt = cfg.mujoco.physics_dt
        self.mj_model, self.mj_data = _build_scene(physics_dt)
        self.decimation = cfg.mujoco.decimation

        # Body IDs (determined after spec.compile())
        def _bid(name: str) -> int:
            return mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, name)

        self.ball_id = _bid("ball")
        self.left_hand_id = _bid("left_hand_link")
        self.right_hand_id = _bid("right_hand_link")

        # Preallocate output tensors
        self.ball_pos_w = torch.zeros(3)
        self.ball_vel_w = torch.zeros(3)
        self.left_hand_pos_w = torch.zeros(3)
        self.right_hand_pos_w = torch.zeros(3)

        # Place robot at training initial pose
        default_pos_np = self.robot.default_joint_pos.numpy()
        self.mj_data.qpos[:3] = _INIT_POS
        self.mj_data.qpos[3:7] = _INIT_QUAT
        self.mj_data.qpos[7:30] = default_pos_np
        # Ball initial position (far away, will be launched after start)
        self.mj_data.qpos[30:33] = [0.0, 8.0, 1.0]
        self.mj_data.qpos[33:37] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(self.mj_model, self.mj_data)

        self._ball_launch_interval_s: float = 3.5
        self._last_launch_time_s: float = -self._ball_launch_interval_s

        # Ball visibility state — mirrors training catchstep + flying + random_vanish gates
        self._ball_step: int = 0         # catchstep countdown (50→0 per launch)
        self._ball_prev_y: float = 0.0   # previous ball world-Y for approach detection
        self._ball_vanish_step: int = 0  # random vanish threshold sampled at each launch
        self._ball_visible_step: int = 0 # consecutive steps ball passed the flying gate
        self.ball_visible: bool = False

    # ------------------------------------------------------------------
    # BaseController interface
    # ------------------------------------------------------------------

    def update_state(self) -> None:
        """Read simulator state into robot.data and controller ball/hand fields."""
        q = self.mj_data.qpos.astype(np.float32)
        dq = self.mj_data.qvel.astype(np.float32)

        # Robot base
        base_pos = q[:3]
        base_quat = q[3:7]   # [w,x,y,z]

        # BUG FIX 1: MuJoCo freejoint qvel[0:3] is world-frame linear velocity
        # (derivative of world-frame position). Training uses quat_apply_inverse to
        # rotate it into body frame before feeding to the policy. Apply that rotation.
        base_lin_vel_b = _rot_world_to_body(base_quat, dq[:3])
        # qvel[3:6] for a MuJoCo freejoint is body-frame angular velocity — correct as-is.
        base_ang_vel_b = dq[3:6]

        # Robot joints
        dof_pos = q[7:30]
        dof_vel = dq[6:29]
        # qfrc_actuator has shape (nv,); robot joints are dofs 6:29
        dof_torque = self.mj_data.qfrc_actuator[6:29].astype(np.float32)

        self.robot.data.root_pos_w = torch.from_numpy(base_pos)
        self.robot.data.root_quat_w = torch.from_numpy(base_quat)
        self.robot.data.root_lin_vel_b = torch.from_numpy(base_lin_vel_b)
        self.robot.data.root_ang_vel_b = torch.from_numpy(base_ang_vel_b)
        self.robot.data.joint_pos = torch.from_numpy(dof_pos)
        self.robot.data.joint_vel = torch.from_numpy(dof_vel)
        self.robot.data.feedback_torque = torch.from_numpy(dof_torque)

        # Advance catchstep countdown (mirrors training's _catchstep per policy step)
        if self._ball_step > 0:
            self._ball_step -= 1

        # Ball position and velocity in world frame
        ball_pos_np = self.mj_data.xpos[self.ball_id].astype(np.float32)
        ball_vel_np = self.mj_data.cvel[self.ball_id, 3:].astype(np.float32)

        # BUG FIX 2: apply visibility masking to match training distribution
        self.ball_visible = self._compute_ball_visibility(ball_pos_np)
        self.ball_pos_w = torch.from_numpy(ball_pos_np)
        self.ball_vel_w = torch.from_numpy(ball_vel_np)

        # Hand positions (world frame)
        self.left_hand_pos_w = torch.from_numpy(
            self.mj_data.xpos[self.left_hand_id].astype(np.float32))
        self.right_hand_pos_w = torch.from_numpy(
            self.mj_data.xpos[self.right_hand_id].astype(np.float32))

    def _compute_ball_visibility(self, ball_pos_w: np.ndarray) -> bool:
        """Mirror training ball visibility logic (observations.py).

        Three independent gates — all must pass for ball to be visible:
          initial_vanish: warmup expired (_ball_step < 43 out of 50)
          flying:         ball approaching, in view, within bounds
          random_vanish:  ball hasn't randomly disappeared mid-flight
        """
        _STARTSTEP = 43
        initial_vanish_ok = self._ball_step < _STARTSTEP

        # Training uses env_origins (fixed at [0,0,0] for env 0) as the reference point,
        # not the robot's current position. Using robot_pos_w would drift as the robot
        # moves laterally, causing visibility to disagree with the training distribution.
        ball_y_local = float(ball_pos_w[1])   # world Y — env origin Y = 0
        ball_x_local = float(ball_pos_w[0])   # world X — env origin X = 0
        ball_z = float(ball_pos_w[2])

        approaching = (ball_y_local < self._ball_prev_y) or (self._ball_prev_y == 0.0)
        self._ball_prev_y = ball_y_local

        flying = (
            ball_y_local > 0.05 and
            ball_y_local < 3.4 and
            abs(ball_x_local) < 2.0 and
            ball_z < 1.8 and
            approaching
        )

        if flying:
            self._ball_visible_step += 1
        else:
            self._ball_visible_step = 0

        random_vanish = self._ball_visible_step > self._ball_vanish_step
        return bool(initial_vanish_ok and flying and not random_vanish)

    def ctrl_step(self, dof_targets: torch.Tensor) -> None:
        """Apply PD torques and advance physics by decimation steps."""
        targets_np = dof_targets.cpu().numpy()
        kp = self.robot.joint_stiffness.numpy()
        kd = self.robot.joint_damping.numpy()
        limit = self.robot.effort_limit.numpy()

        for _ in range(self.decimation):
            pos = self.mj_data.qpos.astype(np.float32)[7:30]
            vel = self.mj_data.qvel.astype(np.float32)[6:29]
            self.mj_data.ctrl[:] = np.clip(
                kp * (targets_np - pos) - kd * vel, -limit, limit
            )
            mujoco.mj_step(self.mj_model, self.mj_data)

        # Auto-relaunch ball when it passes the robot or hits the ground
        current_t = self._step_count * self.cfg.policy_dt
        ball_y_world = float(self.mj_data.qpos[31])
        ball_z_world = float(self.mj_data.qpos[32])
        if (current_t - self._last_launch_time_s > self._ball_launch_interval_s
                or ball_y_world < -1.5 or ball_z_world < 0.05):
            self._launch_ball()
            self._last_launch_time_s = current_t

    def _reset_sim(self) -> None:
        """Restore robot and ball to initial state (called on Backspace)."""
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        default_pos_np = self.robot.default_joint_pos.numpy()
        self.mj_data.qpos[:3] = _INIT_POS
        self.mj_data.qpos[3:7] = _INIT_QUAT
        self.mj_data.qpos[7:30] = default_pos_np
        self.mj_data.qpos[30:33] = [0.0, 8.0, 1.0]
        self.mj_data.qpos[33:37] = [1.0, 0.0, 0.0, 0.0]
        self.mj_data.qvel[:] = 0.0
        mujoco.mj_forward(self.mj_model, self.mj_data)
        # Reset ball timing and policy state
        self._last_launch_time_s = -self._ball_launch_interval_s
        self._ball_step = 0
        self._ball_prev_y = 0.0
        self._ball_vanish_step = 0
        self._ball_visible_step = 0
        self.ball_visible = False
        self._step_count = 0
        self.policy.reset()

    def run(self) -> None:
        _GLFW_KEY_BACKSPACE = 259
        _needs_reset = [False]  # mutable flag accessible from callback

        def _key_callback(keycode: int) -> None:
            if keycode == _GLFW_KEY_BACKSPACE:
                _needs_reset[0] = True

        with mujoco.viewer.launch_passive(
            self.mj_model, self.mj_data, key_callback=_key_callback
        ) as viewer:
            viewer.cam.elevation = -20
            self.update_state()
            self.start()
            while viewer.is_running() and self.is_running:
                if _needs_reset[0]:
                    _needs_reset[0] = False
                    self._reset_sim()
                sleep(self.cfg.mujoco.physics_dt * self.decimation)
                self.update_state()
                dof_targets = self.policy_step()
                self.ctrl_step(dof_targets)
                viewer.cam.lookat[:] = self.mj_data.qpos[:3]
                viewer.sync()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _launch_ball(self) -> None:
        """Reset ball using same formula and ranges as MultiMotionCommand._reset_ball
        (commands.py), which is the authoritative ball spawner in play mode.

        Ball targets the robot's LEFT side (negative X) — matches the lefthand save
        motion. x_end ∈ [-0.84, -0.2], z_end ∈ [0.4, 1.2], t_flight ∈ [0.4, 1.0].
        Projectile-motion vz accounts for gravity so arc shape matches training.
        """
        rng = np.random.default_rng()

        # Reset visibility state for the new ball trajectory
        self._ball_step = 50
        self._ball_prev_y = 0.0
        self._ball_vanish_step = int(rng.integers(0, 30))
        self._ball_visible_step = 0

        g = 9.81

        # Source ranges: MultiMotionCommand._reset_ball / _BALL_END_RANGES[0] (lefthand)
        x_start  = rng.uniform(-1.8,  1.8)
        y_start  = rng.uniform(3.0,   5.0)
        z_start  = rng.uniform(0.3,   1.8)
        x_end    = rng.uniform(-0.84, -0.2)   # left side only (lefthand save)
        z_end    = rng.uniform(0.4,   1.2)
        t_flight = rng.uniform(0.4,   1.0)

        dx = x_end - x_start
        dy = -y_start - 0.3      # target Y = -0.3 (just behind goal line)
        dz = z_end - z_start

        vx = dx / t_flight
        vy = dy / t_flight
        vz = (dz + 0.5 * g * t_flight ** 2) / t_flight

        self.mj_data.qpos[30:33] = [x_start, y_start, z_start]
        self.mj_data.qpos[33:37] = [1.0, 0.0, 0.0, 0.0]
        self.mj_data.qvel[29:32] = [vx, vy, vz]
        self.mj_data.qvel[32:35] = 0.0
        mujoco.mj_forward(self.mj_model, self.mj_data)
