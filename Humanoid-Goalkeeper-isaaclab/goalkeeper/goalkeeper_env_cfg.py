"""Configuration for the Goalkeeper environment (IsaacLab port of g1_29_config.py)."""
from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.converters import UrdfConverterCfg
from isaaclab.sim.spawners.from_files import UrdfFileCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
_GK_ROOT = os.path.join(os.path.dirname(_REPO_ROOT), "Humanoid-Goalkeeper")
_LEGGED_GYM_ROOT = os.path.join(_GK_ROOT, "legged_gym")
_G1_URDF = os.path.join(_LEGGED_GYM_ROOT, "resources", "robots", "g1", "urdf", "g1_29.urdf")
_DATASETS_DIR = os.path.join(_GK_ROOT, "legged_gym", "resources", "datasets", "goalkeeper")

# ---------------------------------------------------------------------------
# G1 Robot ArticulationCfg (manual PD — effort control)
# ---------------------------------------------------------------------------
G1_CFG = ArticulationCfg(
    spawn=UrdfFileCfg(
        asset_path=_G1_URDF,
        fix_base=False,
        merge_fixed_joints=False,
        self_collision=False,
        replace_cylinders_with_capsules=True,
        activate_contact_sensors=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            target_type="none",  # effort control: drives set to 0 in USD
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        joint_pos={
            "left_hip_pitch_joint": -0.1,
            "left_hip_roll_joint": 0.2,
            "left_hip_yaw_joint": 0.0,
            "left_knee_joint": 0.3,
            "left_ankle_pitch_joint": -0.2,
            "left_ankle_roll_joint": -0.2,
            "right_hip_pitch_joint": -0.1,
            "right_hip_roll_joint": -0.2,
            "right_hip_yaw_joint": 0.0,
            "right_knee_joint": 0.3,
            "right_ankle_pitch_joint": -0.2,
            "right_ankle_roll_joint": 0.2,
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.5,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 1.2,
            "left_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": -0.5,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 1.2,
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
        },
    ),
    actuators={
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=0.0,
            damping=0.0,
            effort_limit_sim=400.0,
        ),
    },
)

# ---------------------------------------------------------------------------
# Ball RigidObjectCfg (sphere, mass=0.1 kg, radius=0.1 m)
# ---------------------------------------------------------------------------
BALL_CFG = RigidObjectCfg(
    spawn=sim_utils.SphereCfg(
        radius=0.1,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            max_linear_velocity=50.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=5.0,
            enable_gyroscopic_forces=True,
            disable_gravity=False,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.4,
            dynamic_friction=0.4,
            restitution=0.8,
            friction_combine_mode="multiply",
            restitution_combine_mode="max",
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(3.0, 0.0, 1.5)),
)


# ---------------------------------------------------------------------------
# Main environment config
# ---------------------------------------------------------------------------
@configclass
class GoalkeeperEnvCfg(DirectRLEnvCfg):
    """Configuration for the GoalkeeperEnv."""

    # Observation / action dimensions
    num_actor_history: int = 10
    num_dofs: int = 29
    num_ballobs: int = 3
    # one_step: ball(3) + ang_vel(3) + gravity(3) + dof_pos(29) + dof_vel(29) + actions(29) = 96
    num_one_step_observations: int = 96
    # privileged: one_step(96) + lin_vel(3) + region(1) + end_target(3) + ball_vel(3)
    #             + hand_r(3) + hand_l(3) + dist(1) = 113
    num_privileged_obs: int = 113

    # DirectRLEnvCfg required fields
    decimation: int = 4
    action_space: int = 29
    observation_space: int = 960   # num_actor_history * num_one_step_observations
    state_space: int = 113         # privileged obs for critic
    episode_length_s: float = 3.0

    # Simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 200.0,   # 200 Hz physics
        render_interval=4,  # render every decimation step
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
        ),
        physx=sim_utils.PhysxCfg(
            solver_type=1,
            max_position_iteration_count=8,
            max_velocity_iteration_count=0,
            bounce_threshold_velocity=0.5,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**22,
        ),
    )

    # Terrain
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
        ),
        debug_vis=False,
    )

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1020,
        env_spacing=5.0,
        replicate_physics=True,
    )

    # Assets
    robot: ArticulationCfg = G1_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    ball: RigidObjectCfg = BALL_CFG.replace(prim_path="/World/envs/env_.*/Ball")

    # Contact sensor (all robot bodies)
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        update_period=0.0,
        history_length=3,
        debug_vis=False,
    )

    # Don't block on texture streaming — allows non-headless mode to start immediately
    wait_for_textures: bool = False

    # Control
    action_scale: float = 0.25
    ball_gravity: bool = True
    play: bool = False

    # PD gains (used in manual torque computation)
    stiffness: dict = {
        "hip_yaw": 150.0, "hip_roll": 150.0, "hip_pitch": 150.0,
        "knee": 300.0, "ankle": 40.0,
        "shoulder": 150.0, "elbow": 150.0, "waist": 150.0, "wrist": 20.0,
    }
    damping: dict = {
        "hip_yaw": 2.0, "hip_roll": 2.0, "hip_pitch": 2.0,
        "knee": 4.0, "ankle": 2.0,
        "shoulder": 2.0, "elbow": 2.0, "waist": 2.0, "wrist": 0.5,
    }

    # Body names for index lookup
    foot_name: str = "ankle_pitch"
    contact_foot_names: str = "ankle_roll_link"
    hand_name: str = "hand"
    knee_names: list = ["left_knee_link", "right_knee_link"]
    upper_body_link: str = "pelvis"
    torso_link: str = "torso_link"
    imu_link: str = "imu_link"
    keyframe_name: str = "keyframe"
    waist_joints: list = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
    ankle_joints: list = ["left_ankle_pitch_joint", "right_ankle_pitch_joint"]
    penalize_contacts_on: list = [
        "hip", "knee", "torso", "shoulder", "elbow", "pelvis", "hand", "head"
    ]

    # Joint groups
    curriculum_joints: list = [
        "waist_yaw_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    ]
    left_leg_joints: list = [
        "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    ]
    right_leg_joints: list = [
        "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    ]
    left_hip_joints: list = ["left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint"]
    right_hip_joints: list = ["right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint"]
    left_arm_joints: list = [
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    ]
    right_arm_joints: list = [
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    ]
    elbow_joints: list = ["left_elbow_joint", "right_elbow_joint"]
    wrist_joints: list = [
        "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
        "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    ]

    # Normalization
    clip_observations: float = 100.0
    clip_actions: float = 100.0
    obs_scales_ang_vel: float = 0.25
    obs_scales_dof_pos: float = 1.0
    obs_scales_dof_vel: float = 0.05
    obs_scales_lin_vel: float = 2.0
    obs_scales_ball_vel: float = 0.2

    # Noise
    add_noise: bool = True
    noise_level: float = 1.0
    noise_ball: float = 0.08
    noise_dof_pos: float = 0.01
    noise_dof_vel: float = 1.5
    noise_ang_vel: float = 0.2
    noise_gravity: float = 0.05

    # Reward scales
    rew_eereach: float = 10.0
    rew_success: float = 5.0
    rew_stopball: float = 100.0
    rew_stayonline: float = -2.0
    rew_noretreat: float = -2.0
    rew_successland: float = 4.0
    rew_feetorientaion: float = 3.0
    rew_penalize_sharpcontact: float = -100.0
    rew_penalize_kneeheight: float = -100.0
    rew_feet_slippage: float = 3.0
    rew_postorientation: float = 3.0
    rew_postangvel: float = 3.0
    rew_postupperdofpos: float = 1.0
    rew_postwaistdofpos: float = 1.0
    rew_postlinvel: float = 1.0
    rew_ang_vel_xy: float = -0.1
    rew_dof_acc: float = -2.5e-7
    rew_smoothness: float = -0.1
    rew_torques: float = -1e-5
    rew_dof_vel: float = -5e-4
    rew_dof_pos_limits: float = -3.0
    rew_dof_vel_limits: float = -2.0
    rew_torque_limits: float = -3.0
    rew_deviation_waist_pitch_joint: float = -0.001

    # Reward thresholds
    catch_th: float = 0.5
    handheight_th: float = 1.0
    reach_th: float = 0.2
    strict_th: float = 0.15
    catch_sigma: float = 5.0
    soft_dof_pos_limit: float = 0.9
    soft_dof_vel_limit: float = 0.9
    soft_torque_limit: float = 0.95
    max_contact_force: float = 1000.0

    # Domain randomization
    randomize_joint_injection: bool = True
    joint_injection_range: list = [-0.01, 0.01]
    randomize_actuation_offset: bool = True
    actuation_offset_range: list = [-0.01, 0.01]
    randomize_payload_mass: bool = True
    payload_mass_range: list = [-5.0, 10.0]
    randomize_com_displacement: bool = True
    com_displacement_range: list = [-0.1, 0.1]
    randomize_link_mass: bool = True
    link_mass_range: list = [0.8, 1.2]
    randomize_friction: bool = True
    friction_range: list = [0.1, 2.0]
    randomize_restitution: bool = True
    restitution_range: list = [0.0, 1.0]
    randomize_kp: bool = True
    kp_range: list = [0.8, 1.2]
    randomize_kd: bool = True
    kd_range: list = [0.8, 1.2]
    randomize_initial_joint_pos: bool = True
    continue_keep: bool = True
    initial_joint_pos_scale: list = [0.5, 1.5]
    initial_joint_pos_offset: list = [-0.1, 0.1]
    push_robots: bool = True
    push_interval_s: float = 15.0
    max_push_vel_xy: float = 1.5
    ball_interval_s: float = 0.5
    max_ball_vel: float = 0.5
    delay: bool = True

    # Command regions (6 regions for goalkeeper task)
    commands_ranges: dict = {
        "ranges_0": {"height": [0.4, 1.2], "width": [0.2, 1.2],
                     "maxh": [0.3, 1.5], "maxw": [0.0, 1.8],
                     "evalh": [0.3, 1.5], "evalw": [0.0, 1.5]},
        "ranges_1": {"height": [0.4, 1.2], "width": [-1.2, -0.2],
                     "maxh": [0.3, 1.5], "maxw": [-1.8, -0.0],
                     "evalh": [0.3, 1.5], "evalw": [-1.5, 0.0]},
        "ranges_2": {"height": [1.2, 1.6], "width": [0, 1.0],
                     "maxh": [1.2, 1.8], "maxw": [0, 1.5],
                     "evalh": [1.2, 1.8], "evalw": [0, 1.5]},
        "ranges_3": {"height": [1.2, 1.6], "width": [-1.0, 0.0],
                     "maxh": [1.2, 1.8], "maxw": [-1.5, 0.0],
                     "evalh": [1.2, 1.8], "evalw": [-1.5, 0.0]},
        "ranges_4": {"height": [0.1, 0.3], "width": [0.2, 1.2],
                     "maxh": [0.1, 0.3], "maxw": [0.0, 1.8],
                     "evalh": [0.1, 0.3], "evalw": [0.0, 1.5]},
        "ranges_5": {"height": [0.1, 0.3], "width": [-1.2, -0.2],
                     "maxh": [0.1, 0.3], "maxw": [-1.8, -0.0],
                     "evalh": [0.1, 0.3], "evalw": [-1.5, -0.0]},
    }

    # AMP dataset
    dataset_folder: str = _DATASETS_DIR
    dataset_joint_mapping: str = os.path.join(_DATASETS_DIR, "joint_id.txt")
    dataset_frame_rate: int = 30
    dataset_min_time: float = 0.1
    amp_obs_type: str = "dof"
    amp_num_obs: int = 58  # 29 robot joints * 2 frames — matches original g1_29_config num_obs=29*2
    amp_coef: float = 0.4
    amp_num_steps: int = 2

    # AMP discriminator training
    amp_enabled: bool = True          # Enable AMP motion priors
    disc_hidden_dims: list = [512, 256]
    disc_lr: float = 1e-4
    disc_grad_penalty: float = 5.0
    disc_update_interval: int = 16    # Train discriminators every N policy steps
    disc_buffer_size: int = 2048      # Max policy obs for discriminator training batch

    # Initial pose (from g1_29_config)
    init_pos: list = [
        -0.34930936, -0.03763366, -0.22198406,  0.93093884, -0.50943524, -0.08583859,
         0.13749947, -0.44516975, -0.06791031,  0.11570476, -0.17351833,  0.34241587,
        -0.00869134,  0.00670955,  0.01293622,  0.00395479,  0.49003497, -0.00168978,
         1.2062242,  -0.01060604,  0.00490874, -0.00869134,  0.00319979, -0.4975251,
        -0.00450607,  1.20307243,  0.00536893,  0.0053766,   0.00324437,
    ]
