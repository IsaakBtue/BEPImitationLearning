"""Fix the lefthand_t1.npz: ankle pitch must compensate for the knee clamping so feet stay level.

convert_booster.py clamped knees to >= 0.5 rad but didn't touch ankles.  Original retargeting
had near-zero knees with ankles ~0, so foot absolute pitch (hip+knee+ankle) was ~0.  After
clamping, the foot is tilted forward by 0.4 rad on average → heel digs into the floor.
"""
from pathlib import Path
import numpy as np
import mujoco

NPZ = Path("my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/motions/data/lefthand_t1.npz")
XML = Path("my_mjlab_project_booster_t1/src/my_mjlab_project_booster_t1/assets/booster_t1/T1_serial_clean.xml")

# Indices in dof_pos (MJCF joint order, 23 joints, freejoint excluded)
L_HIP, L_KNEE, L_ANKLE = 11, 14, 15
R_HIP, R_KNEE, R_ANKLE = 17, 20, 21

# Load
d = dict(np.load(NPZ))
joint_pos = d["joint_pos"].copy().astype(np.float64)
T = joint_pos.shape[0]
fps = 30  # convert_booster.py used data["fps"]; default 30 if missing
dt = 1.0 / fps

# Make feet level: ankle = -(hip + knee), within ankle joint limits [-0.87, +0.35]
new_l_ankle = -(joint_pos[:, L_HIP] + joint_pos[:, L_KNEE])
new_r_ankle = -(joint_pos[:, R_HIP] + joint_pos[:, R_KNEE])
new_l_ankle = np.clip(new_l_ankle, -0.87, 0.35)
new_r_ankle = np.clip(new_r_ankle, -0.87, 0.35)

print(f"Before: L ankle range [{joint_pos[:, L_ANKLE].min():.3f}, {joint_pos[:, L_ANKLE].max():.3f}]")
print(f"After:  L ankle range [{new_l_ankle.min():.3f}, {new_l_ankle.max():.3f}]")
joint_pos[:, L_ANKLE] = new_l_ankle
joint_pos[:, R_ANKLE] = new_r_ankle

# Re-run mj_kinematics with corrected joint angles, using the same root_pos/root_quat as before.
# We also need to re-shift root_pos so foot bottoms sit just above ground.
model = mujoco.MjModel.from_xml_path(str(XML))
mj_data = mujoco.MjData(model)

# First pass: compute foot positions with the existing root_pos to see how far below 0 they go
old_root_pos = d["body_pos_w"][:, 0, :].astype(np.float64).copy()
old_root_quat = d["body_quat_w"][:, 0, :].astype(np.float64).copy()

n_bodies = model.nbody - 1
body_pos_w = np.zeros((T, n_bodies, 3), dtype=np.float32)
body_quat_w = np.zeros((T, n_bodies, 4), dtype=np.float32)

for t in range(T):
    mj_data.qpos[:3] = old_root_pos[t]
    mj_data.qpos[3:7] = old_root_quat[t]
    mj_data.qpos[7:] = joint_pos[t]
    mujoco.mj_kinematics(model, mj_data)
    body_pos_w[t] = mj_data.xpos[1:].astype(np.float32)
    body_quat_w[t] = mj_data.xquat[1:].astype(np.float32)

# Find the lowest visual point across all frames (using foot mesh vertices)
foot_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("left_foot_link", "right_foot_link")]
lowest_visual = 1.0
for t in range(T):
    mj_data.qpos[:3] = old_root_pos[t]
    mj_data.qpos[3:7] = old_root_quat[t]
    mj_data.qpos[7:] = joint_pos[t]
    mujoco.mj_kinematics(model, mj_data)
    for foot_id in foot_ids:
        for gi in range(model.body_geomadr[foot_id], model.body_geomadr[foot_id] + model.body_geomnum[foot_id]):
            if model.geom_type[gi] == 7:  # mesh
                mesh_id = model.geom_dataid[gi]
                vert_adr = model.mesh_vertadr[mesh_id]
                vert_num = model.mesh_vertnum[mesh_id]
                verts = model.mesh_vert[vert_adr:vert_adr+vert_num].copy()
                geom_xpos = mj_data.geom_xpos[gi]
                geom_xmat = mj_data.geom_xmat[gi].reshape(3, 3)
                world_verts = (geom_xmat @ verts.T).T + geom_xpos
                lowest_visual = min(lowest_visual, world_verts[:, 2].min())

print(f"Lowest visual foot point across all frames (with corrected ankles): {lowest_visual:.4f}m")

# Shift root upward so the lowest visual foot point is at 2mm above ground
shift = 0.002 - lowest_visual
print(f"Shifting root z by {shift:+.4f} m so visual foot bottoms are ~2mm above ground")
new_root_pos = old_root_pos.copy()
new_root_pos[:, 2] += shift

# Re-run kinematics with shifted root
for t in range(T):
    mj_data.qpos[:3] = new_root_pos[t]
    mj_data.qpos[3:7] = old_root_quat[t]
    mj_data.qpos[7:] = joint_pos[t]
    mujoco.mj_kinematics(model, mj_data)
    body_pos_w[t] = mj_data.xpos[1:].astype(np.float32)
    body_quat_w[t] = mj_data.xquat[1:].astype(np.float32)

# Recompute velocities via finite differences
def finite_diff(arr, dt):
    vel = np.zeros_like(arr)
    vel[1:-1] = (arr[2:] - arr[:-2]) / (2 * dt)
    vel[0] = (arr[1] - arr[0]) / dt
    vel[-1] = (arr[-1] - arr[-2]) / dt
    return vel

def quat_ang_vel(quats, dt):
    T_ = quats.shape[0]
    av = np.zeros((T_, *quats.shape[1:-1], 3))
    for t in range(T_):
        t0 = max(0, t - 1)
        t1 = min(T_ - 1, t + 1)
        dt_eff = (t1 - t0) * dt
        if dt_eff == 0:
            continue
        q0, q1 = quats[t0], quats[t1]
        w0, x0, y0, z0 = q0[..., 0], q0[..., 1], q0[..., 2], q0[..., 3]
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        dx = w0 * x1 - x0 * w1 - y0 * z1 + z0 * y1
        dy = w0 * y1 + x0 * z1 - y0 * w1 - z0 * x1
        dz = w0 * z1 - x0 * y1 + y0 * x1 - z0 * w1
        av[t] = (2.0 / dt_eff) * np.stack([dx, dy, dz], axis=-1)
    return av

joint_vel = finite_diff(joint_pos.astype(np.float32), dt).astype(np.float32)
body_lin_vel_w = finite_diff(body_pos_w, dt).astype(np.float32)
body_ang_vel_w = quat_ang_vel(body_quat_w.astype(np.float64), dt).astype(np.float32)

# Backup & save
import shutil
backup = NPZ.with_suffix(".npz.bak")
if not backup.exists():
    shutil.copy(NPZ, backup)
    print(f"Backup: {backup}")

np.savez(
    NPZ,
    joint_pos=joint_pos.astype(np.float32),
    joint_vel=joint_vel,
    body_pos_w=body_pos_w,
    body_quat_w=body_quat_w,
    body_lin_vel_w=body_lin_vel_w,
    body_ang_vel_w=body_ang_vel_w,
)
print(f"Saved corrected npz: {NPZ}")
