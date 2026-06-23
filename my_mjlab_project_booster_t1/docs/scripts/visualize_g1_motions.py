"""Replay G1 .pt motion files in a MuJoCo passive viewer.

Uses the upstream G1 URDF (g1_23.urdf) patched in-memory to add the floating
base joint and fix the mesh path — the source file is never modified.

Usage:
  python visualize_g1_motions.py                  # list available motions
  python visualize_g1_motions.py lefthand          # play one motion (loop)
  python visualize_g1_motions.py --all             # play all in sequence
  python visualize_g1_motions.py lefthand --fps 15 # half speed
"""
import argparse
import glob
import os
import time

import mujoco
import mujoco.viewer
import numpy as np
import torch

# ── Paths ─────────────────────────────────────────────────────────────────────
MOTIONS_DIR = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "Humanoid-Goalkeeper", "legged_gym",
    "resources", "datasets", "goalkeeper",
)
URDF_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "Humanoid-Goalkeeper", "legged_gym",
    "resources", "robots", "g1", "urdf", "g1_23.urdf",
)
MESH_DIR = os.path.abspath(os.path.join(
    os.path.dirname(URDF_PATH), "..", "meshes",
))
DEFAULT_FPS = 30  # .pt files captured at 30 Hz

# ── Joint mapping ──────────────────────────────────────────────────────────────
# G1 .pt has 21 joints (joint_id.txt order); URDF g1_23 has 23 (adds 2 wrists).
# qpos layout with freejoint: [root_pos(3), root_quat_wxyz(4), joints(23)]
# PT index → qpos offset (qpos[7 + urdf_joint_index])
PT_TO_QPOS = list(range(7, 7 + 13))   # .pt[0:13] → qpos[7:20]  (legs + waist)
PT_TO_QPOS += list(range(20, 24))      # .pt[13:17] → qpos[20:24] (left arm)
# qpos[24] = left_wrist (zero-filled, not in .pt)
PT_TO_QPOS += list(range(25, 29))      # .pt[17:21] → qpos[25:29] (right arm)
# qpos[29] = right_wrist (zero-filled, not in .pt)


def load_model() -> mujoco.MjModel:
    with open(URDF_PATH) as f:
        xml = f.read()
    # Patch meshdir to absolute path (avoids editing upstream file)
    xml = xml.replace('meshdir="meshes"', f'meshdir="{MESH_DIR}"')
    # Uncomment the floating base joint so root pose is controllable
    xml = xml.replace(
        '  <!-- <link name="world"></link>\n'
        '  <joint name="floating_base_joint" type="floating">\n'
        '    <parent link="world"/>\n'
        '    <child link="pelvis"/>\n'
        '  </joint> -->',
        '  <link name="world"></link>\n'
        '  <joint name="floating_base_joint" type="floating">\n'
        '    <parent link="world"/>\n'
        '    <child link="pelvis"/>\n'
        '  </joint>',
    )

    spec = mujoco.MjSpec.from_string(xml)

    # Skybox
    sky = spec.add_texture()
    sky.name = "skybox"
    sky.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
    sky.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
    sky.rgb1 = [0.4, 0.6, 0.8]   # horizon
    sky.rgb2 = [0.9, 0.95, 1.0]  # zenith
    sky.width = 512
    sky.height = 3072

    # Checkerboard floor texture + material
    floor_tex = spec.add_texture()
    floor_tex.name = "floor_tex"
    floor_tex.type = mujoco.mjtTexture.mjTEXTURE_2D
    floor_tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    floor_tex.rgb1 = [0.25, 0.25, 0.28]
    floor_tex.rgb2 = [0.40, 0.40, 0.43]
    floor_tex.width = 300
    floor_tex.height = 300

    floor_mat = spec.add_material()
    floor_mat.name = "floor_mat"
    floor_mat.texrepeat = [20, 20]
    floor_mat.texuniform = True
    floor_mat.textures[0] = "floor_tex"

    # Ground plane
    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [0, 0, 0.05]
    floor.material = "floor_mat"

    # Main overhead directional light
    light1 = spec.worldbody.add_light()
    light1.name = "main_light"
    light1.pos = [0, 0, 4]
    light1.dir = [0, 0, -1]
    light1.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    light1.diffuse = [0.7, 0.7, 0.7]
    light1.specular = [0.2, 0.2, 0.2]

    # Soft fill light from side
    light2 = spec.worldbody.add_light()
    light2.name = "fill_light"
    light2.pos = [-2, -2, 3]
    light2.dir = [1, 1, -1]
    light2.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    light2.diffuse = [0.3, 0.3, 0.3]
    light2.specular = [0.0, 0.0, 0.0]

    return spec.compile()


def list_motions():
    files = sorted(glob.glob(os.path.join(MOTIONS_DIR, "*.pt")))
    if not files:
        print(f"No .pt files found in {MOTIONS_DIR}")
        return []
    print("Available G1 motions:")
    for f in files:
        d = torch.load(f, map_location="cpu")
        T = d["joint_position"].shape[0]
        print(f"  {os.path.basename(f)[:-3]:<20}  {T} frames  ({T / DEFAULT_FPS:.2f}s @ {DEFAULT_FPS} Hz)")
    return files


def resolve_path(name: str) -> str:
    if os.path.isabs(name) and name.endswith(".pt"):
        return name
    if not name.endswith(".pt"):
        name = name + ".pt"
    candidate = os.path.join(MOTIONS_DIR, name)
    if os.path.isfile(candidate):
        return candidate
    if os.path.isfile(name):
        return name
    raise FileNotFoundError(f"Motion not found: {name!r}\n"
                            f"  Looked in {MOTIONS_DIR}")


def play_motion(viewer, model, data, pt_path: str, fps: float, loop: bool = False):
    d = torch.load(pt_path, map_location="cpu")
    root_pos = d["base_position"].numpy()   # (T, 3)
    root_quat_xyzw = d["base_pose"].numpy()  # (T, 4) Isaac Gym XYZW
    joint_pos = d["joint_position"].numpy()  # (T, 21)
    T = joint_pos.shape[0]
    name = os.path.basename(pt_path)
    dt = 1.0 / fps

    # Isaac Gym XYZW → MuJoCo WXYZ
    root_quat_wxyz = root_quat_xyzw[:, [3, 0, 1, 2]]

    print(f"\nPlaying {name}  ({T} frames @ {fps} Hz = {T/fps:.2f}s)"
          + ("  [looping, Ctrl-C to stop]" if loop else ""))

    while True:
        for frame in range(T):
            if not viewer.is_running():
                return
            t0 = time.perf_counter()

            data.qpos[:3] = root_pos[frame]
            data.qpos[3:7] = root_quat_wxyz[frame]
            data.qpos[7:] = 0.0  # zero-fill wrists
            for pt_idx, qpos_idx in enumerate(PT_TO_QPOS):
                data.qpos[qpos_idx] = joint_pos[frame, pt_idx]

            mujoco.mj_kinematics(model, data)
            viewer.sync()

            elapsed = time.perf_counter() - t0
            sleep = dt - elapsed
            if sleep > 0:
                time.sleep(sleep)

        if not loop:
            break


def main():
    parser = argparse.ArgumentParser(description="Visualize G1 .pt motions in MuJoCo.")
    parser.add_argument("motion", nargs="?",
                        help="Motion name (e.g. lefthand) or path to .pt. "
                             "Omit to list available motions.")
    parser.add_argument("--all", action="store_true",
                        help="Play all motions in sequence.")
    parser.add_argument("--loop", action="store_true",
                        help="Loop the selected motion until Ctrl-C.")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS,
                        help=f"Playback FPS (default: {DEFAULT_FPS}).")
    args = parser.parse_args()

    if not args.motion and not args.all:
        list_motions()
        return

    model = load_model()
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 3.5
        viewer.cam.elevation = -15
        viewer.cam.azimuth = 135

        if args.all:
            files = sorted(glob.glob(os.path.join(MOTIONS_DIR, "*.pt")))
            if not files:
                print("No motions found.")
                return
            try:
                while viewer.is_running():
                    for f in files:
                        if not viewer.is_running():
                            break
                        play_motion(viewer, model, data, f, args.fps, loop=False)
            except KeyboardInterrupt:
                pass
        else:
            pt_path = resolve_path(args.motion)
            try:
                play_motion(viewer, model, data, pt_path, args.fps, loop=args.loop)
            except KeyboardInterrupt:
                pass

        print("\nDone.")
        while viewer.is_running():
            time.sleep(0.05)


if __name__ == "__main__":
    main()
