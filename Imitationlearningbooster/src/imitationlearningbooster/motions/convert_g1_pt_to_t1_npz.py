"""Convert G1 .pt motion files → T1 .npz for AMP training.

G1 has 21 joints; T1 has 23. The extra 2 are head joints (AAHead_yaw, Head_pitch)
which are set to zero. All other joints are mapped by anatomical correspondence.

T1 joint order (XML/MuJoCo order, 23 joints):
  0  AAHead_yaw         <- 0 (no G1 equivalent)
  1  Head_pitch         <- 0 (no G1 equivalent)
  2  Left_Shoulder_Pitch  <- G1[13]
  3  Left_Shoulder_Roll   <- G1[14]
  4  Left_Elbow_Pitch     <- G1[16]
  5  Left_Elbow_Yaw       <- G1[15] (G1 shoulder_yaw → closest T1 elbow_yaw)
  6  Right_Shoulder_Pitch <- G1[17]
  7  Right_Shoulder_Roll  <- G1[18]
  8  Right_Elbow_Pitch    <- G1[20]
  9  Right_Elbow_Yaw      <- G1[19]
 10  Waist                <- G1[12]
 11  Left_Hip_Pitch       <- G1[0]
 12  Left_Hip_Roll        <- G1[1]
 13  Left_Hip_Yaw         <- G1[2]
 14  Left_Knee_Pitch      <- G1[3]
 15  Left_Ankle_Pitch     <- G1[4]
 16  Left_Ankle_Roll      <- G1[5]
 17  Right_Hip_Pitch      <- G1[6]
 18  Right_Hip_Roll       <- G1[7]
 19  Right_Hip_Yaw        <- G1[8]
 20  Right_Knee_Pitch     <- G1[9]
 21  Right_Ankle_Pitch    <- G1[10]
 22  Right_Ankle_Roll     <- G1[11]
"""
import numpy as np
import torch
from pathlib import Path
from scipy.interpolate import interp1d

_PT_DIR = Path(__file__).parents[4] / "Humanoid-Goalkeeper" / "legged_gym" / "resources" / "datasets" / "goalkeeper"
_OUT = Path(__file__).parent / "data"
_OUT.mkdir(exist_ok=True)

SRC_FPS = 30
TGT_FPS = 50
TGT_FRAMES = 150  # 3.0 s at 50 Hz

# G1 index → T1 index (-1 means zero-fill)
G1_TO_T1 = [
    (0, 11),   # left_hip_pitch
    (1, 12),   # left_hip_roll
    (2, 13),   # left_hip_yaw
    (3, 14),   # left_knee
    (4, 15),   # left_ankle_pitch
    (5, 16),   # left_ankle_roll
    (6, 17),   # right_hip_pitch
    (7, 18),   # right_hip_roll
    (8, 19),   # right_hip_yaw
    (9, 20),   # right_knee
    (10, 21),  # right_ankle_pitch
    (11, 22),  # right_ankle_roll
    (12, 10),  # waist_yaw → Waist
    (13, 2),   # left_shoulder_pitch
    (14, 3),   # left_shoulder_roll
    (15, 5),   # left_shoulder_yaw → Left_Elbow_Yaw (closest)
    (16, 4),   # left_elbow → Left_Elbow_Pitch
    (17, 6),   # right_shoulder_pitch
    (18, 7),   # right_shoulder_roll
    (19, 9),   # right_shoulder_yaw → Right_Elbow_Yaw
    (20, 8),   # right_elbow → Right_Elbow_Pitch
]

MOTIONS = ["lefthand", "righthand", "leftjump", "rightjump", "leftstep", "rightstep"]


def resample(arr: np.ndarray, src_frames: int, tgt_frames: int) -> np.ndarray:
    t_src = np.linspace(0, 1, src_frames)
    t_dst = np.linspace(0, 1, tgt_frames)
    f = interp1d(t_src, arr, axis=0, kind="linear")
    return f(t_dst).astype(np.float32)


def convert_one(name: str) -> None:
    pt_path = _PT_DIR / f"{name}.pt"
    out_path = _OUT / f"{name}_t1.npz"

    if out_path.exists():
        print(f"  {name}_t1.npz already exists, skipping")
        return

    d = torch.load(pt_path, map_location="cpu")
    g1_joint_pos = d["joint_position"].numpy()  # (T, 21)
    src_frames = g1_joint_pos.shape[0]

    g1_resampled = resample(g1_joint_pos, src_frames, TGT_FRAMES)  # (150, 21)

    t1_joint_pos = np.zeros((TGT_FRAMES, 23), dtype=np.float32)
    for g1_idx, t1_idx in G1_TO_T1:
        t1_joint_pos[:, t1_idx] = g1_resampled[:, g1_idx]

    joint_vel = np.gradient(t1_joint_pos, 1.0 / TGT_FPS, axis=0).astype(np.float32)

    np.savez(out_path, joint_pos=t1_joint_pos, joint_vel=joint_vel)
    print(f"  Saved {out_path}  shape={t1_joint_pos.shape}")


if __name__ == "__main__":
    for name in MOTIONS:
        print(f"Converting {name}...")
        convert_one(name)
    print("Done.")
