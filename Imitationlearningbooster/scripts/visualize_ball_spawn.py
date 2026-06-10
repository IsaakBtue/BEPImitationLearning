"""Ball-spawn trajectory visualisation for all 6 AMP motion types.

Shows the full ball trajectory from +X spawn to end target, one subplot per
motion type, so you can verify:
  - lefthand / leftjump / leftstep  → end target at +Y  (green axis, left hand)
  - righthand / rightjump / rightstep → end target at -Y (right hand)

Run from the Imitationlearningbooster directory:
    python scripts/visualize_ball_spawn.py
Output: scripts/ball_spawn_visualization.png
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless – no display needed
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ── Current ball end-target ranges (after 2026-06-10 Y-axis fix) ───────────
# (y_min, y_max, z_min, z_max)  — left=+Y, right=-Y
_BALL_END_RANGES = [
    ( 0.15,  0.65, 0.40, 1.15),  # 0 lefthand  — +Y (green axis = left hand)
    (-0.65, -0.15, 0.40, 1.15),  # 1 righthand — -Y (right hand)
    ( 0.15,  0.65, 0.85, 1.40),  # 2 leftjump  — +Y, high
    (-0.65, -0.15, 0.85, 1.40),  # 3 rightjump — -Y, high
    ( 0.15,  0.65, 0.20, 0.65),  # 4 leftstep  — +Y, low
    (-0.65, -0.15, 0.20, 0.65),  # 5 rightstep — -Y, low
]

MOTION_NAMES  = ["lefthand", "righthand", "leftjump", "rightjump", "leftstep", "rightstep"]
COLORS        = ["royalblue", "tomato", "deepskyblue", "orangered", "limegreen", "darkorange"]
N_SHOTS       = 30          # trajectories per subplot
G             = 9.81
RNG           = np.random.default_rng(0)

# ── Figure ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 11))
fig.patch.set_facecolor("#1a1a2e")

axes = [fig.add_subplot(2, 3, i + 1, projection="3d") for i in range(6)]

for ax, name, (y_min, y_max, z_min, z_max), color in zip(
    axes, MOTION_NAMES, _BALL_END_RANGES, COLORS
):
    ax.set_facecolor("#16213e")

    for _ in range(N_SHOTS):
        x_start = RNG.uniform(3.0, 4.5)
        y_start = RNG.uniform(-0.8, 0.8)
        z_start = RNG.uniform(0.5, 1.4)
        y_end   = RNG.uniform(y_min, y_max)
        z_end   = RNG.uniform(z_min, z_max)
        t_total = RNG.uniform(0.5, 1.0)

        vx = (-x_start - 0.3) / t_total
        vy = (y_end - y_start) / t_total
        vz = ((z_end - z_start) + 0.5 * G * t_total ** 2) / t_total

        t  = np.linspace(0, t_total, 60)
        xs = x_start + vx * t
        ys = y_start + vy * t
        zs = z_start + vz * t - 0.5 * G * t ** 2

        ax.plot(xs, ys, zs, color=color, alpha=0.35, linewidth=0.9)
        ax.scatter(xs[0],  ys[0],  zs[0],  c="white",  s=8,  zorder=6)
        ax.scatter(xs[-1], ys[-1], zs[-1], c="yellow", s=25, marker="x", zorder=7)

    # ── Robot body (approximate) ─────────────────────────────────────────────
    ax.scatter(0, 0, 0.7, c="white",  s=120, marker="^", zorder=10, label="Robot")

    # Left hand at +Y (green), right hand at -Y (red)
    ax.scatter(0.15,  0.45, 1.05, c="#00ff88", s=80, marker="*",
               zorder=11, label="L hand (+Y)")
    ax.scatter(0.15, -0.45, 1.05, c="#ff4466", s=80, marker="*",
               zorder=11, label="R hand (-Y)")

    # +Y axis indicator (green arrow direction)
    ax.quiver(0, 0, 0.7, 0, 0.9, 0, color="#00cc44", linewidth=2,
              arrow_length_ratio=0.15, label="+Y (green axis)")

    # ── End-target zone shading ───────────────────────────────────────────────
    ys_zone = np.linspace(y_min, y_max, 4)
    zs_zone = np.linspace(z_min, z_max, 4)
    YY, ZZ  = np.meshgrid(ys_zone, zs_zone)
    XX      = np.full_like(YY, -0.3)
    ax.plot_surface(XX, YY, ZZ, color=color, alpha=0.25)

    # ── Labels ───────────────────────────────────────────────────────────────
    side = "+Y ← LEFT hand" if y_min > 0 else "-Y ← RIGHT hand"
    ax.set_title(f"{name}\n{side}", color="white", fontsize=9, pad=4)
    ax.set_xlabel("X (approach →robot)", color="#aaaaaa", fontsize=7, labelpad=2)
    ax.set_ylabel("Y  (+Y = left/green)", color="#aaaaaa", fontsize=7, labelpad=2)
    ax.set_zlabel("Z",                    color="#aaaaaa", fontsize=7, labelpad=2)
    ax.tick_params(colors="#888888", labelsize=6)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
    ax.set_xlim(-0.5, 5.0)
    ax.set_ylim(-1.1, 1.1)
    ax.set_zlim(0.0, 1.8)
    ax.view_init(elev=20, azim=-55)

    if name == "lefthand":
        ax.legend(fontsize=6, loc="upper right", facecolor="#222244",
                  labelcolor="white", framealpha=0.7)

# ── Global title ─────────────────────────────────────────────────────────────
fig.suptitle(
    "Ball spawn trajectories — 6 AMP motion types\n"
    "White dot = spawn (+X side)   Yellow × = target   Coloured wall = target zone\n"
    "Green ★ = left hand (+Y)   Red ★ = right hand (−Y)   Green arrow = +Y axis",
    color="white", fontsize=10, y=0.99,
)

out = Path(__file__).parent / "ball_spawn_visualization.png"
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved → {out}")
