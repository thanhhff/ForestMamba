"""
3D Instance Segmentation Visualization for ForestFormer3D predictions.

Reads the best prediction scene (NIBIO_NIBIO_plot_17) and produces a multi-panel
PDF showing GT vs Predicted instances from top-down and side views.

Usage:
    python viz/visualize_instance_seg.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from plyfile import PlyData

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PLY_PATH = os.path.join(
    os.path.dirname(__file__),
    "../work_dirs/forestmamba_chm_radius16_qp300_2many_v6/epoch3000/"
    "NIBIO_NIBIO_plot_17_annotated_test.ply",
)
SAVE_DIR = os.path.join(os.path.dirname(__file__), "save")
SUBSAMPLE = 8          # keep 1 in N points for scatter (speed / file size)
PT_SIZE   = 0.4        # scatter point size
ALPHA     = 0.55       # scatter alpha
DPI       = 180


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------
def distinct_colors(n: int) -> np.ndarray:
    """Return (n, 4) RGBA array of perceptually distinct colors."""
    if n <= 20:
        base = plt.cm.tab20(np.linspace(0, 1, 20))
    else:
        base = plt.cm.hsv(np.linspace(0, 1, n, endpoint=False))
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(base))
    return base[idx[:n]]


def assign_colors(instance_ids: np.ndarray, noise_id: int = -1):
    """Return per-point RGBA array; noise/background gets light grey."""
    unique = sorted(set(instance_ids.tolist()) - {noise_id})
    palette = distinct_colors(len(unique))
    id2col  = {uid: palette[i] for i, uid in enumerate(unique)}
    grey    = np.array([0.7, 0.7, 0.7, 0.35])
    colors  = np.array([
        id2col[iid] if iid != noise_id else grey
        for iid in instance_ids
    ])
    return colors, id2col


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_scene(path: str):
    ply = PlyData.read(path)
    v   = ply["vertex"].data
    return {
        "x":            v["x"].astype(np.float32),
        "y":            v["y"].astype(np.float32),
        "z":            v["z"].astype(np.float32),
        "instance_gt":  v["instance_gt"].astype(np.int32),
        "instance_pred": v["instance_pred"].astype(np.int32),
        "semantic_gt":  v["semantic_gt"].astype(np.int32),
        "semantic_pred": v["semantic_pred"].astype(np.int32),
        "score":        v["score"].astype(np.float32),
    }


def subsample(data: dict, every: int):
    idx = np.arange(0, len(data["x"]), every)
    return {k: v[idx] for k, v in data.items()}


# ---------------------------------------------------------------------------
# Panel helpers
# ---------------------------------------------------------------------------
def scatter_topdown(ax, x, y, colors, title, xlabel="X (m)", ylabel="Y (m)"):
    ax.scatter(x, y, c=colors, s=PT_SIZE, alpha=ALPHA, linewidths=0, rasterized=True)
    ax.set_title(title, fontsize=10, pad=6)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2, linewidth=0.4)


def scatter_side(ax, x, z, colors, title, xlabel="X (m)", ylabel="Z / Height (m)"):
    ax.scatter(x, z, c=colors, s=PT_SIZE, alpha=ALPHA, linewidths=0, rasterized=True)
    ax.set_title(title, fontsize=10, pad=6)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2, linewidth=0.4)


def legend_patches(id2col: dict, noise_id, max_items: int = 30):
    patches = []
    for uid, col in sorted(id2col.items()):
        if uid == noise_id:
            continue
        patches.append(mpatches.Patch(color=col, label=f"Tree {uid}"))
        if len(patches) >= max_items:
            patches.append(mpatches.Patch(color="white", label="…"))
            break
    return patches


def score_barplot(ax, data: dict):
    """Per-instance predicted score bar chart."""
    pred_ids = sorted(set(data["instance_pred"].tolist()) - {-1, 0})
    scores   = []
    sizes    = []
    for pid in pred_ids:
        mask = data["instance_pred"] == pid
        scores.append(float(data["score"][mask].mean()))
        sizes.append(int(mask.sum()))

    palette = distinct_colors(len(pred_ids))
    bars = ax.bar(range(len(pred_ids)), scores, color=palette, edgecolor="white", linewidth=0.3)
    ax.set_xticks(range(len(pred_ids)))
    ax.set_xticklabels([str(p) for p in pred_ids], rotation=45, ha="right", fontsize=6)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Confidence score", fontsize=8)
    ax.set_xlabel("Predicted instance ID", fontsize=8)
    ax.set_title("Per-instance confidence scores", fontsize=10, pad=6)
    ax.axhline(0.9, color="red", linewidth=0.8, linestyle="--", alpha=0.7, label="0.9 threshold")
    ax.legend(fontsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", alpha=0.3, linewidth=0.4)

    # annotate point count on each bar
    for i, (bar, sz) in enumerate(zip(bars, sizes)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{sz//1000}k", ha="center", va="bottom", fontsize=5, color="0.3")


def instance_count_comparison(ax, gt_ids, pred_ids):
    """Bar chart comparing GT vs predicted tree counts."""
    n_gt   = len([i for i in set(gt_ids)   if i not in (0, -1)])
    n_pred = len([i for i in set(pred_ids) if i not in (0, -1)])
    bars = ax.bar(["Ground Truth", "Predicted"], [n_gt, n_pred],
                  color=["#4C72B0", "#DD8452"], edgecolor="white", width=0.5)
    for bar, val in zip(bars, [n_gt, n_pred]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(val), ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Tree instance count", fontsize=8)
    ax.set_title("GT vs Predicted\ninstance counts", fontsize=10, pad=6)
    ax.tick_params(labelsize=8)
    ax.set_ylim(0, max(n_gt, n_pred) * 1.25)
    ax.grid(axis="y", alpha=0.3, linewidth=0.4)


def height_profile(ax, data: dict):
    """Violin / strip of per-instance height ranges."""
    pred_ids = sorted(set(data["instance_pred"].tolist()) - {-1, 0})
    palette  = distinct_colors(len(pred_ids))
    for i, pid in enumerate(pred_ids):
        mask  = data["instance_pred"] == pid
        z_min = float(data["z"][mask].min())
        z_max = float(data["z"][mask].max())
        ax.plot([i, i], [z_min, z_max], color=palette[i], linewidth=3.5, solid_capstyle="round")
        ax.scatter([i], [z_max], s=14, color=palette[i], zorder=3)

    ax.set_xticks(range(len(pred_ids)))
    ax.set_xticklabels([str(p) for p in pred_ids], rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("Height range (m)", fontsize=8)
    ax.set_xlabel("Predicted instance ID", fontsize=8)
    ax.set_title("Per-tree height range (predicted)", fontsize=10, pad=6)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", alpha=0.3, linewidth=0.4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"Loading {os.path.basename(PLY_PATH)} ...")
    data_full = load_scene(PLY_PATH)
    data      = subsample(data_full, SUBSAMPLE)

    n_pts  = len(data_full["x"])
    n_pts_s = len(data["x"])
    print(f"  Total points : {n_pts:,}  |  subsampled : {n_pts_s:,}")

    # separate noise from foreground for GT (id==0 is noise/ground)
    gt_noise   = 0
    pred_noise = -1

    col_gt,   id2col_gt   = assign_colors(data["instance_gt"],   noise_id=gt_noise)
    col_pred, id2col_pred = assign_colors(data["instance_pred"], noise_id=pred_noise)

    # -----------------------------------------------------------------------
    # Figure layout
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(22, 26))
    fig.patch.set_facecolor("#F7F7F7")

    scene_name = os.path.basename(PLY_PATH).replace("_test.ply", "")

    fig.suptitle(
        f"3D Forest Instance Segmentation  |  {scene_name}\n"
        f"ForestMamba · epoch 3000  |  {n_pts:,} points  ·  "
        f"{len(id2col_gt)} GT trees  ·  {len(id2col_pred)} predicted trees",
        fontsize=13, fontweight="bold", y=0.995,
    )

    # Row 1: top-down GT | top-down Pred
    ax_td_gt   = fig.add_axes([0.03, 0.72, 0.42, 0.24])
    ax_td_pred = fig.add_axes([0.55, 0.72, 0.42, 0.24])

    # Row 2: side GT | side Pred
    ax_sd_gt   = fig.add_axes([0.03, 0.46, 0.42, 0.22])
    ax_sd_pred = fig.add_axes([0.55, 0.46, 0.42, 0.22])

    # Row 3: score bar | count compare | height profile
    ax_scores  = fig.add_axes([0.03, 0.23, 0.42, 0.18])
    ax_counts  = fig.add_axes([0.55, 0.23, 0.18, 0.18])
    ax_heights = fig.add_axes([0.78, 0.23, 0.20, 0.18])

    # Row 4: legend GT | legend Pred
    ax_leg_gt   = fig.add_axes([0.03, 0.02, 0.42, 0.18])
    ax_leg_pred = fig.add_axes([0.55, 0.02, 0.42, 0.18])

    # -----------------------------------------------------------------------
    # Row 1 — top-down views
    # -----------------------------------------------------------------------
    scatter_topdown(ax_td_gt,   data["x"], data["y"], col_gt,   "Ground Truth — top-down (XY)")
    scatter_topdown(ax_td_pred, data["x"], data["y"], col_pred, "Predicted — top-down (XY)")

    # -----------------------------------------------------------------------
    # Row 2 — side views (XZ)
    # -----------------------------------------------------------------------
    scatter_side(ax_sd_gt,   data["x"], data["z"], col_gt,   "Ground Truth — side view (XZ)")
    scatter_side(ax_sd_pred, data["x"], data["z"], col_pred, "Predicted — side view (XZ)")

    # -----------------------------------------------------------------------
    # Row 3 — stats panels
    # -----------------------------------------------------------------------
    score_barplot(ax_scores, data_full)
    instance_count_comparison(ax_counts, data_full["instance_gt"], data_full["instance_pred"])
    height_profile(ax_heights, data_full)

    # -----------------------------------------------------------------------
    # Row 4 — legends
    # -----------------------------------------------------------------------
    for ax, id2col, noise_id, title in [
        (ax_leg_gt,   id2col_gt,   gt_noise,   "Ground Truth legend"),
        (ax_leg_pred, id2col_pred, pred_noise, "Predicted legend"),
    ]:
        ax.set_title(title, fontsize=9, pad=4, loc="left")
        patches = legend_patches(id2col, noise_id, max_items=40)
        # add noise patch
        patches.append(mpatches.Patch(color=[0.7, 0.7, 0.7, 0.5], label="Ground / noise"))
        ax.legend(handles=patches, ncol=6, loc="center", fontsize=7,
                  framealpha=0.8, edgecolor="0.7",
                  handlelength=1.2, handleheight=0.8, columnspacing=0.8)
        ax.axis("off")

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    out_path = os.path.join(SAVE_DIR, f"{scene_name}_instance_seg.pdf")
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
