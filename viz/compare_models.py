"""
4-model comparison visualization for 3D forest instance segmentation.

Layout  (2 rows × 4 cols):
    Top-down (XY)  |  GT  |  OneFormer  |  ForestFormer3D  |  ForestMamba (Ours)
    Side view (XZ) |  GT  |  OneFormer  |  ForestFormer3D  |  ForestMamba (Ours)

Full point cloud is rendered (no sub-sampling).
PLY files are loaded in parallel via ThreadPoolExecutor; color assignment is
fully vectorized — no Python per-point loop.

Usage:
    python viz/compare_models.py
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # only used for colormaps + single-scene main()
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import FigureCanvasPdf
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import torch
    _CUDA = torch.cuda.is_available()
except ImportError:
    _CUDA = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "..")

SCENES = {
    "GT": None,   # read from any file; we take instance_gt
    "OneFormer": os.path.join(
        ROOT,
        "work_dirs/2_Oneformer3d/epoch_2100/"
        "NIBIO_NIBIO_plot_17_annotated_test_final_results.ply",
    ),
    "ForestFormer3D": os.path.join(
        ROOT,
        "work_dirs/1_ForestFormer3D_Retrain/new_split/epoch_3000/"
        "NIBIO_NIBIO_plot_17_annotated_test.ply",
    ),
    "ForestMamba\n(Ours)": os.path.join(
        ROOT,
        "work_dirs/forestmamba_chm_radius16_qp300_2many_v6/epoch3000/"
        "NIBIO_NIBIO_plot_17_annotated_test.ply",
    ),
}
# Use ForestMamba file as source for GT and shared xyz
GT_SOURCE = SCENES["ForestMamba\n(Ours)"]

SAVE_DIR = os.path.join(BASE, "save")
OUT_NAME = "NIBIO_plot_17_model_comparison.pdf"

# ---------------------------------------------------------------------------
# Render settings
# ---------------------------------------------------------------------------
PT_SIZE_TOP  = 0.15   # top-down view — denser projection, smaller dots
PT_SIZE_SIDE = 0.12
ALPHA        = 0.50
DPI          = 180
FIG_W        = 28     # inches
FIG_H        = 14



# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------
def make_palette(n: int, seed: int = 42) -> np.ndarray:
    """(n, 4) RGBA, perceptually distinct, shuffled so adjacent IDs differ."""
    if n <= 20:
        base = plt.cm.tab20(np.linspace(0, 1, 20))
    elif n <= 40:
        base = np.vstack([plt.cm.tab20(np.linspace(0, 1, 20)),
                          plt.cm.tab20b(np.linspace(0, 1, 20))])
    else:
        base = plt.cm.hsv(np.linspace(0, 1, n + 1, endpoint=False))
    rng = np.random.default_rng(seed)
    base = base[rng.permutation(len(base))]
    return base[:n]


NOISE_COLOR = np.array([0.72, 0.72, 0.72, 0.30], dtype=np.float32)


def instance_colors(ids: np.ndarray, noise_ids=(0, -1)):
    """
    Map per-point instance IDs → (N,4) RGBA  —  fully vectorized, no Python loop.
    noise_ids are rendered grey; all others get distinct colors.
    """
    noise_set = set(noise_ids)
    unique_fg = sorted(i for i in np.unique(ids) if i not in noise_set)
    palette   = make_palette(len(unique_fg))           # (K, 4)
    id2col    = {uid: palette[k] for k, uid in enumerate(unique_fg)}

    # Build a lookup array indexed by shifted ID value
    min_id   = int(ids.min())
    max_id   = int(ids.max())
    lut_size = max_id - min_id + 1
    lut      = np.tile(NOISE_COLOR, (lut_size, 1))     # default → grey
    for uid, col in id2col.items():
        lut[uid - min_id] = col

    out = lut[(ids.astype(np.int64) - min_id)]         # vectorized lookup
    return out.astype(np.float32), id2col


# ---------------------------------------------------------------------------
# Fast PLY loader  (pandas CSV — 15× faster than plyfile for ASCII PLY)
# ---------------------------------------------------------------------------
_PLY_COLS  = ['x', 'y', 'z', 'sem_pred', 'inst_pred', 'score', 'sem_gt', 'inst_gt']
_PLY_DTYPE = {'x': 'f4', 'y': 'f4', 'z': 'f4',
              'sem_pred': 'i4', 'inst_pred': 'i4', 'score': 'f4',
              'sem_gt': 'i4', 'inst_gt': 'i4'}


def _header_size(path: str) -> int:
    with open(path) as f:
        for i, line in enumerate(f, 1):
            if line.strip() == 'end_header':
                return i
    raise ValueError(f"No end_header found in {path}")


def _load_ply_fast(path: str, cols_only: list = None) -> dict:
    """Load ASCII PLY via pandas.  cols_only limits which columns are parsed."""
    n_skip = _header_size(path)
    kwargs = dict(sep=' ', skiprows=n_skip, header=None,
                  names=_PLY_COLS, engine='c')
    if cols_only:
        kwargs['usecols']  = cols_only
        kwargs['dtype']    = {c: _PLY_DTYPE[c] for c in cols_only}
    else:
        kwargs['dtype'] = _PLY_DTYPE
    df = pd.read_csv(path, **kwargs)
    return {c: df[c].values for c in df.columns}


def load_all_parallel(paths: dict) -> dict:
    """Load 3 PLY files for one scene.

    Strategy:
    - Primary file (first in dict): load all columns (x,y,z,gt,pred,...) — 2.25s
    - Other files: load only inst_pred column — 0.86s each (xyz shared from primary)
    - All 3 loaded in parallel threads for max throughput.
    """
    labels  = list(paths.keys())
    primary = labels[0]

    def _load(label):
        if label == primary:
            return label, _load_ply_fast(paths[label])
        else:
            return label, _load_ply_fast(paths[label], cols_only=['inst_pred'])

    results = {}
    with ThreadPoolExecutor(max_workers=len(paths)) as pool:
        for label, data in pool.map(_load, labels):
            results[label] = data
            print(f"  ✓ loaded {label.replace(chr(10), ' ')}")

    # Share xyz + gt from primary to partial-loaded files
    for label in labels[1:]:
        for key in ('x', 'y', 'z', 'sem_gt', 'inst_gt'):
            results[label][key] = results[primary][key]

    return results


# ---------------------------------------------------------------------------
# GPU rasterizer  (used when CUDA is available — replaces matplotlib scatter)
# ---------------------------------------------------------------------------
# Canvas resolution: figure panel is ~5" wide at DPI=180 → ~900px.
# We render at 2× for crispness then let imshow downsample.
_RASTER_W = 1800
_RASTER_H = 1800
_BG_COLOR  = np.array([255, 255, 255], dtype=np.uint8)   # white background


def _rasterize_gpu(px: np.ndarray, py: np.ndarray,
                   colors_rgba: np.ndarray,
                   xlim: tuple, ylim: tuple,
                   width: int = _RASTER_W,
                   height: int = _RASTER_H,
                   skip_noise: bool = False) -> np.ndarray:
    """
    Paint N points onto an (H, W, 3) uint8 canvas on the GPU.

    Parameters
    ----------
    px, py      : float32 world coordinates (same units as xlim/ylim)
    colors_rgba : (N, 4) float32 RGBA in [0, 1]
    xlim, ylim  : (min, max) axis limits for the viewport
    skip_noise  : if True, ground/noise points (grey) are not painted —
                  the white background shows through instead.

    Returns
    -------
    (H, W, 3) uint8 numpy array ready for ax.imshow()
    """
    device = "cuda"

    # --- normalise to [0, 1] then to pixel indices --------------------------
    x_norm = (px - xlim[0]) / (xlim[1] - xlim[0])
    y_norm = (py - ylim[0]) / (ylim[1] - ylim[0])

    col_idx = (x_norm * (width  - 1)).astype(np.int32)
    row_idx = ((1.0 - y_norm) * (height - 1)).astype(np.int32)   # flip Y

    # clip to canvas
    valid = (col_idx >= 0) & (col_idx < width) & (row_idx >= 0) & (row_idx < height)
    col_idx = col_idx[valid]
    row_idx = row_idx[valid]
    rgb     = (colors_rgba[valid, :3] * 255).astype(np.uint8)

    # --- detect noise (grey = nearly-equal R, G, B all > 180) --------------
    is_noise = (rgb[:, 0] > 180) & (rgb[:, 1] > 180) & (rgb[:, 2] > 180) & \
               (np.abs(rgb[:, 0].astype(np.int16) - rgb[:, 1]) < 15)

    if skip_noise:
        fg = ~is_noise
        col_idx = col_idx[fg]
        row_idx = row_idx[fg]
        rgb     = rgb[fg]
    else:
        # Paint noise first so foreground instances overwrite it
        order   = np.argsort(is_noise.astype(np.int8))
        col_idx = col_idx[order]
        row_idx = row_idx[order]
        rgb     = rgb[order]

    # --- GPU canvas ---------------------------------------------------------
    canvas = torch.full((height, width, 3), fill_value=0,
                        dtype=torch.uint8, device=device)
    canvas[:] = torch.tensor(_BG_COLOR, dtype=torch.uint8, device=device)

    flat_idx = torch.from_numpy((row_idx * width + col_idx).astype(np.int64)).to(device)
    rgb_t    = torch.from_numpy(rgb).to(device)                 # (N, 3) uint8

    canvas_flat = canvas.view(-1, 3)
    canvas_flat.index_put_((flat_idx,), rgb_t, accumulate=False)

    return canvas.cpu().numpy()


def _ax_imshow(ax, img: np.ndarray, xlim, ylim,
               show_xlabel: bool, show_ylabel: bool,
               xlabel: str, ylabel: str, aspect="auto"):
    """Display a pre-rasterized image on ax with correct axis labels."""
    # origin="upper": row 0 of the array is at the top (high Y/Z), which matches
    # the rasterizer that places high-Y values at row 0 via (1 - y_norm).
    ax.imshow(img, extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
              origin="upper", aspect=aspect, interpolation="bilinear")
    ax.tick_params(labelsize=6.5)
    ax.grid(False)
    if show_xlabel:
        ax.set_xlabel(xlabel, fontsize=8, labelpad=3)
    else:
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)
    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=8, labelpad=3)
    else:
        ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("0.6")


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def _scatter(ax, px, py, colors, pt_size,
             show_xlabel=False, show_ylabel=False,
             xlabel="X (m)", ylabel="Y (m)",
             xlim=None, ylim=None):
    ax.scatter(px, py, c=colors, s=pt_size, alpha=ALPHA,
               linewidths=0, rasterized=True)
    ax.tick_params(labelsize=6.5)
    ax.grid(False)
    # Only show axis labels on edge panels to avoid overlap
    if show_xlabel:
        ax.set_xlabel(xlabel, fontsize=8, labelpad=3)
    else:
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)
    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=8, labelpad=3)
    else:
        ax.set_ylabel("")
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    # Minimal spines
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("0.6")


def _precompute_pixels(px: np.ndarray, py: np.ndarray,
                       xlim: tuple, ylim: tuple,
                       width: int, height: int):
    """Compute pixel indices once; reuse for all 4 panels sharing the same viewport.

    Returns (flat_idx, valid_mask) as int64/bool numpy arrays.
    """
    x_norm  = (px - xlim[0]) / (xlim[1] - xlim[0])
    y_norm  = (py - ylim[0]) / (ylim[1] - ylim[0])
    col_idx = (x_norm * (width  - 1)).astype(np.int32)
    row_idx = ((1.0 - y_norm) * (height - 1)).astype(np.int32)
    valid   = (col_idx >= 0) & (col_idx < width) & \
              (row_idx >= 0) & (row_idx < height)
    flat    = (row_idx[valid] * width + col_idx[valid]).astype(np.int64)
    return flat, valid


def _rasterize_precomp(flat_idx: np.ndarray, valid_mask: np.ndarray,
                       colors_rgba: np.ndarray,
                       height: int, width: int,
                       skip_noise: bool = False) -> np.ndarray:
    """Paint using precomputed pixel indices (avoids coord re-normalisation)."""
    rgb = (colors_rgba[valid_mask, :3] * 255).astype(np.uint8)

    is_noise = (rgb[:, 0] > 180) & (rgb[:, 1] > 180) & (rgb[:, 2] > 180) & \
               (np.abs(rgb[:, 0].astype(np.int16) - rgb[:, 1]) < 15)

    if skip_noise:
        fg       = ~is_noise
        flat_use = flat_idx[fg]
        rgb_use  = rgb[fg]
    else:
        order    = np.argsort(is_noise.astype(np.int8))   # noise first → overwritten
        flat_use = flat_idx[order]
        rgb_use  = rgb[order]

    device = "cuda"
    canvas = torch.full((height * width, 3), fill_value=0,
                        dtype=torch.uint8, device=device)
    canvas[:] = torch.tensor(_BG_COLOR, dtype=torch.uint8, device=device)
    flat_t = torch.from_numpy(flat_use).to(device)
    rgb_t  = torch.from_numpy(rgb_use).to(device)
    canvas.index_put_((flat_t,), rgb_t, accumulate=False)
    return canvas.view(height, width, 3).cpu().numpy()


def draw_topdown(ax, x, y, colors,
                 show_xlabel=False, show_ylabel=False,
                 xlim=None, ylim=None, skip_noise=True,
                 _precomp=None):
    """_precomp: (flat_idx, valid_mask) from _precompute_pixels — avoids recomputing coords."""
    if _CUDA:
        if _precomp is not None:
            flat_idx, valid = _precomp
            img = _rasterize_precomp(flat_idx, valid, colors,
                                     _RASTER_H, _RASTER_W, skip_noise)
        else:
            img = _rasterize_gpu(x, y, colors, xlim, ylim, skip_noise=skip_noise)
        _ax_imshow(ax, img, xlim, ylim,
                   show_xlabel, show_ylabel, "X (m)", "Y (m)", aspect="equal")
    else:
        ax.set_aspect("equal")
        _scatter(ax, x, y, colors, PT_SIZE_TOP,
                 show_xlabel=show_xlabel, show_ylabel=show_ylabel,
                 xlabel="X (m)", ylabel="Y (m)",
                 xlim=xlim, ylim=ylim)


def draw_side(ax, x, z, colors,
              show_xlabel=False, show_ylabel=False,
              xlim=None, zlim=None,
              _precomp=None):
    """_precomp: (flat_idx, valid_mask) from _precompute_pixels — avoids recomputing coords."""
    if _CUDA:
        h = _RASTER_H // 2
        if _precomp is not None:
            flat_idx, valid = _precomp
            img = _rasterize_precomp(flat_idx, valid, colors,
                                     h, _RASTER_W, skip_noise=False)
        else:
            img = _rasterize_gpu(x, z, colors, xlim, zlim,
                                 width=_RASTER_W, height=h)
        _ax_imshow(ax, img, xlim, zlim,
                   show_xlabel, show_ylabel, "X (m)", "Height (m)")
    else:
        _scatter(ax, x, z, colors, PT_SIZE_SIDE,
                 show_xlabel=show_xlabel, show_ylabel=show_ylabel,
                 xlabel="X (m)", ylabel="Height (m)",
                 xlim=xlim, ylim=zlim)


def legend_for(ax, id2col: dict, n_trees: int, model_name: str,
               n_unlabeled: int = 0):
    ax.axis("off")
    patches = [
        mpatches.Patch(color=col, label=f"Tree {uid}")
        for uid, col in sorted(id2col.items())
    ]
    patches.append(mpatches.Patch(color=NOISE_COLOR, label="Ground / noise"))
    ax.legend(
        handles=patches, ncol=max(1, min(7, (n_trees + 1) // 3 + 1)),
        loc="center", fontsize=6.5,
        framealpha=0.85, edgecolor="0.75",
        handlelength=1.0, handleheight=0.85,
        columnspacing=0.55, handletextpad=0.35,
        title=f"{model_name}  ·  {n_trees} trees",
        title_fontsize=8,
    )


# ---------------------------------------------------------------------------
# Core rendering  (importable by batch runner)
# ---------------------------------------------------------------------------
COL_LABELS  = ["GT", "OneFormer", "ForestFormer3D", "ForestMamba\n(Ours)"]
COL_DISPLAY = ["Ground Truth", "OneFormer", "ForestFormer3D", "ForestMamba (Ours)"]


def render_scene(scene_id: str, model_paths: dict, save_dir: str,
                 verbose: bool = True,
                 status_fn=None) -> str:
    """
    Render one scene comparison PDF.

    Parameters
    ----------
    scene_id    : bare scene name (no .ply extension)
    model_paths : {"OneFormer": path, "ForestFormer3D": path, "ForestMamba\\n(Ours)": path}
    save_dir    : directory where the PDF is written
    verbose     : print progress lines to stdout
    status_fn   : optional callable(scene_id, stage, detail) for live progress.
                  stage is one of: "loading", "coloring", "rendering", "saving", "done"

    Returns
    -------
    Path to the saved PDF.
    """
    def _status(stage: str, detail: str = ""):
        if status_fn:
            status_fn(scene_id, stage, detail)
        if verbose:
            msg = f"[{scene_id}] {stage}"
            if detail:
                msg += f" — {detail}"
            print(msg)

    t0 = time.perf_counter()

    _status("loading", f"{len(model_paths)} files")
    loaded = load_all_parallel(model_paths)

    first_data = next(iter(loaded.values()))
    x = first_data["x"].astype(np.float32)
    y = first_data["y"].astype(np.float32)
    z = first_data["z"].astype(np.float32)

    xlim = (float(x.min()) - 0.5, float(x.max()) + 0.5)
    ylim = (float(y.min()) - 0.5, float(y.max()) + 0.5)
    zlim = (float(z.min()) - 0.3, float(z.max()) + 0.5)

    _status("coloring", f"{len(x):,} points")
    col_gt, id2col_gt = instance_colors(first_data["inst_gt"], noise_ids=(0,))
    n_unlabeled = 0

    model_colors  = {}
    model_id2col  = {}
    model_ncounts = {}
    for label in COL_LABELS:
        if label == "GT":
            continue
        d = loaded[label]
        col, id2c = instance_colors(d["inst_pred"], noise_ids=(-1,))
        model_colors[label]  = col
        model_id2col[label]  = id2c
        model_ncounts[label] = len(id2c)

    n_gt_trees = len(id2col_gt)

    # -- figure ---------------------------------------------------------------
    # Use Figure() directly (not plt.figure()) so multiple threads can create
    # figures concurrently without contending on pyplot's global lock.
    _status("rendering", f"{n_gt_trees} GT trees")
    row_h   = [0.40, 0.32, 0.13]
    row_bot = [0.54, 0.19, 0.03]
    col_w   = 0.222
    col_gap = 0.012
    left0   = 0.045

    fig = Figure(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"3D Forest Instance Segmentation — {scene_id}   "
        f"({len(x):,} points · {n_gt_trees} GT trees)",
        fontsize=14, fontweight="bold", y=0.995,
    )

    axes = {}
    for ci in range(len(COL_LABELS)):
        left = left0 + ci * (col_w + col_gap)
        for ri, (bot, h) in enumerate(zip(row_bot, row_h)):
            axes[(ri, ci)] = fig.add_axes([left, bot, col_w, h])

    # column headers
    for ci, (label, display) in enumerate(zip(COL_LABELS, COL_DISPLAY)):
        n_trees = n_gt_trees if label == "GT" else model_ncounts[label]
        axes[(0, ci)].set_title(f"{display}\n({n_trees} trees)",
                                fontsize=11, fontweight="bold", pad=5)

    # row labels
    for ri, row_label in enumerate(["Top-down (XY)", "Side view (XZ)"]):
        axes[(ri, 0)].annotate(
            row_label, xy=(-0.22, 0.5), xycoords="axes fraction",
            fontsize=9, fontweight="bold", color="0.35",
            rotation=90, va="center", ha="center",
        )

    # Precompute pixel coords once per viewport — reused for all 4 panels per row
    td_precomp = _precompute_pixels(x, y, xlim, ylim, _RASTER_W, _RASTER_H) if _CUDA else None
    sv_precomp = _precompute_pixels(x, z, xlim, zlim, _RASTER_W, _RASTER_H // 2) if _CUDA else None

    # row 0 — top-down  (GT keeps noise/ground; model cols hide it)
    for ci, label in enumerate(COL_LABELS):
        cols = col_gt if label == "GT" else model_colors[label]
        draw_topdown(axes[(0, ci)], x, y, cols,
                     show_xlabel=False, show_ylabel=(ci == 0),
                     xlim=xlim, ylim=ylim,
                     skip_noise=(label != "GT"),
                     _precomp=td_precomp)

    # row 1 — side view
    for ci, label in enumerate(COL_LABELS):
        cols = col_gt if label == "GT" else model_colors[label]
        draw_side(axes[(1, ci)], x, z, cols,
                  show_xlabel=True, show_ylabel=(ci == 0),
                  xlim=xlim, zlim=zlim,
                  _precomp=sv_precomp)

    # row 2 — legends
    for ci, (label, display) in enumerate(zip(COL_LABELS, COL_DISPLAY)):
        ax = axes[(2, ci)]
        if label == "GT":
            legend_for(ax, id2col_gt, n_gt_trees, display,
                       n_unlabeled=n_unlabeled)
        else:
            legend_for(ax, model_id2col[label], model_ncounts[label], display)

    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, f"{scene_id}_comparison.pdf")
    _status("saving", os.path.basename(out))
    canvas = FigureCanvasPdf(fig)
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())

    elapsed = time.perf_counter() - t0
    size_mb = os.path.getsize(out) / 1e6
    _status("done", f"{elapsed:.0f}s  {size_mb:.1f} MB")
    return out


# ---------------------------------------------------------------------------
# Main — single scene (NIBIO plot 17)
# ---------------------------------------------------------------------------
def _nibio17_paths(root: str) -> dict:
    sid = "NIBIO_NIBIO_plot_17_annotated_test"
    return {
        "OneFormer": os.path.join(
            root, "work_dirs/2_Oneformer3d/epoch_2100",
            f"{sid}_final_results.ply"),
        "ForestFormer3D": os.path.join(
            root, "work_dirs/1_ForestFormer3D_Retrain/new_split/epoch_3000",
            f"{sid}.ply"),
        "ForestMamba\n(Ours)": os.path.join(
            root, "work_dirs/forestmamba_chm_radius16_qp300_2many_v6/epoch3000",
            f"{sid}.ply"),
    }


def main():
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    render_scene(
        scene_id   = "NIBIO_NIBIO_plot_17_annotated_test",
        model_paths= _nibio17_paths(root),
        save_dir   = SAVE_DIR,
    )


if __name__ == "__main__":
    main()
