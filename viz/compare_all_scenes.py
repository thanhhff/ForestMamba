"""
Batch comparison visualization for all 33 common scenes.

For each scene, produces a 4-column PDF (GT | OneFormer | ForestFormer3D | ForestMamba)
and saves it to viz/save/.  Already-existing PDFs are skipped.

Scene-level parallelism: each scene is processed in its own thread.
Within a scene the 3 PLY files are also loaded in parallel (inner threads).

Usage:
    python viz/compare_all_scenes.py [--workers N]

    N defaults to 3 — tune down if RAM is tight (BlueCat scenes are ~3.3 GB each).
"""

import os
import sys
import argparse
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Make compare_models importable regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_models import render_scene   # noqa: E402

# ---------------------------------------------------------------------------
# Model directories
# ---------------------------------------------------------------------------
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

MODEL_DIRS = {
    # "OneFormer":          os.path.join(ROOT, "work_dirs/2_Oneformer3d/epoch_2100"),
    "OneFormer":          os.path.join(ROOT, "work_dirs/2_Oneformer3d/oneformer3d_radius16_qp300/epoch_3000"),
    "ForestFormer3D":     os.path.join(ROOT, "work_dirs/1_ForestFormer3D_Retrain/new_split/epoch_3000"),
    "ForestMamba\n(Ours)":os.path.join(ROOT, "work_dirs/forestmamba_chm_radius16_qp300_2many_v6/epoch3000"),
}

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "save")

# ---------------------------------------------------------------------------
# Scene list — edit this to control which scenes are rendered.
# Set to None to auto-discover all common scenes across models.
# ---------------------------------------------------------------------------
SCENES = [
    "CULS_CULS_plot_2_annotated_test",
    "NIBIO2_NIBIO2_plot10_annotated_test",
    "NIBIO2_NIBIO2_plot15_annotated_test",
    "NIBIO2_NIBIO2_plot1_annotated_test",
    "NIBIO2_NIBIO2_plot27_annotated_test",
    "NIBIO2_NIBIO2_plot32_annotated_test",
    "NIBIO2_NIBIO2_plot34_annotated_test",
    "NIBIO2_NIBIO2_plot35_annotated_test",
    "NIBIO2_NIBIO2_plot3_annotated_test",
    "NIBIO2_NIBIO2_plot48_annotated_test",
    "NIBIO2_NIBIO2_plot49_annotated_test",
    "NIBIO2_NIBIO2_plot52_annotated_test",
    "NIBIO2_NIBIO2_plot53_annotated_test",
    "NIBIO2_NIBIO2_plot58_annotated_test",
    "NIBIO2_NIBIO2_plot60_annotated_test",
    "NIBIO2_NIBIO2_plot6_annotated_test",
    "NIBIO_MLS_MLS_burumPlot_2_panoptic_test",
    "NIBIO_NIBIO_plot_17_annotated_test",
    "NIBIO_NIBIO_plot_18_annotated_test",
    "NIBIO_NIBIO_plot_1_annotated_test",
    "NIBIO_NIBIO_plot_22_annotated_test",
    "NIBIO_NIBIO_plot_23_annotated_test",
    "NIBIO_NIBIO_plot_5_annotated_test",
    "RMIT_RMIT_test_test",
    "SCION_SCION_plot_31_annotated_test",
    "SCION_SCION_plot_61_annotated_test",
    "TUWIEN_TUWIEN_test_test",
    "Yuchen_2023_dls_merged_230209_panoptic_test",
    "BlueCat_RN_merged_trees_test_subset000",
    "BlueCat_RN_merged_trees_test_subset001",
    "BlueCat_RN_merged_trees_test_subset002",
    "BlueCat_RN_merged_trees_test_subset003",
    "BlueCat_RN_merged_trees_test_subset004",
]


def find_ply(directory: str, scene_id: str) -> str | None:
    """Return the PLY path for a scene, trying both naming conventions."""
    for suffix in (f"{scene_id}.ply", f"{scene_id}_final_results.ply"):
        p = os.path.join(directory, suffix)
        if os.path.exists(p):
            return p
    return None


def discover_scenes() -> list[str]:
    """Return sorted list of scene IDs present in all three model directories."""
    sets = []
    for d in MODEL_DIRS.values():
        scenes = set()
        for fname in os.listdir(d):
            if not fname.endswith(".ply"):
                continue
            sid = fname.replace("_final_results.ply", "").replace(".ply", "")
            scenes.add(sid)
        sets.append(scenes)
    common = sorted(sets[0] & sets[1] & sets[2])
    return common


# ---------------------------------------------------------------------------
# Shared per-scene stage tracking (thread-safe)
# ---------------------------------------------------------------------------
_lock        = threading.Lock()
_scene_times : list[float] = []   # elapsed seconds for completed renders

# Stage display order and short labels
_STAGE_LABEL = {
    "loading":   "load ",
    "coloring":  "color",
    "rendering": "rend ",
    "saving":    "save ",
    "done":      "done ",
}
# Live status per in-flight scene: {scene_id: (stage, detail, t_start)}
_status_table: dict[str, tuple[str, str, float]] = {}
# Reference to the tqdm bar so the callback can refresh it
_bar = None


def _stage_callback(scene_id: str, stage: str, detail: str):
    """Called from worker threads at each stage transition."""
    with _lock:
        if stage == "done":
            entry = _status_table.pop(scene_id, None)
            if entry:
                _scene_times.append(time.perf_counter() - entry[2])
        else:
            t_start = _status_table[scene_id][2] if scene_id in _status_table else time.perf_counter()
            _status_table[scene_id] = (stage, detail, t_start)

        if _bar is not None:
            # Build compact in-flight summary for the postfix
            lines = []
            for sid, (stg, det, t0) in _status_table.items():
                age    = time.perf_counter() - t0
                label  = _STAGE_LABEL.get(stg, stg[:5])
                short  = sid.split("_")[-2] if "_" in sid else sid[:12]
                lines.append(f"{short}:{label}({age:.0f}s)")
            _bar.set_postfix_str("  |  ".join(lines) if lines else "")


def process_scene(scene_id: str) -> tuple[str, str]:
    """Load, render, and save one scene. Returns (status_tag, message)."""
    out_path = os.path.join(SAVE_DIR, f"{scene_id}_comparison.pdf")
    if os.path.exists(out_path):
        return "SKIP", scene_id

    paths = {}
    for label, d in MODEL_DIRS.items():
        p = find_ply(d, scene_id)
        if p is None:
            return "MISS", f"{scene_id}  — missing in {label}"
        paths[label] = p

    if _bar is not None:
        _bar.write(f"  → [START]  {scene_id}")

    try:
        render_scene(scene_id, paths, SAVE_DIR, verbose=False,
                     status_fn=_stage_callback)
        size_mb = os.path.getsize(out_path) / 1e6
        return "DONE", f"{scene_id}  ({size_mb:.1f} MB)"
    except Exception as exc:
        with _lock:
            _status_table.pop(scene_id, None)
        return "ERR", f"{scene_id}  — {exc}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global _bar

    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6,
                        help="Max parallel scenes (default 6; reduce if OOM)")
    args = parser.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)

    scenes = SCENES if SCENES is not None else discover_scenes()
    already_done = sum(
        1 for s in scenes
        if os.path.exists(os.path.join(SAVE_DIR, f"{s}_comparison.pdf"))
    )
    to_run = len(scenes) - already_done

    print(f"Scenes       : {len(scenes)}  |  already done : {already_done}  |  to render : {to_run}")
    print(f"Workers      : {args.workers}")
    print(f"Output dir   : {SAVE_DIR}\n")
    print("Stage key:  load=loading PLYs  color=color assign  rend=matplotlib  save=writing PDF\n")

    tag_icons = {"DONE": "✓", "SKIP": "–", "MISS": "⚠", "ERR": "✗"}

    _bar = tqdm(
        total=len(scenes),
        initial=already_done,
        unit="scene",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_scene, sid): sid for sid in scenes}
        for fut in as_completed(futures):
            tag, msg = fut.result()
            icon = tag_icons.get(tag, "?")
            _bar.update(1)
            _bar.write(f"  {icon} [{tag}]  {msg}")

    _bar.close()
    avg = sum(_scene_times) / len(_scene_times) if _scene_times else 0
    print(f"\nAll done.  Rendered {len(_scene_times)} scenes  |  avg {avg:.0f}s/scene")


if __name__ == "__main__":
    main()
