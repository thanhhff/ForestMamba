"""
Evaluate 3D forest instance segmentation predictions on the Wytham + LAUTx dataset.

Key differences from eval_predictions.py (ForAINetV2):

  1. Binary GT labels only — no wood/leaf sub-class.
     semantic_gt is reconstructed from instance_gt:
         instance_gt == 0  →  non-tree / ground  (label 0 → +1 = 1)
         instance_gt  > 0  →  tree                (label 1 → +1 = 2)

  2. Leaf+Wood=Tree collapse on predictions.
     The model predicts 3 classes (0=ground, 1=wood, 2=leaf); per the dataset
     authors (GitHub issue #24) leaf and wood are both counted as tree for
     evaluation.  Leaf(3 after +1) is collapsed to tree(2) before all metrics.

  3. Semantic evaluation over all points (matching final_eval.py).
     Non-crown points (including mid-canopy) have instance_gt=0 → sem_gt=1
     (background).  IoU_ground will be lower than ideal because the model
     correctly classifies unlabelled canopy as tree — this is the expected
     behaviour per the author's evaluation protocol.
     Instance evaluation uses its own idxc filter (GT-tree OR pred-tree).

  4. Region grouping is adapted to 'Wytham' and 'LAUTx' sites.

  5. Optional unclassified-filtering (--src_dir).
     When the source PLY directory is provided, points with LAZ classification=3
     (unclassified — includes unlabelled mid-canopy crowns) are excluded from
     all metrics.  This gives cleaner IoU numbers since those points have no
     GT label but the model correctly predicts them as tree.
"""

import glob
import os
import re
import sys
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from plyfile import PlyData

# ── class layout ──────────────────────────────────────────────────────────────
# After the +1 offset applied to raw 0-indexed labels:
#   0 = unclassified  (never appears in practice after collapse)
#   1 = ground  (stuff)
#   2 = tree    (thing)
NUM_CLASSES     = 3   # 0=unclassified, 1=ground(stuff), 2=tree(thing)
NUM_CLASSES_sem = 3   # same 3 — no leaf sub-class
ins_classcount   = [2]
stuff_classcount = [1]
sem_classcount   = [1, 2]
stuff_classes    = [1]
thing_classes    = [2]
# ──────────────────────────────────────────────────────────────────────────────


def _load_source_classification(scene_name: str, src_dir: str) -> np.ndarray | None:
    """Return per-point LAZ classification array for a scene, or None if unavailable.

    Wytham: src_dir/wytham_vox0.1_subset{NNN}.ply  (fields: classification)
    LAUTx:  src_dir/p{N}_vox0.1.laz               (requires laspy)
    """
    # Wytham subset
    m = re.match(r'wytham_subset(\d+)_test', scene_name)
    if m:
        subset_id = m.group(1)
        ply_path = os.path.join(src_dir, f'wytham_vox0.1_subset{subset_id}.ply')
        if not os.path.isfile(ply_path):
            return None
        from plyfile import PlyData as _PD
        d = _PD(text=True).read(ply_path)
        return d.elements[0].data['classification'].astype(np.int32)

    # LAUTx plot
    m = re.match(r'lautx_p(\d+)_test', scene_name)
    if m:
        plot_id = m.group(1)
        laz_path = os.path.join(src_dir, f'p{plot_id}_vox0.1.laz')
        if not os.path.isfile(laz_path):
            return None
        try:
            import laspy
            las = laspy.read(laz_path)
            return np.array(las.classification, dtype=np.int32)
        except Exception:
            return None
    return None


def _make_contiguous_labels(ins_arr: np.ndarray):
    unique_ids = np.sort(np.unique(ins_arr[ins_arr > 0]))
    if len(unique_ids) == 0:
        return np.zeros(len(ins_arr), dtype=np.int64), unique_ids
    pos    = np.searchsorted(unique_ids, ins_arr)
    labels = np.where(ins_arr != -1, pos + 1, 0).astype(np.int64)
    return labels, unique_ids


def _instance_sem_classes(labels: np.ndarray, sem: np.ndarray, M: int) -> np.ndarray:
    if M == 0:
        return np.empty(0, dtype=np.int32)
    MAX_SEM = int(sem.max()) + 1 if len(sem) else 1
    valid   = labels > 0
    flat    = labels[valid] * MAX_SEM + sem[valid].astype(np.int64)
    counts  = np.bincount(flat, minlength=(M + 1) * MAX_SEM)
    matrix  = counts.reshape(M + 1, MAX_SEM)[1:]
    return matrix.argmax(axis=1).astype(np.int32)


def _iou_matrix(pred_labels: np.ndarray, gt_labels: np.ndarray,
                M_pred: int, M_gt: int) -> np.ndarray:
    pred_sizes = np.bincount(pred_labels, minlength=M_pred + 1)[1:].astype(np.float64)
    gt_sizes   = np.bincount(gt_labels,   minlength=M_gt   + 1)[1:].astype(np.float64)
    flat  = pred_labels * np.int64(M_gt + 1) + gt_labels
    raw   = np.bincount(flat, minlength=(M_pred + 1) * (M_gt + 1))
    inter = raw.reshape(M_pred + 1, M_gt + 1)[1:, 1:].astype(np.float64)
    union = pred_sizes[:, None] + gt_sizes[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


# ──────────────────────────────────────────────────────────────────────────────

def process_single_file(ply_file: str,
                        src_classification: np.ndarray | None = None) -> dict:
    data = PlyData(text=True).read(ply_file)
    el   = data.elements[0].data

    ins_pred = el["instance_pred"].astype(np.int64)
    ins_gt   = el["instance_gt"  ].astype(np.int64)

    # ── Reconstruct semantic_gt from instance_gt ──────────────────────────────
    # instance_gt > 0  →  tree (label 1 → +1 = 2)
    # instance_gt == 0 →  ground/non-tree (label 0 → +1 = 1)
    sem_gt = (np.where(ins_gt > 0, 1, 0) + 1).astype(np.int32)   # 1=ground, 2=tree

    # ── Collapse model predictions: leaf(3) → tree(2) ────────────────────────
    # Model outputs ground(0), wood(1), leaf(2) → +1 = 1, 2, 3.
    # Leaf + Wood = Tree for wytham/LAUTx (GitHub issue #24).
    sem_pred = el["semantic_pred"].astype(np.int32) + 1
    sem_pred = np.where(sem_pred == 3, 2, sem_pred)   # leaf → tree, in-place safe

    # ── Optional: exclude LAZ classification=3 (unclassified) points ─────────
    # These are mid-canopy points from unlabelled trees: GT calls them "ground"
    # but the model correctly labels them "tree".  Excluding them gives a
    # cleaner evaluation that matches ForAINetV2 protocol (where void/unclassified
    # points are excluded from metrics via the idxc filter).
    if src_classification is not None:
        keep = (src_classification != 3)
        ins_pred = ins_pred[keep]
        ins_gt   = ins_gt[keep]
        sem_gt   = sem_gt[keep]
        sem_pred = sem_pred[keep]

    # ── Semantic evaluation — all (kept) points ───────────────────────────────
    sem_gt_m   = sem_gt
    sem_pred_m = sem_pred

    correct = (sem_gt_m == sem_pred_m)
    gt_classes            = np.bincount(sem_gt_m,           minlength=NUM_CLASSES_sem).astype(float)
    positive_classes      = np.bincount(sem_pred_m,         minlength=NUM_CLASSES_sem).astype(float)
    true_positive_classes = np.bincount(sem_gt_m[correct],  minlength=NUM_CLASSES_sem).astype(float)

    # Binary collapse on masked arrays
    sem_pred_bi_m = np.where(np.isin(sem_pred_m, stuff_classes), 1,
                    np.where(np.isin(sem_pred_m, thing_classes), 2, sem_pred_m))
    sem_gt_bi_m   = np.where(np.isin(sem_gt_m,   stuff_classes), 1,
                    np.where(np.isin(sem_gt_m,   thing_classes), 2, sem_gt_m))

    correct_bi = (sem_gt_bi_m == sem_pred_bi_m)
    gt_classes_bi            = np.bincount(sem_gt_bi_m,              minlength=NUM_CLASSES).astype(float)
    positive_classes_bi      = np.bincount(sem_pred_bi_m,            minlength=NUM_CLASSES).astype(float)
    true_positive_classes_bi = np.bincount(sem_gt_bi_m[correct_bi],  minlength=NUM_CLASSES).astype(float)

    # ── Instance evaluation — full point cloud, own filter ───────────────────
    # sem_gt / sem_pred here are over ALL points (not masked) for the idxc filter.
    sem_pred_bi_full = np.where(np.isin(sem_pred, stuff_classes), 1,
                       np.where(np.isin(sem_pred, thing_classes), 2, sem_pred))
    sem_gt_bi_full   = np.where(np.isin(sem_gt,   stuff_classes), 1,
                       np.where(np.isin(sem_gt,   thing_classes), 2, sem_gt))

    idxc = ((sem_gt_bi_full != 0) & (sem_gt_bi_full != 1)) | \
           ((sem_pred_bi_full != 0) & (sem_pred_bi_full != 1))
    pred_ins_f = ins_pred[idxc]
    gt_ins_f   = ins_gt[idxc]
    pred_sem_f = sem_pred_bi_full[idxc]
    gt_sem_f   = sem_gt_bi_full[idxc]

    pred_labels, pred_unique = _make_contiguous_labels(pred_ins_f)
    gt_labels,   gt_unique   = _make_contiguous_labels(gt_ins_f)
    M_pred_tot = len(pred_unique)
    M_gt_tot   = len(gt_unique)

    pred_inst_sem = _instance_sem_classes(pred_labels, pred_sem_f, M_pred_tot)
    gt_inst_sem   = _instance_sem_classes(gt_labels,   gt_sem_f,   M_gt_tot)

    # ── Per-class IoU / coverage / TP-FP ─────────────────────────────────────
    total_gt_ins          = np.zeros(NUM_CLASSES)
    tpsins                = [[] for _ in range(NUM_CLASSES)]
    fpsins                = [[] for _ in range(NUM_CLASSES)]
    IoU_Tp                = np.zeros(NUM_CLASSES)
    IoU_Mc                = np.zeros(NUM_CLASSES)
    all_mean_cov          = [[] for _ in range(NUM_CLASSES)]
    all_mean_weighted_cov = [[] for _ in range(NUM_CLASSES)]

    for i_sem in range(NUM_CLASSES):
        pred_idx_cls = np.where(pred_inst_sem == i_sem)[0] + 1
        gt_idx_cls   = np.where(gt_inst_sem   == i_sem)[0] + 1
        M_p, M_g = len(pred_idx_cls), len(gt_idx_cls)

        no_pred = (M_p == 0)
        no_gt   = (M_g == 0)

        if no_pred and no_gt:
            all_mean_cov[i_sem].append(0.0)
            all_mean_weighted_cov[i_sem].append(0.0)
            continue
        if no_pred:
            all_mean_cov[i_sem].append(0.0)
            all_mean_weighted_cov[i_sem].append(0.0)
            total_gt_ins[i_sem] += M_g
            continue
        if no_gt:
            all_mean_cov[i_sem].append(0.0)
            all_mean_weighted_cov[i_sem].append(0.0)
            tpsins[i_sem] += [0.] * M_p
            fpsins[i_sem] += [1.] * M_p
            continue

        pred_remap = np.zeros(M_pred_tot + 1, dtype=np.int64)
        pred_remap[pred_idx_cls] = np.arange(1, M_p + 1, dtype=np.int64)
        pred_sub = pred_remap[pred_labels]

        gt_remap = np.zeros(M_gt_tot + 1, dtype=np.int64)
        gt_remap[gt_idx_cls] = np.arange(1, M_g + 1, dtype=np.int64)
        gt_sub = gt_remap[gt_labels]

        iou_mat = _iou_matrix(pred_sub, gt_sub, M_p, M_g)

        gt_sizes_cls = np.bincount(gt_sub, minlength=M_g + 1)[1:].astype(float)
        cov_per_gt   = iou_mat.max(axis=0)
        all_mean_cov[i_sem].append(float(cov_per_gt.mean()))
        all_mean_weighted_cov[i_sem].append(
            float((cov_per_gt * gt_sizes_cls).sum() / gt_sizes_cls.sum()))

        total_gt_ins[i_sem] += M_g
        best    = iou_mat.max(axis=1)
        tp_mask = (best >= 0.5).astype(float)
        tpsins[i_sem] += tp_mask.tolist()
        fpsins[i_sem] += (1.0 - tp_mask).tolist()
        IoU_Tp[i_sem] += best[best >= 0.5].sum()
        IoU_Mc[i_sem] += best[best >  0  ].sum()

    # ── Semantic IoU (computed on masked points only) ─────────────────────────
    iou_list, sem_classcount_have = [], []
    for i in range(NUM_CLASSES_sem):
        if gt_classes[i] > 0:
            sem_classcount_have.append(i)
            iou = true_positive_classes[i] / float(
                gt_classes[i] + positive_classes[i] - true_positive_classes[i])
        else:
            iou = 0.0
        iou_list.append(iou)

    scf  = list(set(sem_classcount) & set(sem_classcount_have))
    miou = sum(iou_list[i] for i in scf) / len(scf) if scf else 0.0

    iou_list_bi = []
    for i in range(NUM_CLASSES):
        if gt_classes_bi[i] > 0:
            iou = true_positive_classes_bi[i] / float(
                gt_classes_bi[i] + positive_classes_bi[i] - true_positive_classes_bi[i])
        else:
            iou = 0.0
        iou_list_bi.append(iou)

    precision = np.zeros(NUM_CLASSES)
    recall    = np.zeros(NUM_CLASSES)
    RQ = np.zeros(NUM_CLASSES);  SQ = np.zeros(NUM_CLASSES)
    PQ = np.zeros(NUM_CLASSES);  PQStar = np.zeros(NUM_CLASSES)
    icf = list(set(ins_classcount) & set(sem_classcount_have))

    for i_sem in ins_classcount:
        if not tpsins[i_sem]:
            continue
        tp   = float(np.sum(tpsins[i_sem]))
        fp   = float(np.sum(fpsins[i_sem]))
        rec  = tp / total_gt_ins[i_sem] if total_gt_ins[i_sem] > 0 else 0.0
        prec = tp / (tp + fp)           if (tp + fp) > 0          else 0.0
        precision[i_sem] = prec;  recall[i_sem] = rec
        RQ[i_sem]     = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
        SQ[i_sem]     = IoU_Tp[i_sem] / tp    if tp > 0          else 0.0
        PQ[i_sem]     = SQ[i_sem] * RQ[i_sem]
        PQStar[i_sem] = PQ[i_sem]

    for i_sem in stuff_classcount:
        SQ[i_sem]     = iou_list_bi[i_sem]
        RQ[i_sem]     = 1.0 if SQ[i_sem] >= 0.5 else 0.0
        PQ[i_sem]     = SQ[i_sem] * RQ[i_sem]
        PQStar[i_sem] = SQ[i_sem]

    mean_prec = float(np.mean(precision[icf])) if icf else 0.0
    mean_rec  = float(np.mean(recall[icf]))    if icf else 0.0
    F1 = 2*mean_prec*mean_rec/(mean_prec+mean_rec) if (mean_prec+mean_rec) > 0 else 0.0

    MUCov = np.array([np.mean(all_mean_cov[i]) if all_mean_cov[i] else 0.0
                      for i in range(NUM_CLASSES)])

    n_trees = int(np.sum(gt_inst_sem == 2))

    return {
        'scene': os.path.basename(ply_file).rsplit('.', 1)[0],
        'prec': mean_prec, 'rec': mean_rec, 'f1': F1,
        'cov':  float(MUCov[2]),
        'iou_g': iou_list[1], 'iou_w': iou_list[2],
        'miou': miou,
        'n_trees': n_trees,
        # aggregation intermediates
        'true_positive_classes': true_positive_classes,
        'positive_classes':      positive_classes,
        'gt_classes':            gt_classes,
        'true_positive_classes_bi': true_positive_classes_bi,
        'positive_classes_bi':      positive_classes_bi,
        'gt_classes_bi':            gt_classes_bi,
        'total_gt_ins': total_gt_ins,
        'tpsins': tpsins, 'fpsins': fpsins,
        'IoU_Tp': IoU_Tp, 'IoU_Mc': IoU_Mc,
        'all_mean_cov':          all_mean_cov,
        'all_mean_weighted_cov': all_mean_weighted_cov,
        'iou_list': iou_list, 'iou_list_bi': iou_list_bi,
        'sem_classcount_have': sem_classcount_have,
    }


# ── Global aggregation ────────────────────────────────────────────────────────

def aggregate_global(results: list) -> dict:
    tpc_g  = np.zeros(NUM_CLASSES_sem); pc_g  = np.zeros(NUM_CLASSES_sem)
    gtc_g  = np.zeros(NUM_CLASSES_sem)
    tpc_bi = np.zeros(NUM_CLASSES);    pc_bi = np.zeros(NUM_CLASSES)
    gtc_bi = np.zeros(NUM_CLASSES)
    tgi_g  = np.zeros(NUM_CLASSES)
    tps_g  = [[] for _ in range(NUM_CLASSES)]
    fps_g  = [[] for _ in range(NUM_CLASSES)]
    IoU_Tp_g = np.zeros(NUM_CLASSES);  IoU_Mc_g = np.zeros(NUM_CLASSES)
    cov_g    = [[] for _ in range(NUM_CLASSES)]
    wcov_g   = [[] for _ in range(NUM_CLASSES)]

    for r in results:
        tpc_g  += r['true_positive_classes'];  pc_g  += r['positive_classes']
        gtc_g  += r['gt_classes']
        tpc_bi += r['true_positive_classes_bi']; pc_bi += r['positive_classes_bi']
        gtc_bi += r['gt_classes_bi']
        tgi_g  += r['total_gt_ins']
        for i in range(NUM_CLASSES):
            tps_g[i]    += r['tpsins'][i];    fps_g[i]    += r['fpsins'][i]
            IoU_Tp_g[i] += r['IoU_Tp'][i];   IoU_Mc_g[i] += r['IoU_Mc'][i]
            cov_g[i]    += r['all_mean_cov'][i]
            wcov_g[i]   += r['all_mean_weighted_cov'][i]

    iou_g, sch_g = [], []
    for i in range(NUM_CLASSES_sem):
        if gtc_g[i] > 0:
            sch_g.append(i)
            iou_g.append(tpc_g[i] / float(gtc_g[i] + pc_g[i] - tpc_g[i]))
        else:
            iou_g.append(0.0)
    scf_g  = list(set(sem_classcount) & set(sch_g))
    miou_g = sum(iou_g[i] for i in scf_g) / len(scf_g) if scf_g else 0.0

    iou_bi_g, sch_bi_g = [], []
    for i in range(NUM_CLASSES):
        if gtc_bi[i] > 0:
            sch_bi_g.append(i)
            iou_bi_g.append(tpc_bi[i] / float(gtc_bi[i] + pc_bi[i] - tpc_bi[i]))
        else:
            iou_bi_g.append(0.0)
    scbi_g = [1, 2]
    scfb_g = list(set(scbi_g) & set(sch_bi_g))
    stcf_g = list(set(stuff_classcount) & set(sch_bi_g))

    prec_g = np.zeros(NUM_CLASSES);  rec_g = np.zeros(NUM_CLASSES)
    RQ_g   = np.zeros(NUM_CLASSES);  SQ_g  = np.zeros(NUM_CLASSES)
    PQ_g   = np.zeros(NUM_CLASSES);  PQs_g = np.zeros(NUM_CLASSES)
    icf_g  = list(set(ins_classcount) & set(sch_g))

    for i_sem in ins_classcount:
        if not tps_g[i_sem]:
            continue
        tp = float(np.sum(tps_g[i_sem]));  fp = float(np.sum(fps_g[i_sem]))
        rec  = tp / tgi_g[i_sem] if tgi_g[i_sem] > 0 else 0.0
        prec = tp / (tp + fp)    if (tp + fp) > 0     else 0.0
        prec_g[i_sem] = prec;  rec_g[i_sem] = rec
        RQ_g[i_sem]  = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
        SQ_g[i_sem]  = IoU_Tp_g[i_sem] / tp  if tp > 0          else 0.0
        PQ_g[i_sem]  = SQ_g[i_sem] * RQ_g[i_sem]
        PQs_g[i_sem] = PQ_g[i_sem]

    for i_sem in stuff_classcount:
        SQ_g[i_sem]  = iou_bi_g[i_sem]
        RQ_g[i_sem]  = 1.0 if SQ_g[i_sem] >= 0.5 else 0.0
        PQ_g[i_sem]  = SQ_g[i_sem] * RQ_g[i_sem]
        PQs_g[i_sem] = SQ_g[i_sem]

    MUCov_g = np.array([np.mean(cov_g[i])  if cov_g[i]  else 0.0 for i in range(NUM_CLASSES)])
    MWCov_g = np.array([np.mean(wcov_g[i]) if wcov_g[i] else 0.0 for i in range(NUM_CLASSES)])

    mp = float(np.mean(prec_g[icf_g])) if icf_g else 0.0
    mr = float(np.mean(rec_g[icf_g]))  if icf_g else 0.0
    F1 = 2*mp*mr/(mp+mr) if (mp+mr) > 0 else 0.0

    return dict(
        prec=mp, rec=mr, f1=F1,
        cov=float(MUCov_g[2]),
        iou_g=iou_g[1], iou_w=iou_g[2], miou=miou_g,
        iou_list_global=iou_g, iou_list_bi_global=iou_bi_g,
        scf_g=scf_g, scfb_g=scfb_g, stcf_g=stcf_g, icf_g=icf_g, scbi_g=scbi_g,
        tpc_g=tpc_g, pc_g=pc_g, gtc_g=gtc_g,
        tpc_bi=tpc_bi, pc_bi=pc_bi, gtc_bi=gtc_bi,
        MUCov_g=MUCov_g, MWCov_g=MWCov_g,
        prec_g=prec_g, rec_g=rec_g,
        RQ_g=RQ_g, SQ_g=SQ_g, PQ_g=PQ_g, PQs_g=PQs_g,
    )


# ── Region mapping ────────────────────────────────────────────────────────────

REGIONS = ['Wytham', 'LAUTx']

_REGION_PREFIXES = {
    'Wytham': ('wytham', 'Wytham'),
    'LAUTx':  ('lautx',  'LAUTx'),
}

def get_region(scene_name: str) -> str:
    for region, prefixes in _REGION_PREFIXES.items():
        for p in prefixes:
            if scene_name.startswith(p):
                return region
    return 'Other'


def _weighted_mean(values, weights):
    w = np.asarray(weights, dtype=float)
    v = np.asarray(values,  dtype=float)
    mask = (v > 0)
    if not np.any(mask):
        return 0.0
    w = w[mask];  v = v[mask]
    total = w.sum()
    return float((v * w).sum() / total) if total > 0 else 0.0


def format_region_table(scene_rows: list) -> str:
    METRIC_KEYS = ('prec', 'rec', 'f1', 'cov', 'iou_g', 'iou_w', 'miou')
    METRIC_COLS = ['Prec', 'Rec', 'F1', 'Cov', 'IoU_ground', 'IoU_tree', 'mIoU']

    COL_REGION = 10
    COL_TREES  = 7
    col_w      = 11

    hdr = (f"{'Region':<{COL_REGION}}"
           f"{'#Trees':>{COL_TREES}}"
           + "".join(f"{c:>{col_w}}" for c in METRIC_COLS))
    sep = "-" * len(hdr)

    from collections import defaultdict
    groups = defaultdict(list)
    for row in scene_rows:
        groups[get_region(row['scene'])].append(row)

    lines = [sep, hdr, sep]

    for region in REGIONS:
        rows = groups.get(region, [])
        if not rows:
            continue
        weights     = [r['n_trees'] for r in rows]
        total_trees = sum(weights)
        region_row  = f"{region:<{COL_REGION}}{total_trees:>{COL_TREES}}"
        for k in METRIC_KEYS:
            wm = _weighted_mean([r[k] for r in rows], weights)
            region_row += f"{wm:>{col_w}.4f}"
        lines.append(region_row)

    other_rows = groups.get('Other', [])

    lines.append(sep)
    total_all_trees = sum(r['n_trees'] for r in scene_rows)
    all_row = f"{'ALL (weighted)':<{COL_REGION}}{total_all_trees:>{COL_TREES}}"
    for k in METRIC_KEYS:
        vals    = [r[k]         for r in scene_rows]
        weights = [r['n_trees'] for r in scene_rows]
        all_row += f"{_weighted_mean(vals, weights):>{col_w}.4f}"
    lines += [all_row, sep]

    if other_rows:
        lines.append(f"  (unmatched scenes: {', '.join(r['scene'] for r in other_rows)})")

    return "\n".join(lines)


# ── Output formatting ─────────────────────────────────────────────────────────

def format_table(scene_rows: list, g: dict) -> str:
    COL_SCENE = 40
    col_w = 11
    cols  = ['Prec', 'Rec', 'F1', 'Cov', 'IoU_ground', 'IoU_tree', 'mIoU']
    keys  = ('prec', 'rec', 'f1', 'cov', 'iou_g',      'iou_w',   'miou')
    hdr   = f"{'Scene':<{COL_SCENE}}" + "".join(f"{c:>{col_w}}" for c in cols)
    sep   = "-" * len(hdr)
    lines = [sep, hdr, sep]
    for row in scene_rows:
        line  = f"{row['scene']:<{COL_SCENE}}"
        line += "".join(f"{row[k]:>{col_w}.4f}" for k in keys)
        lines.append(line)
    lines.append(sep)
    mean_line  = f"{'MEAN':<{COL_SCENE}}"
    mean_line += "".join(f"{g[k]:>{col_w}.4f}" for k in keys)
    lines += [mean_line, sep]
    return "\n".join(lines)


def format_verbose(g: dict) -> str:
    scf  = g['scf_g'];   scfb = g['scfb_g'];  stcf = g['stcf_g']
    icf  = g['icf_g'];   scbi = g['scbi_g']
    iou_g  = g['iou_list_global']
    iou_bi = g['iou_list_bi_global']

    L = ["\n=== Global Semantic Segmentation ==="]
    L.append(f"  Tree = Leaf + Wood (model classes 1+2);  Ground = non-tree (model class 0)")
    L.append(f"  IoU_ground note: unlabelled canopy (LAZ cls=3) is GT=ground but model predicts tree.")
    L.append(f"  Use --src_dir to exclude those points; this report reflects the current filter setting.")
    L.append(f"  IoU[0=none, 1=ground, 2=tree(leaf+wood)]")
    L.append(f"oAcc:  {g['tpc_g'].sum() / g['pc_g'].sum():.4f}")
    L.append(f"mAcc:  {np.mean(g['tpc_g'][scf] / g['gtc_g'][scf]):.4f}")
    L.append(f"IoU_ground={iou_g[1]:.4f}  IoU_tree={iou_g[2]:.4f}")
    L.append(f"mIoU:  {g['miou']:.4f}")

    L.append("\n=== Global Binary Semantic Segmentation (all points) ===")
    L.append(f"  IoU[0=none, 1=ground, 2=tree(leaf+wood)]")
    L.append(f"oAcc:  {g['tpc_bi'].sum() / g['pc_bi'].sum():.4f}")
    L.append(f"mAcc:  {np.mean(g['tpc_bi'][scfb] / g['gtc_bi'][scfb]):.4f}")
    L.append(f"IoU_ground={iou_bi[1]:.4f}  IoU_tree={iou_bi[2]:.4f}")
    miou_bi = sum(iou_bi[i] for i in scfb) / len(scfb) if scfb else 0.0
    L.append(f"mIoU:  {miou_bi:.4f}")

    L.append("\n=== Global Instance Segmentation ===")
    L.append(f"MUCov:         {g['MUCov_g'][ins_classcount]}")
    L.append(f"mMUCov:        {np.mean(g['MUCov_g'][icf]):.4f}")
    L.append(f"MWCov:         {g['MWCov_g'][ins_classcount]}")
    L.append(f"mMWCov:        {np.mean(g['MWCov_g'][icf]):.4f}")
    L.append(f"Precision:     {g['prec_g'][ins_classcount]}")
    L.append(f"mPrecision:    {g['prec']:.4f}")
    L.append(f"Recall:        {g['rec_g'][ins_classcount]}")
    L.append(f"mRecall:       {g['rec']:.4f}")
    L.append(f"F1 score:      {g['f1']:.4f}")
    L.append(f"RQ:            {g['RQ_g'][scbi]}")
    L.append(f"meanRQ:        {np.mean(g['RQ_g'][scfb]):.4f}")
    L.append(f"SQ:            {g['SQ_g'][scbi]}")
    L.append(f"meanSQ:        {np.mean(g['SQ_g'][scfb]):.4f}")
    L.append(f"PQ:            {g['PQ_g'][scbi]}")
    L.append(f"meanPQ:        {np.mean(g['PQ_g'][scfb]):.4f}")
    L.append(f"PQ*:           {g['PQs_g'][scbi]}")
    L.append(f"mean PQ*:      {np.mean(g['PQs_g'][scfb]):.4f}")
    L.append(f"RQ (things):   {g['RQ_g'][ins_classcount]}")
    L.append(f"meanRQ (th):   {np.mean(g['RQ_g'][icf]):.4f}")
    L.append(f"SQ (things):   {g['SQ_g'][ins_classcount]}")
    L.append(f"meanSQ (th):   {np.mean(g['SQ_g'][icf]):.4f}")
    L.append(f"PQ (things):   {g['PQ_g'][ins_classcount]}")
    L.append(f"meanPQ (th):   {np.mean(g['PQ_g'][icf]):.4f}")
    L.append(f"RQ (stuff):    {g['RQ_g'][stuff_classcount]}")
    L.append(f"meanRQ (st):   {np.mean(g['RQ_g'][stcf]):.4f}")
    L.append(f"SQ (stuff):    {g['SQ_g'][stuff_classcount]}")
    L.append(f"meanSQ (st):   {np.mean(g['SQ_g'][stcf]):.4f}")
    L.append(f"PQ (stuff):    {g['PQ_g'][stuff_classcount]}")
    L.append(f"meanPQ (st):   {np.mean(g['PQ_g'][stcf]):.4f}")
    return "\n".join(L)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate 3D instance segmentation on Wytham + LAUTx dataset.')
    parser.add_argument('pred_path',
                        help='Directory containing prediction .ply files')
    parser.add_argument('--workers', type=int, default=8,
                        help='Parallel worker processes (default: 8)')
    parser.add_argument('--src_dir', default=None,
                        help='Directory containing the source voxelized PLY/LAZ files.  '
                             'When set, LAZ classification=3 (unclassified mid-canopy) '
                             'points are excluded from all metrics.  '
                             'Example: data/wytham_lautx/wytham/voxelized_subsets')
    args = parser.parse_args()

    ply_files = sorted(glob.glob(os.path.join(args.pred_path, '*.ply')))
    if not ply_files:
        print(f"No .ply files found in {args.pred_path}")
        sys.exit(1)

    n = len(ply_files)
    filter_note = f"  |  src_dir={args.src_dir} (unclassified filtered)" if args.src_dir else ""
    print(f"Evaluating {n} scenes  |  workers={args.workers}{filter_note}")

    # Pre-load source classifications (cheap per-scene) so workers receive arrays
    src_cls_map: dict[str, np.ndarray | None] = {}
    if args.src_dir:
        for f in ply_files:
            scene = os.path.basename(f).rsplit('.', 1)[0]
            src_cls_map[scene] = _load_source_classification(scene, args.src_dir)
            status = "loaded" if src_cls_map[scene] is not None else "not found"
            print(f"  src classification [{scene}]: {status}")
    else:
        for f in ply_files:
            scene = os.path.basename(f).rsplit('.', 1)[0]
            src_cls_map[scene] = None

    results = [None] * n
    errors  = []
    done    = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_idx = {
            executor.submit(
                process_single_file, f,
                src_cls_map[os.path.basename(f).rsplit('.', 1)[0]]
            ): i
            for i, f in enumerate(ply_files)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
                done += 1
                print(f"  [{done}/{n}] {os.path.basename(ply_files[idx])}")
            except Exception as exc:
                errors.append((ply_files[idx], str(exc)))
                print(f"  ERROR {os.path.basename(ply_files[idx])}: {exc}")

    results = [r for r in results if r is not None]
    if not results:
        print("All files failed.")
        sys.exit(1)

    global_metrics = aggregate_global(results)
    table        = format_table(results, global_metrics)
    region_table = format_region_table(results)
    verbose      = format_verbose(global_metrics)

    filter_header = ""
    if args.src_dir:
        filter_header = "\n[NOTE: LAZ classification=3 (unclassified mid-canopy) points excluded]\n"

    output = (filter_header + table
              + "\n\n=== Tree-count-weighted averages per region ===\n"
              + region_table
              + "\n" + verbose + "\n")

    if errors:
        output += "\n=== Errors ===\n"
        for path, msg in errors:
            output += f"{os.path.basename(path)}: {msg}\n"

    print("\n" + output)

    suffix = "_filtered" if args.src_dir else ""
    out_file = os.path.join(args.pred_path, f'evaluation_summary_wytham_lautx{suffix}.txt')
    with open(out_file, 'w') as f:
        f.write(output)
    print(f"Results saved to: {out_file}")
