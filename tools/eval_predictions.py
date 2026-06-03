"""
Evaluate 3D forest instance segmentation predictions.

Speed-ups over final_eval.py:
  - ProcessPoolExecutor (true multiprocessing, bypasses GIL)
  - Vectorised semantic confusion via np.bincount  (no per-point Python loop)
  - Vectorised IoU matrix via label-pair encoding  (no nested boolean-mask loop)
  - Vectorised instance-semantic-class assignment  (no scipy.stats.mode per instance)
"""

import glob
import os
import sys
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from plyfile import PlyData

# ── class layout ──────────────────────────────────────────────────────────────
NUM_CLASSES     = 3   # 0=unclassified, 1=non-tree(stuff), 2=tree(thing)
NUM_CLASSES_sem = 4   # same + class 3 (leaf sub-class)
ins_classcount    = [2]
stuff_classcount  = [1]
sem_classcount    = [1, 2, 3]
stuff_classes     = [1]
thing_classes     = [2, 3]
# ──────────────────────────────────────────────────────────────────────────────


def _make_contiguous_labels(ins_arr: np.ndarray):
    """
    Map arbitrary instance-ID array to 1-indexed contiguous labels.
    Background (value <= 0 or == -1) maps to 0.
    Dataset convention: instance_gt == 0 means unclassified/no-instance.
    Returns (labels, unique_ids).
    """
    unique_ids = np.sort(np.unique(ins_arr[ins_arr > 0]))
    if len(unique_ids) == 0:
        return np.zeros(len(ins_arr), dtype=np.int64), unique_ids

    pos = np.searchsorted(unique_ids, ins_arr)          # O(N log M)
    labels = np.where(ins_arr != -1, pos + 1, 0).astype(np.int64)   # 0=bg, 1..M
    return labels, unique_ids


def _instance_sem_classes(labels: np.ndarray, sem: np.ndarray, M: int) -> np.ndarray:
    """
    For each of the M instances (1-indexed in *labels*) return its semantic class
    as the mode of *sem* over its member points.
    Returns int array of shape (M,), values in [0, MAX_SEM].
    """
    if M == 0:
        return np.empty(0, dtype=np.int32)
    MAX_SEM = int(sem.max()) + 1 if len(sem) else 1
    valid = labels > 0
    flat = labels[valid] * MAX_SEM + sem[valid].astype(np.int64)
    counts = np.bincount(flat, minlength=(M + 1) * MAX_SEM)
    matrix = counts.reshape(M + 1, MAX_SEM)[1:]          # (M, MAX_SEM)
    return matrix.argmax(axis=1).astype(np.int32)


def _iou_matrix(pred_labels: np.ndarray, gt_labels: np.ndarray,
                M_pred: int, M_gt: int) -> np.ndarray:
    """
    Compute (M_pred × M_gt) IoU matrix via label-pair bincount.
    O(N_points) – no nested loops.
    """
    pred_sizes = np.bincount(pred_labels, minlength=M_pred + 1)[1:].astype(np.float64)
    gt_sizes   = np.bincount(gt_labels,   minlength=M_gt   + 1)[1:].astype(np.float64)

    flat  = pred_labels * np.int64(M_gt + 1) + gt_labels
    raw   = np.bincount(flat, minlength=(M_pred + 1) * (M_gt + 1))
    inter = raw.reshape(M_pred + 1, M_gt + 1)[1:, 1:].astype(np.float64)

    union = pred_sizes[:, None] + gt_sizes[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


# ──────────────────────────────────────────────────────────────────────────────

def process_single_file(ply_file: str) -> dict:
    data = PlyData(text=True).read(ply_file)
    el   = data.elements[0].data

    sem_pred = el["semantic_pred"].astype(np.int32) + 1   # 1-indexed
    sem_gt   = el["semantic_gt" ].astype(np.int32) + 1
    ins_pred = el["instance_pred"].astype(np.int64)
    ins_gt   = el["instance_gt"  ].astype(np.int64)

    # ── 4-class semantic confusion (vectorised) ───────────────────────────────
    correct = (sem_gt == sem_pred)
    gt_classes            = np.bincount(sem_gt,           minlength=NUM_CLASSES_sem).astype(float)
    positive_classes      = np.bincount(sem_pred,         minlength=NUM_CLASSES_sem).astype(float)
    true_positive_classes = np.bincount(sem_gt[correct],  minlength=NUM_CLASSES_sem).astype(float)

    # ── Collapse to binary  (stuff→1, thing→2) ───────────────────────────────
    sem_pred_bi = sem_pred.copy()
    sem_gt_bi   = sem_gt.copy()
    for c in stuff_classes:
        sem_pred_bi[sem_pred == c] = 1;  sem_gt_bi[sem_gt == c] = 1
    for c in thing_classes:
        sem_pred_bi[sem_pred == c] = 2;  sem_gt_bi[sem_gt == c] = 2

    correct_bi = (sem_gt_bi == sem_pred_bi)
    gt_classes_bi            = np.bincount(sem_gt_bi,              minlength=NUM_CLASSES).astype(float)
    positive_classes_bi      = np.bincount(sem_pred_bi,            minlength=NUM_CLASSES).astype(float)
    true_positive_classes_bi = np.bincount(sem_gt_bi[correct_bi],  minlength=NUM_CLASSES).astype(float)

    # ── Filter to non-background points for instance eval ────────────────────
    idxc = ((sem_gt != 0) & (sem_gt != 1)) | ((sem_pred != 0) & (sem_pred != 1))
    pred_ins_f = ins_pred[idxc]
    gt_ins_f   = ins_gt[idxc]
    pred_sem_f = sem_pred_bi[idxc]
    gt_sem_f   = sem_gt_bi[idxc]

    # ── Build contiguous labels ───────────────────────────────────────────────
    pred_labels, pred_unique = _make_contiguous_labels(pred_ins_f)
    gt_labels,   gt_unique   = _make_contiguous_labels(gt_ins_f)
    M_pred_tot = len(pred_unique)
    M_gt_tot   = len(gt_unique)

    # ── Instance → semantic class (vectorised mode) ───────────────────────────
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

    # Pre-build remap lookup tables (one per class) – avoids per-point Python loops
    for i_sem in range(NUM_CLASSES):
        pred_idx_cls = np.where(pred_inst_sem == i_sem)[0] + 1  # 1-indexed
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

        # Vectorised remap: old contiguous index → new class-local index
        pred_remap = np.zeros(M_pred_tot + 1, dtype=np.int64)
        pred_remap[pred_idx_cls] = np.arange(1, M_p + 1, dtype=np.int64)
        pred_sub = pred_remap[pred_labels]   # O(N), no Python loop

        gt_remap = np.zeros(M_gt_tot + 1, dtype=np.int64)
        gt_remap[gt_idx_cls] = np.arange(1, M_g + 1, dtype=np.int64)
        gt_sub = gt_remap[gt_labels]

        iou_mat = _iou_matrix(pred_sub, gt_sub, M_p, M_g)   # (M_p, M_g)

        # Coverage (for each GT, best matching pred)
        gt_sizes_cls = np.bincount(gt_sub, minlength=M_g + 1)[1:].astype(float)
        cov_per_gt   = iou_mat.max(axis=0)                   # (M_g,)
        all_mean_cov[i_sem].append(float(cov_per_gt.mean()))
        all_mean_weighted_cov[i_sem].append(
            float((cov_per_gt * gt_sizes_cls).sum() / gt_sizes_cls.sum()))

        # TP / FP (for each pred, best matching GT)
        total_gt_ins[i_sem] += M_g
        best = iou_mat.max(axis=1)                           # (M_p,)
        tp_mask = (best >= 0.5).astype(float)
        tpsins[i_sem] += tp_mask.tolist()
        fpsins[i_sem] += (1.0 - tp_mask).tolist()
        IoU_Tp[i_sem] += best[best >= 0.5].sum()
        IoU_Mc[i_sem] += best[best >  0  ].sum()

    # ── Display metrics ───────────────────────────────────────────────────────
    iou_list, sem_classcount_have = [], []
    for i in range(NUM_CLASSES_sem):
        if gt_classes[i] > 0:
            sem_classcount_have.append(i)
            iou = true_positive_classes[i] / float(
                gt_classes[i] + positive_classes[i] - true_positive_classes[i])
        else:
            iou = 0.0
        iou_list.append(iou)

    scf = list(set(sem_classcount) & set(sem_classcount_have))
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
        tp = float(np.sum(tpsins[i_sem]))
        fp = float(np.sum(fpsins[i_sem]))
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

    # GT tree count for weighting (class 2 = tree, regardless of predictions)
    n_trees = int(np.sum(gt_inst_sem == 2))

    return {
        'scene': os.path.basename(ply_file).rsplit('.', 1)[0],
        # ── table columns ──
        'prec': mean_prec, 'rec': mean_rec, 'f1': F1,
        'cov':  float(MUCov[2]),
        'iou_g': iou_list[1], 'iou_w': iou_list[2], 'iou_l': iou_list[3],
        'miou': miou,
        'n_trees': n_trees,
        # ── aggregation intermediates ──
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

    # 4-class IoU
    iou_g, sch_g = [], []
    for i in range(NUM_CLASSES_sem):
        if gtc_g[i] > 0:
            sch_g.append(i)
            iou_g.append(tpc_g[i] / float(gtc_g[i] + pc_g[i] - tpc_g[i]))
        else:
            iou_g.append(0.0)
    scf_g   = list(set(sem_classcount) & set(sch_g))
    miou_g  = sum(iou_g[i] for i in scf_g) / len(scf_g) if scf_g else 0.0

    # Binary IoU
    iou_bi_g, sch_bi_g = [], []
    for i in range(NUM_CLASSES):
        if gtc_bi[i] > 0:
            sch_bi_g.append(i)
            iou_bi_g.append(tpc_bi[i] / float(gtc_bi[i] + pc_bi[i] - tpc_bi[i]))
        else:
            iou_bi_g.append(0.0)
    scbi_g  = [1, 2]
    scfb_g  = list(set(scbi_g) & set(sch_bi_g))
    stcf_g  = list(set(stuff_classcount) & set(sch_bi_g))

    # Instance metrics
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
        iou_g=iou_g[1], iou_w=iou_g[2], iou_l=iou_g[3], miou=miou_g,
        # verbose fields
        iou_list_global=iou_g, iou_list_bi_global=iou_bi_g,
        scf_g=scf_g, scfb_g=scfb_g, stcf_g=stcf_g, icf_g=icf_g, scbi_g=scbi_g,
        tpc_g=tpc_g, pc_g=pc_g, gtc_g=gtc_g,
        tpc_bi=tpc_bi, pc_bi=pc_bi, gtc_bi=gtc_bi,
        MUCov_g=MUCov_g, MWCov_g=MWCov_g,
        prec_g=prec_g, rec_g=rec_g,
        RQ_g=RQ_g, SQ_g=SQ_g, PQ_g=PQ_g, PQs_g=PQs_g,
    )


# ── Region mapping ────────────────────────────────────────────────────────────

# Order matches the user-requested display order
REGIONS = ['CULS', 'YuChen', 'Tuwein', 'Scion', 'RMIT', 'BlueCat', 'NIBIO']

_REGION_PREFIXES = {
    'CULS':    ('CULS',),
    'YuChen':  ('Yuchen', 'yuchen'),
    'Tuwein':  ('TUWIEN', 'Tuwein'),
    'Scion':   ('SCION', 'Scion'),
    'RMIT':    ('RMIT',),
    'BlueCat': ('BlueCat', 'bluecat'),
    'NIBIO':   ('NIBIO',),          # covers NIBIO_ and NIBIO2_
}

def get_region(scene_name: str) -> str:
    """Return the region label for a scene name, or 'Other'."""
    for region, prefixes in _REGION_PREFIXES.items():
        for p in prefixes:
            if scene_name.startswith(p):
                return region
    return 'Other'


def _weighted_mean(values, weights):
    """
    Weighted mean ignoring zero values.
    Only entries with v > 0 are considered.
    """
    w = np.asarray(weights, dtype=float)
    v = np.asarray(values,  dtype=float)

    mask = (v > 0)   # Ignore the value = 0 (BlueCat dataset)

    if not np.any(mask):
        return 0.0

    w = w[mask]
    v = v[mask]

    total = w.sum()
    return float((v * w).sum() / total) if total > 0 else 0.0


def _region_metrics(rows: list) -> dict:
    """
    Compute aggregated metrics for a group of per-scene result dicts.

    F1  is derived from weighted Prec and Rec — not averaged directly.
    mIoU is derived from the three weighted IoU components — not averaged directly.
    This is consistent with how the global MEAN row is computed from pooled counts.
    """
    weights = [r['n_trees'] for r in rows]
    w_prec  = _weighted_mean([r['prec']  for r in rows], weights)
    w_rec   = _weighted_mean([r['rec']   for r in rows], weights)
    w_f1    = (2 * w_prec * w_rec / (w_prec + w_rec)
               if (w_prec + w_rec) > 0 else 0.0)
    w_cov   = _weighted_mean([r['cov']   for r in rows], weights)
    w_iou_g = _weighted_mean([r['iou_g'] for r in rows], weights)
    w_iou_w = _weighted_mean([r['iou_w'] for r in rows], weights)
    w_iou_l = _weighted_mean([r['iou_l'] for r in rows], weights)
    # mIoU: average of the weighted IoU components that are non-zero
    # (iou_g == 0 for datasets with no labelled ground class, e.g. BlueCat)
    iou_vals = [v for v in [w_iou_g, w_iou_w, w_iou_l] if v > 0]
    w_miou   = float(np.mean(iou_vals)) if iou_vals else 0.0
    return dict(prec=w_prec, rec=w_rec, f1=w_f1, cov=w_cov,
                iou_g=w_iou_g, iou_w=w_iou_w, iou_l=w_iou_l, miou=w_miou)


def format_region_table(scene_rows: list) -> str:
    """
    Build two tables:
      1. Per-region tree-count-weighted averages  (one row per region).
      2. A single all-regions tree-count-weighted row.

    F1 and mIoU are computed from weighted Prec/Rec and weighted IoU
    components respectively — NOT as weighted means of per-scene F1/mIoU.
    """
    METRIC_KEYS = ('prec', 'rec', 'f1', 'cov', 'iou_g', 'iou_w', 'iou_l', 'miou')
    METRIC_COLS = ['Prec', 'Rec', 'F1', 'Cov', 'IoU_g', 'IoU_w', 'IoU_l', 'mIoU']

    COL_REGION = 12
    COL_TREES  = 7
    col_w      = 7

    hdr = (f"{'Region':<{COL_REGION}}"
           f"{'#Trees':>{COL_TREES}}"
           + "".join(f"{c:>{col_w}}" for c in METRIC_COLS))
    sep = "-" * len(hdr)

    # Group scenes by region
    from collections import defaultdict
    groups = defaultdict(list)
    for row in scene_rows:
        groups[get_region(row['scene'])].append(row)

    lines = [sep, hdr, sep]

    for region in REGIONS:
        rows = groups.get(region, [])
        if not rows:
            continue
        total_trees = sum(r['n_trees'] for r in rows)
        m = _region_metrics(rows)
        region_row  = f"{region:<{COL_REGION}}{total_trees:>{COL_TREES}}"
        region_row += "".join(f"{m[k]:>{col_w}.4f}" for k in METRIC_KEYS)
        lines.append(region_row)

    other_rows = groups.get('Other', [])

    # ── All-regions weighted row ──────────────────────────────────────────────
    lines.append(sep)
    total_all_trees = sum(r['n_trees'] for r in scene_rows)
    m_all   = _region_metrics(scene_rows)
    all_row = f"{'ALL (weighted)':<{COL_REGION}}{total_all_trees:>{COL_TREES}}"
    all_row += "".join(f"{m_all[k]:>{col_w}.4f}" for k in METRIC_KEYS)
    lines += [all_row, sep]

    if other_rows:
        lines.append(f"  (unmatched scenes: {', '.join(r['scene'] for r in other_rows)})")

    return "\n".join(lines)


# ── Output formatting ─────────────────────────────────────────────────────────

def format_table(scene_rows: list, g: dict) -> str:
    COL_SCENE = 54
    cols  = ['Prec', 'Rec', 'F1', 'Cov', 'IoU_g', 'IoU_w', 'IoU_l', 'mIoU']
    keys  = ('prec', 'rec', 'f1', 'cov', 'iou_g', 'iou_w', 'iou_l', 'miou')
    col_w = 7
    hdr   = f"{'Scene':<{COL_SCENE}}" + "".join(f"{c:>{col_w}}" for c in cols)
    sep   = "-" * len(hdr)
    lines = [sep, hdr, sep]
    for row in scene_rows:
        line = f"{row['scene']:<{COL_SCENE}}"
        line += "".join(f"{row[k]:>{col_w}.4f}" for k in keys)
        lines.append(line)
    lines.append(sep)
    mean_line = f"{'MEAN':<{COL_SCENE}}"
    mean_line += "".join(f"{g[k]:>{col_w}.4f}" for k in keys)
    lines += [mean_line, sep]
    return "\n".join(lines)


def format_verbose(g: dict) -> str:
    scf = g['scf_g'];   scfb = g['scfb_g'];  stcf = g['stcf_g']
    icf = g['icf_g'];   scbi = g['scbi_g']
    iou_g  = g['iou_list_global']
    iou_bi = g['iou_list_bi_global']

    L = ["\n=== Global Semantic Segmentation ==="]
    L.append(f"oAcc:  {g['tpc_g'].sum() / g['pc_g'].sum():.4f}")
    L.append(f"mAcc:  {np.mean(g['tpc_g'][scf] / g['gtc_g'][scf]):.4f}")
    L.append(f"IoU:   {[f'{v:.4f}' for v in iou_g]}")
    L.append(f"mIoU:  {g['miou']:.4f}")

    L.append("\n=== Global Binary Semantic Segmentation ===")
    L.append(f"oAcc:  {g['tpc_bi'].sum() / g['pc_bi'].sum():.4f}")
    L.append(f"mAcc:  {np.mean(g['tpc_bi'][scfb] / g['gtc_bi'][scfb]):.4f}")
    L.append(f"IoU:   {[f'{v:.4f}' for v in iou_bi]}")
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
        description='Evaluate 3D forest instance segmentation predictions.')
    parser.add_argument('pred_path',
                        help='Directory containing prediction .ply files')
    parser.add_argument('--workers', type=int, default=8,
                        help='Parallel worker processes (default: 8)')
    args = parser.parse_args()

    ply_files = sorted(glob.glob(os.path.join(args.pred_path, '*.ply')))
    if not ply_files:
        print(f"No .ply files found in {args.pred_path}")
        sys.exit(1)

    n = len(ply_files)
    print(f"Evaluating {n} scenes with {args.workers} workers ...")

    results  = [None] * n
    errors   = []
    done     = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_idx = {executor.submit(process_single_file, f): i
                         for i, f in enumerate(ply_files)}
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
    output = (table
              + "\n\n=== Tree-count-weighted averages per region ===\n"
              + region_table
              + "\n" + verbose + "\n")

    if errors:
        output += "\n=== Errors ===\n"
        for path, msg in errors:
            output += f"{os.path.basename(path)}: {msg}\n"

    print("\n" + output)

    out_file = os.path.join(args.pred_path, 'evaluation_summary.txt')
    with open(out_file, 'w') as f:
        f.write(output)
    print(f"Results saved to: {out_file}")
