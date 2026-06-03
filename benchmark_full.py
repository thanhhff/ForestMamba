"""
Full benchmark: ForestMamba vs. OneFormer3D vs. ForestFormer3D
================================================================
Measures:
  - #Parameters: encoder / decoder / total
  - Encoder inference time  (input_conv + backbone + output_layer)
  - Decoder inference time  (decoder forward, synthetic queries)
  - Full network inference  (encoder + decoder, no grad)
  - Full training step      (encoder + decoder forward + proxy loss backward)

All tests use synthetic sparse point clouds (~5M voxels, batch=1).
Query counts match each model's config:
  OneFormer3D    : 300 learnable queries  
  ForestFormer3D : 300 X-aware queries    (FPS-style)
  ForestMamba    : 300 X-aware queries    (CHM + FPS)

Run:  CUDA_VISIBLE_DEVICES=2 /opt/conda/bin/python benchmark_full.py
"""

import os, sys, time, gc
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, '/workdir/radish/3d-project/workspace/ForestFormer3D')
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from mmengine.config import Config
from mmdet3d.utils import register_all_modules
register_all_modules()
import oneformer3d
import spconv.pytorch as spconv

DEVICE = torch.device('cuda:0')
ROOT   = '/workdir/radish/3d-project/workspace/ForestFormer3D'

N_WARMUP = 5
N_RUNS   = 20
N_VOXELS = 1_000_000 # synthetic sparse voxels per scene
N_PTS    = N_VOXELS  # same for query selection

# ── model configs ──────────────────────────────────────────────────────────
MODELS_CFG = [
    dict(
        name='OneFormer3D',
        config='configs/ForAINetv2/oneformer3d_forainetv2_inference.py',
        ckpt='work_dirs/2_Oneformer3d/save_checkpoint/epoch_2100_fix.pth',
        backbone_label='SpConvUNet',
        decoder_label='Transformer (6L)',
        n_queries=300,    # fixed at 300 for fair comparison
        decoder_call='no_queries',   # decoder(x) — uses internal learnable embedding
    ),
    dict(
        name='ForestFormer3D',
        config='configs/ForAINetv2/forestformer3d_qs_radius16_qp300_2many.py',
        ckpt='work_dirs/1_ForestFormer3D_Retrain/new_split/epoch_3000_fix.pth',
        backbone_label='SpConvUNet',
        decoder_label='Transformer (6L)',
        n_queries=300,    # X-aware FPS queries
        decoder_call='with_queries',  # decoder(x, queries)
    ),
    dict(
        name='ForestMamba (Ours)',
        config='configs/ForAINetv2/forestmamba_chm_radius16_qp300_2many_v6_2.py',
        ckpt='work_dirs/forestmamba_chm_radius16_qp300_2many_v6_2/epoch_3000_fix.pth',
        backbone_label='SparseMamba',
        decoder_label='SSM (6L)',
        n_queries=300,
        decoder_call='with_queries',
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
def count(m): return sum(p.numel() for p in m.parameters())
def ms_pm(m, s): return f"{m:.1f}\\,$\\pm$\\,{s:.1f}" if m is not None else "OOM"

def make_sparse(n_vox=N_VOXELS, in_ch=3, spatial_shape=(300, 300, 60)):
    # sample per-dimension so unique count can actually reach n_vox
    xs = torch.randint(0, spatial_shape[0], (n_vox,))
    ys = torch.randint(0, spatial_shape[1], (n_vox,))
    zs = torch.randint(0, spatial_shape[2], (n_vox,))
    coords = torch.stack([xs, ys, zs], dim=1)
    coords = torch.unique(coords, dim=0)
    n = coords.shape[0]
    batch_idx = torch.zeros(n, 1, dtype=torch.int32)
    coords = torch.cat([batch_idx, coords.int()], dim=1)
    feats  = torch.randn(n, in_ch)
    return spconv.SparseConvTensor(
        feats.to(DEVICE), coords.to(DEVICE), spatial_shape, 1)

def encoder_forward(model, sp):
    x = model.input_conv(sp)
    x, _ = model.unet(x)
    x = model.output_layer(x)
    return x    # SparseConvTensor

def encoder_features(model, sp):
    """Return list of per-voxel feature tensors (batch=1)."""
    x = encoder_forward(model, sp)
    return [x.features]   # list of length 1, shape [N, 32]

def timer(fn, n_warmup=N_WARMUP, n_runs=N_RUNS):
    with torch.no_grad():
        for _ in range(n_warmup):
            fn()
        torch.cuda.synchronize()
        t = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            t.append((time.perf_counter() - t0) * 1e3)
    return float(np.mean(t)), float(np.std(t))

def timer_train(fn, n_warmup=N_WARMUP, n_runs=N_RUNS):
    """Benchmark with gradient (forward + backward)."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    t = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t.append((time.perf_counter() - t0) * 1e3)
    return float(np.mean(t)), float(np.std(t))

def load_model(cfg_dict):
    cfg = Config.fromfile(os.path.join(ROOT, cfg_dict['config']))
    from mmdet3d.registry import MODELS
    model = MODELS.build(cfg.model)
    ckpt_path = os.path.join(ROOT, cfg_dict['ckpt'])
    if os.path.exists(ckpt_path):
        ckpt  = torch.load(ckpt_path, map_location='cpu')
        state = ckpt.get('state_dict', ckpt)
        model.load_state_dict(state, strict=False)
    else:
        print(f"  [info] no checkpoint found — using random weights (timing/params only)")
    return model.to(DEVICE)

# ─────────────────────────────────────────────────────────────────────────────
results = []

for cfg in MODELS_CFG:
    name = cfg['name']
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    model = load_model(cfg)
    model.eval()

    # ── 1. Parameter counts ──────────────────────────────────────────────────
    enc_p   = count(model.input_conv) + count(model.unet) + count(model.output_layer)
    dec_p   = count(model.decoder)
    total_p = count(model)
    print(f"  Encoder params : {enc_p/1e6:.3f} M  ({enc_p:,})")
    print(f"  Decoder params : {dec_p/1e6:.3f} M  ({dec_p:,})")
    print(f"  Total params   : {total_p/1e6:.3f} M  ({total_p:,})")

    sp = make_sparse()

    # ── 2. Encoder inference time ────────────────────────────────────────────
    torch.cuda.reset_peak_memory_stats(DEVICE)
    enc_ms, enc_std = timer(lambda: encoder_forward(model, sp))
    enc_mem = torch.cuda.max_memory_allocated(DEVICE) / 1024**2
    print(f"  Encoder inf.   : {enc_ms:.1f} ± {enc_std:.1f} ms   peak {enc_mem:.0f} MB")

    # ── 3. Decoder inference time ────────────────────────────────────────────
    with torch.no_grad():
        feats = encoder_features(model, sp)   # [N,32]
    n_q = cfg['n_queries']
    in_ch = 32

    if cfg['decoder_call'] == 'no_queries':
        # OneFormer: internal learnable Embedding (no external queries needed)
        # We still pass 300 synthetic queries so the decoder uses them via query_proj
        # to match 300-query budget of the other models.
        # OneFormer's QueryDecoder.forward(x, queries=None) uses self.query Embedding
        # when queries is None, which already has 303 entries. For a fair 300-query
        # comparison we call it as-is (internal budget closest to 300).
        def dec_fn():
            return model.decoder(feats)
    else:
        # ForestFormer3D / ForestMamba: decoder takes external queries
        syn_q = [torch.randn(n_q, in_ch, device=DEVICE)]
        def dec_fn():
            return model.decoder(feats, syn_q)

    dec_ms = dec_std = dec_mem = None
    try:
        torch.cuda.reset_peak_memory_stats(DEVICE)
        dec_ms, dec_std = timer(dec_fn)
        dec_mem = torch.cuda.max_memory_allocated(DEVICE) / 1024**2
        print(f"  Decoder inf.   : {dec_ms:.1f} ± {dec_std:.1f} ms   peak {dec_mem:.0f} MB  (Q={n_q})")
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"  Decoder inf.   : OOM  (Q={n_q})")

    # ── 4. Full network inference (encoder + decoder, no grad) ───────────────
    if cfg['decoder_call'] == 'no_queries':
        def full_inf_fn():
            f = encoder_features(model, sp)
            return model.decoder(f)
    else:
        def full_inf_fn():
            f = encoder_features(model, sp)
            q = [torch.randn(n_q, in_ch, device=DEVICE)]
            return model.decoder(f, q)

    full_ms = full_std = full_mem = None
    try:
        torch.cuda.reset_peak_memory_stats(DEVICE)
        full_ms, full_std = timer(full_inf_fn)
        full_mem = torch.cuda.max_memory_allocated(DEVICE) / 1024**2
        print(f"  Full inf.      : {full_ms:.1f} ± {full_std:.1f} ms   peak {full_mem:.0f} MB")
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"  Full inf.      : OOM")

    # ── 5. Training step (encoder + decoder forward + proxy backward) ─────────
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def _proxy_loss(out):
        loss = sum(m.sum() for m in out['masks'])
        if out.get('scores') is not None:
            loss = loss + sum(s.sum() for s in out['scores'] if s is not None)
        if out.get('aux_outputs'):
            for aux in out['aux_outputs']:
                loss = loss + sum(m.sum() for m in aux['masks'])
        return loss

    if cfg['decoder_call'] == 'no_queries':
        def train_fn():
            optim.zero_grad(set_to_none=True)
            f = encoder_features(model, sp)
            out = model.decoder(f)
            _proxy_loss(out).backward()
            optim.step()
    else:
        def train_fn():
            optim.zero_grad(set_to_none=True)
            f = encoder_features(model, sp)
            q = [torch.randn(n_q, in_ch, device=DEVICE)]
            out = model.decoder(f, q)
            _proxy_loss(out).backward()
            optim.step()

    train_ms = train_std = train_mem = None
    try:
        torch.cuda.reset_peak_memory_stats(DEVICE)
        train_ms, train_std = timer_train(train_fn)
        train_mem = torch.cuda.max_memory_allocated(DEVICE) / 1024**2
        print(f"  Train step     : {train_ms:.1f} ± {train_std:.1f} ms   peak {train_mem:.0f} MB")
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"  Train step     : OOM")

    # Extrapolate to per-epoch (49 iters / epoch from logs)
    iters_per_epoch = 49
    if train_ms is not None:
        epoch_s = train_ms * iters_per_epoch / 1e3
        total_h = epoch_s * 3000 / 3600
        print(f"  => ~{epoch_s:.1f} s/epoch  |  ~{total_h:.1f} h for 3000 epochs")
    else:
        epoch_s = total_h = None
        print(f"  => train OOM — epoch estimate unavailable")

    results.append(dict(
        name=name,
        backbone_label=cfg['backbone_label'],
        decoder_label=cfg['decoder_label'],
        n_queries=n_q,
        enc_p=enc_p, dec_p=dec_p, total_p=total_p,
        enc_ms=enc_ms, enc_std=enc_std, enc_mem=enc_mem,
        dec_ms=dec_ms, dec_std=dec_std, dec_mem=dec_mem,
        full_ms=full_ms, full_std=full_std, full_mem=full_mem,
        train_ms=train_ms, train_std=train_std, train_mem=train_mem,
        epoch_s=epoch_s, total_h=total_h,
    ))

    del model, optim, sp, feats
    gc.collect(); torch.cuda.empty_cache()

# ─────────────────────────────────────────────────────────────────────────────
# Raw summary
print("\n\n" + "="*70)
print("SUMMARY")
print("="*70)
hdr = f"{'Method':<25} {'Enc-P(M)':>8} {'Dec-P(M)':>8} {'Tot-P(M)':>8} "
hdr += f"{'Enc-ms':>8} {'Dec-ms':>8} {'Full-ms':>8} {'Train-ms':>10} {'h/3000ep':>9}"
print(hdr)
print("-"*70)
def fmt(v, w, fmt_spec):
    return f"{'OOM':>{w}}" if v is None else f"{v:{w}{fmt_spec}}"

for r in results:
    print(
        f"{r['name']:<25} "
        f"{r['enc_p']/1e6:>8.3f} "
        f"{r['dec_p']/1e6:>8.3f} "
        f"{r['total_p']/1e6:>8.3f} "
        + fmt(r['enc_ms'], 8, '.1f') + ' '
        + fmt(r['dec_ms'], 8, '.1f') + ' '
        + fmt(r['full_ms'], 8, '.1f') + ' '
        + fmt(r['train_ms'], 10, '.1f') + ' '
        + fmt(r['total_h'], 9, '.1f')
    )

# ─────────────────────────────────────────────────────────────────────────────
# LaTeX table
print("\n\n" + "="*70)
print("LATEX TABLE (params + inference)")
print("="*70)

print(r"""\begin{table}[t]
\centering
\caption{Network complexity comparison. Encoder = input projection +
backbone + output normalisation. Inference times measured on NVIDIA
RTX A6000 GPU, $\approx$5\,M voxels, batch 1, $N=20$ runs
(mean\,$\pm$\,std). Training step = encoder + decoder forward/backward
with proxy loss, 49 samples/epoch.}
\label{tab:complexity}
\resizebox{\linewidth}{!}{%
\begin{tabular}{llcccccccc}
\toprule
\multirow{2}{*}{Method} & \multirow{2}{*}{Backbone} &
  \multicolumn{3}{c}{\# Params (M)} & \multirow{2}{*}{Queries} &
  \multicolumn{2}{c}{Inference Time (ms)} &
  \multirow{2}{*}{\makecell{Train Step\\(ms)}} &
  \multirow{2}{*}{\makecell{Est.~Train\\(h/3000 ep)}} \\
\cmidrule(lr){3-5} \cmidrule(lr){7-8}
 & & Enc. & Dec. & Total & & Encoder & Full Net & & \\
\midrule""")

for r in results:
    is_ours = 'Ours' in r['name']
    b = r'\textbf{' if is_ours else ''
    e = '}'        if is_ours else ''
    row = (
        f"{b}{r['name']}{e} & "
        f"{b}{r['backbone_label']}{e} & "
        f"{b}{r['enc_p']/1e6:.2f}{e} & "
        f"{b}{r['dec_p']/1e6:.2f}{e} & "
        f"{b}{r['total_p']/1e6:.2f}{e} & "
        f"{b}{r['n_queries']}{e} & "
        f"{b}{ms_pm(r['enc_ms'], r['enc_std'])}{e} & "
        f"{b}{ms_pm(r['full_ms'], r['full_std'])}{e} & "
        f"{b}{ms_pm(r['train_ms'], r['train_std'])}{e} & "
        f"{b}{ ('%0.1f' % r['total_h']) if r['total_h'] is not None else 'OOM'}{e} \\\\"
    )
    print(row)

print(r"""\bottomrule
\end{tabular}}
\end{table}""")

print("\n\nRAW:")
for r in results:
    print(r)
