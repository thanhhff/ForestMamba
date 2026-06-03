_base_ = [
    'mmdet3d::_base_/default_runtime.py',
]
custom_imports = dict(imports=['oneformer3d'])

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl', timeout=14400),
)

# model settings
num_channels = 32
num_instance_classes = 3
num_semantic_classes = 3
radius = 16       # cylindrical crop radius (m)
score_th = 0.2    # lowered from 0.4: accept minor precision drop to recover missed trees
chunk = 20_000

# ── C2: CHM-Guided Instance Query Initialization ─────────────────────────────
# Instead of FPS on backbone embeddings, query seeds are placed at local maxima
# of the on-the-fly Canopy Height Model (CHM) built from tree-class voxels.
# Local maxima in the 2-D height map correspond to individual tree tops, giving
# ecologically meaningful and spatially diverse query initialisation.
#
# Key hyper-parameters:
#   chm_resolution   – XY grid cell size (m). Smaller = finer peak detection.
#   chm_min_height   – Minimum Z value for a cell to count as a treetop.
#   chm_local_window – Half-size of local-max filter (window = 2*w+1 cells).
#
# All other settings are identical to the C1 baseline config
# (forestmamba_qs_radius16_qp300_2many.py).
# ─────────────────────────────────────────────────────────────────────────────

model = dict(
    type='ForAINetV2OneFormer3D_CHMquery',
    data_preprocessor=dict(type='Det3DDataPreprocessor'),
    in_channels=3,  # XYZ only — geometric augmentation is train-only
    num_channels=num_channels,
    voxel_size=0.2,
    num_classes=num_instance_classes,
    min_spatial_shape=128,
    stuff_classes=[0],
    thing_cls=[1, 2],
    prepare_epoch=20, #-1, # warm up BiSemantic for 200 epochs before enabling CHM
    query_point_num=300,         # kept at 300; topk_insts raised to 400 in test_cfg
    radius=radius,
    score_th=score_th,
    # ── C2 CHM-Guided Query Initialization parameters ──────────────────────
    # Step 1 — CHM rasterization
    chm_resolution=0.5,        # 0.5 m XY grid: fine enough to resolve tree tops
    chm_min_height=1.5,        # lowered from 2.0: dataset P10 height = 2.75 m,
                               # 1.5 m captures ~10% more short trees while
                               # still filtering most ground/shrub returns
    # Step 2 — Allometric local-maximum detection: w(h) = a * h^b
    # Dataset: avg inter-tree spacing ~3.9 m; old a=0.5,b=0.6 gave w=3.9 m for
    # 25 m trees — suppressing nearly every neighbour. New values halve the radius.
    chm_allometric_a=0.25,     # halved from 0.5: reduces suppression radius
    chm_allometric_b=0.5,      # lowered from 0.6: slower growth for tall trees
                               # new w for 25 m tree: 0.25*25^0.5 = 1.25 m (was 3.9 m)
    # Step 3 — Cylinder feature pooling around each detected treetop
    chm_pool_radius=1.5,       # raised from 0.5: dataset median crown radius ~1-2 m;
                               # larger cylinder captures full crown context
    # Step 4 — Remaining budget filled by FPS from tree voxels (same as C1)
    backbone=dict(
        type='SparseMambaEncoder',
        num_planes=[num_channels * (i + 1) for i in range(5)],
        return_blocks=True,
        d_state=16,
        d_conv=4,
        expand=2,
        slab_thickness=5),
    decoder=dict(
        type='ForAINetv2QueryDecoder_XAwarequery',
        num_layers=6,
        num_classes=1,
        num_instance_queries=0,
        num_semantic_queries=num_semantic_classes,
        num_instance_classes=num_instance_classes,
        in_channels=32,
        d_model=256,
        num_heads=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn='gelu',
        iter_pred=True,
        attn_mask=True,
        fix_attention=True,
        objectness_flag=True),
    criterion=dict(
        type='ForAINetv2UnifiedCriterion_XAwarequery',
        num_semantic_classes=num_semantic_classes,
        sem_criterion=dict(
            type='S3DISSemanticCriterion',
            loss_weight=0.2,
            class_weight=[1.0, 2.0, 1.0]),  # ground / wood (boosted) / leaf
        inst_criterion=dict(
            type='InstanceCriterionForAI_OneToManyMatch',
            matcher=dict(
                type='One2ManyMatcher'),
            loss_weight=[1.0, 1.0, 0.5],
            fix_dice_loss_weight=True,
            iter_matcher=True,
            fix_mean_loss=True)),
    train_cfg=dict(),
    test_cfg=dict(
        topk_insts=300, 
        inst_score_thr=0.0,
        pan_score_thr=0.0,
        npoint_thr=10,
        obj_normalization=True,
        obj_normalization_thr=0.01,
        sp_score_thr=0.15,
        nms=True,
        matrix_nms_kernel='linear',
        num_sem_cls=num_semantic_classes,
        stuff_cls=[0],
        thing_cls=[0]))

# dataset settings
dataset_type = 'ForAINetV2SegDataset_'
data_root_forainetv2 = '/home_ssd/nguyent/ForAINetV2/'
# data_root_forainetv2 = '/home_ssd/ForAINetV2/'
data_prefix = dict(
    pts='points',
    pts_instance_mask='instance_mask',
    pts_semantic_mask='semantic_mask')

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=False,
        load_dim=3,
        use_dim=[0, 1, 2]),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=True,
        with_seg_3d=True),
    dict(type='CylinderCrop', radius=radius),
    dict(type='GridSample', grid_size=0.2),
    dict(
        type='PointSample_',
        num_points=640000),
    dict(type='GeometricFeatureAugmentation',
         k=16,
         linearity_thr=0.7,
         verticality_thr=0.6,
         oversample_ratio=2.0,
         jitter_std=0.05),
    dict(type='SkipEmptyScene_'),
    dict(type='PointInstClassMapping_',
         num_classes=num_instance_classes),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.0),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-3.14, 3.14],
        scale_ratio_range=[0.8, 1.2],
        translation_std=[0.1, 0.1, 0.1],
        shift_height=False),
    dict(
        type='Pack3DDetInputs_',
        keys=[
            'points', 'gt_labels_3d', 'pts_semantic_mask', 'pts_instance_mask',
            'ratio_inspoint', 'vote_label', 'instance_mask'
        ])
]
val_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=False,
        load_dim=3,
        use_dim=[0, 1, 2]),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=True,
        with_seg_3d=True),
    dict(type='CylinderCrop', radius=radius),
    dict(type='GridSample', grid_size=0.2),
    dict(
        type='PointSample_',
        num_points=640000),
    dict(type='PointInstClassMapping_',
         num_classes=num_instance_classes),
    dict(type='Pack3DDetInputs_',
         keys=['points', 'gt_labels_3d', 'pts_semantic_mask', 'pts_instance_mask',
               'instance_mask'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=False,
        load_dim=3,
        use_dim=[0, 1, 2]),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=True,
        with_seg_3d=True),
    dict(type='Pack3DDetInputs_',
         keys=['points', 'gt_labels_3d', 'pts_semantic_mask', 'pts_instance_mask',
               'instance_mask'])
]

# run settings
train_dataloader = dict(
    batch_size=1,
    num_workers=8,
    prefetch_factor=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root_forainetv2,
        ann_file='forainetv2_oneformer3d_infos_train.pkl',
        data_prefix=data_prefix,
        pipeline=train_pipeline,
        filter_empty_gt=True,
        box_type_3d='Depth',
        backend_args=None))
val_dataloader = dict(
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root_forainetv2,
        ann_file='forainetv2_oneformer3d_infos_val.pkl',
        data_prefix=data_prefix,
        pipeline=val_pipeline,
        box_type_3d='Depth',
        test_mode=True,
        backend_args=None))
test_dataloader = dict(
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root_forainetv2,
        ann_file='forainetv2_oneformer3d_infos_test.pkl',
        data_prefix=data_prefix,
        pipeline=test_pipeline,
        box_type_3d='Depth',
        test_mode=True,
        backend_args=None))

class_names = ['ground', 'wood', 'leaf']
label2cat = {i: name for i, name in enumerate(class_names)}
metric_meta = dict(
    label2cat=label2cat,
    ignore_index=[],
    classes=class_names,
    dataset_name='ForAINetV2')

sem_mapping = [0, 1, 2]
inst_mapping = sem_mapping[1:]
val_evaluator = dict(
    type='UnifiedSegMetric',
    stuff_class_inds=[0],
    thing_class_inds=list(range(1, num_semantic_classes)),
    min_num_points=1,
    id_offset=2**16,
    sem_mapping=sem_mapping,
    inst_mapping=inst_mapping,
    metric_meta=metric_meta)
test_evaluator = val_evaluator

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.05),
    clip_grad=dict(max_norm=10, norm_type=2))

param_scheduler = dict(type='PolyLR', begin=0, end=450000, power=0.9, by_epoch=False)

# custom_hooks = [dict(type='EmptyCacheHook', after_iter=True)]
custom_hooks = []
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=50,
        max_keep_ckpts=10,
        save_optimizer=True),
    logger=dict(type='LoggerHook', interval=20),
    visualization=dict(type='Det3DVisualizationHook', draw=False))

vis_backends = [dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=3000,
    val_interval=50)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
find_unused_parameters = True
