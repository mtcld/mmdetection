_base_ = '../htc/htc_r50_fpn_1x_coco.py'

model = dict(
    backbone=dict(
        type='DetectoRS_ResNet',
        conv_cfg=dict(type='ConvAWS'),
        sac=dict(type='SAC', use_deform=True),
        stage_with_sac=(False, True, True, True),
        output_img=True),
    neck=dict(
        type='RFP',
        rfp_steps=2,
        aspp_out_channels=64,
        aspp_dilations=(1, 3, 6, 1),
        rfp_backbone=dict(
            rfp_inplanes=256,
            type='DetectoRS_ResNet',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            conv_cfg=dict(type='ConvAWS'),
            sac=dict(type='SAC', use_deform=True),
            stage_with_sac=(False, True, True, True),
            pretrained='torchvision://resnet50',
            style='pytorch')))


classes=['dent']
dataset_type = 'CocoDataset'
data_root = '/mmdetection/data/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations', with_bbox=True, with_mask=True, with_seg=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='SegRescale', scale_factor=0.125),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg'])
]

dataset_merimen_train = dict(
    type='RepeatDataset',
    times=2,
    dataset=dict(
        type=dataset_type,
        classes=classes,
        test_mode=False,
        ann_file=data_root+'merimen_coco_30_7_mask/dent/annotations/total.json',
        img_prefix=data_root+'merimen_coco_30_7_mask/dent/images/',
        pipeline=train_pipeline,
        seg_prefix=data_root+'merimen_coco_30_7_mask/dent/mask_total'
    )
)

data = dict(
    samples_per_gpu=3,
    workers_per_gpu=2,
    train=[dict(
        type=dataset_type,
        classes=classes,
        test_mode=False,
        ann_file=data_root + 'dent/annotations/train.json',
        img_prefix=data_root + 'dent/images/',
        pipeline=train_pipeline,
        seg_prefix=data_root+'dent/mask_train/'),
        dataset_merimen_train],
    val=dict(
        type=dataset_type,
        classes=classes,
        test_mode=False,
        ann_file=data_root + 'dent/annotations/valid.json',
        #worflow = [('train', 1), ('val', 1)],
        img_prefix=data_root + 'dent/images/',
        seg_prefix=data_root+'dent/mask_valid/'),
    test=dict(
        type=dataset_type,
        classes=classes,
        test_mode=False,
        #worflow = [('train', 1), ('val', 1)],
        ann_file=data_root + 'annotations/test.json',
        img_prefix=data_root + 'images/',
        seg_prefix=data_root+'masks_valid_test/'))
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(_delete_=True,grad_clip=dict(max_norm=35, norm_type=2))
# # learning policy
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=1.0 / 3,
#     step=[8, 11])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
evaluation = dict(interval=1)
# runtime settings
total_epochs = 13
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/pseudo_dent'
load_from = None
resume_from = None
#workflow = [('train', 1)]
workflow = [('train', 1)]
