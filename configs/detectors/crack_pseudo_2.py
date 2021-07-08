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


classes=['crack']
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
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=[dict(
        type=dataset_type,
        classes=classes,
        test_mode=False,
        ann_file=data_root + 'crack_train/annotations/train_merge_2.json',
        img_prefix=data_root + 'crack_train/images/',
        pipeline=train_pipeline,
        seg_prefix=data_root+'crack_train/masks_train/'),
        dict(
        type=dataset_type,
        classes=classes,
        test_mode=False,
        ann_file=data_root + 'merimen_coco_15_6_aug/crack/annotations/pseudo_total_aug.json',
        img_prefix=data_root + 'merimen_coco_15_6_aug/crack/images/',
        pipeline=train_pipeline,
        seg_prefix=data_root+'merimen_coco_15_6_aug/crack/masks_pseudo_total_aug/')],
    val=dict(
        type=dataset_type,
        classes=classes,
        test_mode=False,
        ann_file=data_root + 'crack_train/annotations/valid.json',
        #worflow = [('train', 1), ('val', 1)],
        img_prefix=data_root + 'crack_train/images/',
        seg_prefix=data_root+'crack_train/masks_valid_test/'),
    test=dict(
        type=dataset_type,
        classes=classes,
        test_mode=False,
        #worflow = [('train', 1), ('val', 1)],
        ann_file=data_root + 'crack_train/annotations/test.json',
        img_prefix=data_root + 'crack_train/images/',
        seg_prefix=data_root+'crack_train/masks_valid_test/'))
# optimizer
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
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
total_epochs = 50
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/pseudo_crack'
load_from = None
resume_from = None
#workflow = [('train', 1)]
workflow = [('train', 1)]