_base_ = [
    '../../_base_/models/r2plus1d_r34.py', '../../_base_/default_runtime.py'
]


model = dict(
    cls_head=dict(
        type='I3DHead',
        num_classes=101))

# dataset settings
dataset_type = 'RawframeDataset'
data_root = './data/ucf101/rawframes'
data_root_val = './data/ucf101/rawframes'
split = 1  # official train/test splits. valid numbers: 1, 2, 3
ann_file_train = f'./data/ucf101/ucf101_train_split_{split}_rawframes.txt'
ann_file_val = f'./data/ucf101/ucf101_val_split_{split}_rawframes.txt'
ann_file_test = f'./data/ucf101/ucf101_val_split_{split}_rawframes.txt'

file_client_args = dict(io_backend='disk')
train_pipeline = [
    # dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=1),
    # dict(type='DecordDecode'),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    # dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=8,
        num_clips=1,
        test_mode=True),
    # dict(type='DecordDecode'),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
# test_pipeline = [
#     # dict(type='DecordInit', **file_client_args),
#     dict(
#         type='SampleFrames',
#         clip_len=8,
#         frame_interval=8,
#         num_clips=10,
#         test_mode=True),
#     # dict(type='DecordDecode'),
#     dict(type='Resize', scale=(-1, 256)),
#     dict(type='ThreeCrop', crop_size=256),
#     dict(type='FormatShape', input_format='NCTHW'),
#     dict(type='PackActionInputs')
# ]
train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(img=data_root),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=32,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(img=data_root_val),
        pipeline=val_pipeline,
        test_mode=True))
# test_dataloader = dict(
#     batch_size=1,
#     num_workers=8,
#     persistent_workers=True,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         ann_file=ann_file_test,
#         data_prefix=dict(video=data_root_val),
#         pipeline=test_pipeline,
#         test_mode=True))

val_evaluator = dict(type='AccMetric')
# test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=50,  # change from 100 to 50
    val_begin=1,
    val_interval=1)
val_cfg = dict(type='ValLoop')
# test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=50,  # change from 100 to 50
        by_epoch=True,
        milestones=[20, 40],  # change milestones
        gamma=0.1)
]

optim_wrapper = dict(
    optimizer=dict(
        type='SGD',
        lr=0.005, # change from 0.01 to 0.005
        momentum=0.9,
        weight_decay=0.0001),
    clip_grad=dict(max_norm=40, norm_type=2))


default_hooks = dict(checkpoint=dict(max_keep_ckpts=3))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (8 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=256)
