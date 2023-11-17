from mmaction.utils import register_all_modules

register_all_modules(init_default_scope=True)
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.optim import SGD
from torch.utils.data import DataLoader
from mmaction.registry import DATASETS, MODELS
from mmengine.evaluator import BaseMetric
from mmengine.model import BaseModel
from mmengine.runner import Runner
# from tools.custom_train import SyntheticDataset
from mmengine.analysis import get_model_complexity_info
# class MMResNet50(BaseModel):
#     def __init__(self):
#         super().__init__()
#         self.resnet = torchvision.models.resnet50()

#     def forward(self, imgs, labels, mode):
#         x = self.resnet(imgs)
#         if mode == 'loss':
#             return {'loss': F.cross_entropy(x, labels)}
#         elif mode == 'predict':
#             return x, labels


# class Accuracy(BaseMetric):
#     def process(self, data_batch, data_samples):
#         score, gt = data_samples
#         self.results.append({
#             'batch_size': len(gt),
#             'correct': (score.argmax(dim=1) == gt).sum().cpu(),
#         })

#     def compute_metrics(self, results):
#         total_correct = sum(item['correct'] for item in results)
#         total_size = sum(item['batch_size'] for item in results)
#         return dict(accuracy=100 * total_correct / total_size)

# train_pipeline_cfg = [
#         dict(type='PackDistillInputs')]

# dataset_type = 'SyntheticDataset'

# dataset=dict(
#         type=dataset_type,
#         pipeline=train_pipeline_cfg,
#         ipc=1)

# ds = DATASETS.build(dataset)


# norm_cfg = dict(mean=[0.491, 0.482, 0.447], std=[0.202, 0.199, 0.201])
# train_dataloader = DataLoader(batch_size=32,
#                               shuffle=True,
#                               dataset=ds)

# val_dataloader = DataLoader(batch_size=32,
#                             shuffle=False,
#                             dataset=SyntheticDataset(pipeline=train_pipeline_cfg, ipc=50))

# runner = Runner(
#     model=MMResNet50(),
#     work_dir='./work_dir',
#     train_dataloader=train_dataloader,
#     optim_wrapper=dict(optimizer=dict(type=SGD, lr=0.001, momentum=0.9)),
#     train_cfg=dict(by_epoch=True, max_epochs=5, val_interval=1),
#     val_dataloader=val_dataloader,
#     val_cfg=dict(),
#     val_evaluator=dict(type=Accuracy),
# )
# runner.train()

model_hub = dict(
    tsn=
    dict(
    type='Recognizer2D',
    backbone=dict(
        type='ResNet',
        pretrained='https://download.pytorch.org/models/resnet50-11ad3fa6.pth',
        depth=50,
        norm_eval=False),
    cls_head=dict(
        type='TSNHead',
        num_classes=101,
        in_channels=2048,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.4,
        init_std=0.01,
        average_clips='prob'),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCHW'),
    train_cfg=None,
    test_cfg=None), 
    dense161=
    dict(
    type='Recognizer2D',
    backbone=dict(
        type='torchvision.densenet161', pretrained=True),
    cls_head=dict(
        type='TSNHead',
        num_classes=101,
        in_channels=2208,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.4,
        init_std=0.01,
        average_clips='prob'),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCHW'),
    train_cfg=None,
    test_cfg=None), 
    alexnet=dict(
    type='Recognizer2D',
    backbone=dict(
        type='torchvision.alexnet', pretrained=True),
    cls_head=dict(
        type='TSNHead',
        num_classes=101,
        in_channels=2048,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.4,
        init_std=0.01,
        average_clips='prob'),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCHW'),
    train_cfg=None,
    test_cfg=None), 
    c3d = 
    dict(
    type='Recognizer3D',
    backbone=dict(
        type='C3D',
        pretrained=  # noqa: E251
        'https://download.openmmlab.com/mmaction/recognition/c3d/c3d_sports1m_pretrain_20201016-dcc47ddc.pth',  # noqa: E501
        style='pytorch',
        conv_cfg=dict(type='Conv3d'),
        norm_cfg=None,
        act_cfg=dict(type='ReLU'),
        dropout_ratio=0.5,
        init_std=0.005),
    cls_head=dict(
        type='I3DHead',
        num_classes=101,
        in_channels=4096,
        spatial_type=None,
        dropout_ratio=0.5,
        init_std=0.01,
        average_clips='prob'),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[104, 117, 128],
        std=[1, 1, 1],
        format_shape='NCTHW'),
    train_cfg=None,
    test_cfg=None),
    c2d = 
    dict(
    type='Recognizer3D',
    backbone=dict(
        type='C2D',
        depth=50,
        pretrained='https://download.pytorch.org/models/resnet50-11ad3fa6.pth',
        norm_eval=False),
    cls_head=dict(
        type='I3DHead',
        num_classes=101,
        in_channels=2048,
        spatial_type='avg',
        dropout_ratio=0.5,
        init_std=0.01,
        average_clips='prob'),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW')),
    i3d=
    dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3d',
        pretrained2d=True,
        pretrained='torchvision://resnet50',
        depth=50,
        conv1_kernel=(5, 7, 7),
        conv1_stride_t=2,
        pool1_stride_t=2,
        conv_cfg=dict(type='Conv3d'),
        norm_eval=False,
        inflate=((1, 1, 1), (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), (0, 1, 0)),
        zero_init_residual=False),
    cls_head=dict(
        type='I3DHead',
        num_classes=101,
        in_channels=2048,
        spatial_type='avg',
        dropout_ratio=0.5,
        init_std=0.01,
        average_clips='prob'),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW')),
    r2plus1d=
    dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet2Plus1d',
        depth=34,
        pretrained=None,
        pretrained2d=False,
        norm_eval=False,
        conv_cfg=dict(type='Conv2plus1d'),
        norm_cfg=dict(type='SyncBN', requires_grad=True, eps=1e-3),
        conv1_kernel=(3, 7, 7),
        conv1_stride_t=1,
        pool1_stride_t=1,
        inflate=(1, 1, 1, 1),
        spatial_strides=(1, 2, 2, 2),
        temporal_strides=(1, 2, 2, 2),
        zero_init_residual=False),
    cls_head=dict(
        type='I3DHead',
        num_classes=101,
        in_channels=512,
        spatial_type='avg',
        dropout_ratio=0.5,
        init_std=0.01,
        average_clips='prob'),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW')),
    slowfast=
    dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3dSlowFast',
        pretrained=None,
        resample_rate=8,  # tau
        speed_ratio=8,  # alpha
        channel_ratio=8,  # beta_inv
        slow_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            lateral=True,
            conv1_kernel=(1, 7, 7),
            dilations=(1, 1, 1, 1),
            conv1_stride_t=1,
            pool1_stride_t=1,
            inflate=(0, 0, 1, 1),
            norm_eval=False),
        fast_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            lateral=False,
            base_channels=8,
            conv1_kernel=(5, 7, 7),
            conv1_stride_t=1,
            pool1_stride_t=1,
            norm_eval=False)),
    cls_head=dict(
        type='SlowFastHead',
        in_channels=2304,  # 2048+256
        num_classes=101,
        spatial_type='avg',
        dropout_ratio=0.5,
        average_clips='prob'),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW')),
    x3d=
    dict(
    type='Recognizer3D',
    backbone=dict(type='X3D', gamma_w=1, gamma_b=2.25, gamma_d=2.2),
    cls_head=dict(
        type='X3DHead',
        in_channels=432,
        num_classes=101,
        spatial_type='avg',
        dropout_ratio=0.5,
        fc1_bias=False,
        average_clips='prob'),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[114.75, 114.75, 114.75],
        std=[57.38, 57.38, 57.38],
        format_shape='NCTHW'),
    # model training and testing settings
    train_cfg=None,
    test_cfg=None) ,
    tsn_mob=
    dict(
    type='Recognizer2D',
    backbone=dict(
        type='mmpretrain.MobileOne',
        arch='s0',
        init_cfg=dict(
            type='Pretrained', checkpoint=('https://download.openmmlab.com/mmclassification/v0/mobileone/mobileone-s0_8xb32_in1k_20221110-0bc94952.pth'), prefix='backbone'),
        norm_eval=False),
    cls_head=dict(
        type='TSNHead',
        num_classes=101,
        in_channels=1024,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.4,
        init_std=0.01,
        average_clips='prob'),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCHW'),
    train_cfg=None,
    test_cfg=None) ,
    tsm=
    dict(
    type='Recognizer2D',
    backbone=dict(
        type='ResNetTSM',
        pretrained='torchvision://resnet50',
        depth=50,
        norm_eval=False,
        shift_div=8),
    cls_head=dict(
        type='TSMHead',
        num_classes=101,
        in_channels=2048,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.5,
        init_std=0.001,
        is_shift=True,
        average_clips='prob'),
    data_preprocessor=dict(type='ActionDataPreprocessor', **dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])),
    train_cfg=None,
    test_cfg=None),
    tanet=
    dict(
    type='Recognizer2D',
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.5],
        std=[58.395, 57.12, 57.375],
        format_shape='NCHW'),
    backbone=dict(
        type='TANet',
        pretrained='torchvision://resnet50',
        depth=50,
        num_segments=8,
        tam_cfg=None),
    cls_head=dict(
        type='TSMHead',
        num_classes=101,
        in_channels=2048,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.5,
        init_std=0.001,
        average_clips='prob')),
    tin=
    dict(
    type='Recognizer2D',
    backbone=dict(
        type='ResNetTIN',
        pretrained='torchvision://resnet50',
        depth=50,
        norm_eval=False,
        shift_div=4),
    cls_head=dict(
        type='TSMHead',
        num_classes=101,
        in_channels=2048,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.5,
        init_std=0.001,
        is_shift=False,
        average_clips='prob'),
    data_preprocessor=dict(type='ActionDataPreprocessor', **dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    format_shape='NCHW')),
    # model training and testing settings
    train_cfg=None,
    test_cfg=None),
    trn=
    dict(
    type='Recognizer2D',
    backbone=dict(
        type='ResNet',
        pretrained='torchvision://resnet50',
        depth=50,
        norm_eval=False,
        partial_bn=True),
    cls_head=dict(
        type='TRNHead',
        num_classes=101,
        in_channels=2048,
        num_segments=8,
        spatial_type='avg',
        relation_type='TRNMultiScale',
        hidden_dim=256,
        dropout_ratio=0.8,
        init_std=0.001,
        average_clips='prob'),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCHW')),
    convnet=
    dict(
    type='Recognizer2D',
    backbone=dict(
        type="ConvNet",
        channel=3,
        num_classes=101,
        net_act='relu',
        net_norm='batchnorm',
        net_width=128,
        net_depth=5,
        net_pooling='avgpooling'
    ),
    cls_head=dict(
        type='TSNHead',
        num_classes=101,
        in_channels=128,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.4,
        init_std=0.01,
        average_clips='prob'),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCHW'),
    train_cfg=None,
    test_cfg=None
    ),
    res18=
    dict(
    type='Recognizer2D',
    backbone=dict(
        type="ResNet",
        depth=18,
        norm_eval=False
    ),
    cls_head=dict(
        type='TSNHead',
        num_classes=101,
        in_channels=512,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.4,
        init_std=0.01,
        average_clips='prob'),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCHW'),
    train_cfg=None,
    test_cfg=None
    ),
    vgg11=
    dict(
    type='Recognizer2D',
    backbone=dict(
        type="VGG",
        vgg_name="VGG11", 
        channel=3,  
        norm='batchnorm', 
        res=64
    ),
    cls_head=dict(
        type='TSNHead',
        num_classes=101,
        in_channels=512,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.4,
        init_std=0.01,
        average_clips='prob'),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCHW'),
    train_cfg=None,
    test_cfg=None
    )
     
)

input_hub = dict(c3d=(1,3,16,112,112), 
                 tsn=(1,3,224,224), 
                 tsn_mob=(1,3,224,224),
                 tanet=(1,3,224,224),
                 tin=(8,3,224,224),
                 trn=(1,3,224,224),
                 tsm=(8,3,224,224),
                 c2d=(1,3,2,224,224),
                 i3d=(1,3,5,224,224),
                 r2plus1d=(1,3,1,224,224),
                 slowfast=(1,3,32,224,224),
                 x3d=(1,3,1,182,182),
                 convnet=(1,3,224,224),
                 res18=(1,3,224,224),
                 vgg11=(1,3,224,224),
                 dense161=(1,3,224,224),
                 alexnet=(1,3,224,224)
)

model = input()

m = MODELS.build(model_hub[model])
analysis = get_model_complexity_info(m, input_hub[model])
print("{} Model Flops: {}".format(model,analysis['flops_str']))
print("{} Model Parameters: {}".format(model,analysis['params_str']))
