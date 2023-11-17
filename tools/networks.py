networks = dict(
            alexnet = dict(
            type='Recognizer2D',
            backbone=dict(
                pretrained=True,
                type="torchvision.alexnet",
                ),
            cls_head=dict(
                type='TSNHead',
                num_classes=101,
                in_channels=256,
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
            test_cfg=None)
            ,
            resnet = dict(
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
            test_cfg=None)
            ,
            convnet = dict(
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
            )
)