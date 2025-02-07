from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple, Union

import mmengine
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.logging import MMLogger
from mmengine.model import BaseModule
from mmengine.runner.checkpoint import _load_checkpoint
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm
from torch.utils import checkpoint as cp

from mmaction.registry import MODELS
from mmaction.utils import ConfigType

''' ConvNet '''
@MODELS.register_module()
class ConvNet(BaseModule):
    def __init__(self, 
                 channel, 
                 num_classes, 
                 net_width, 
                 net_depth, 
                 net_act, 
                 net_norm, 
                 net_pooling, 
                 im_size = (224,224),
                 init_cfg: Optional[Union[Dict, List[Dict]]] = [
                        dict(type='Kaiming', layer='Conv2d'),
                        dict(type='Constant', layer='BatchNorm2d', val=1.)
                    ]
                 ):
        super().__init__(init_cfg=init_cfg)

        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size)
        num_feat = shape_feat[0]*shape_feat[1]*shape_feat[2]
        # self.classifier = nn.Linear(num_feat, num_classes)

    def forward(self, x):
        # resnet out: [bs, 2048, 7, 7]
        out = self.features(x) # [bs, channel, 224,224]
        # [bs, channel*224*224]
        # out = out.view(out.size(0), -1)
        # out = self.classifier(out)
        return out

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        else:
            exit('unknown activation function: %s'%net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            # return nn.MaxPool2d(kernel_size=2, stride=2)
            return nn.MaxPool2d(kernel_size=3, stride=2,padding=1)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s'%net_pooling)
    
    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c*h*w)
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s'%net_norm)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
        layers = []
        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers += [nn.Conv2d(in_channels, net_width, kernel_size=3, padding=3 if channel == 1 and d == 0 else 1)]
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self._get_activation(net_act)]
            in_channels = net_width
            if net_pooling != 'none':
                layers += [self._get_pooling(net_pooling)]
                shape_feat[1] //= 2
                shape_feat[2] //= 2


        return nn.Sequential(*layers), shape_feat
    
