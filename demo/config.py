from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry
import torch.nn as nn
from mmdet.models.builder import BACKBONES as MMDET_BACKBONES
from mmdet.models.builder import DETECTORS as MMDET_DETECTORS
from mmdet.models.builder import NECKS as MMDET_NECKS
MODELS = Registry('model', parent=MMCV_MODELS)

@MMDET_BACKBONES.register_module()
class MobileNet(nn.Module):

    def __init__(self, arg1, arg2):
        pass

    def forward(self, x):  # should return a tuple
        pass

@MODELS.register_module()
class PAFPN(nn.Module):

    def __init__(self,
                in_channels,
                out_channels,
                num_outs,
                start_level=0,
                end_level=-1,
                add_extra_convs=False):
        pass

    def forward(self, inputs):
        # implementation is ignored
        pass


