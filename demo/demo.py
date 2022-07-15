from pyparsing import nested_expr
from config import MobileNet, PAFPN
from mmdet.models.builder import BACKBONES
from mmdet.models.builder import NECKS
from mmdet.models.builder import DETECTORS

model_cfg = dict(
    backbone = dict(
        type='MobileNet',
        arg1 = 1,
        arg2 = 2
    ),
    neck=dict(
        type='PAFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
    )
)

backbone = dict(
    type='MobileNet',
    arg1 = 1,
    arg2 = 2
)
neck=dict(
    type='PAFPN',
    in_channels=[256, 512, 1024, 2048],
    out_channels=256,
    num_outs = 3
)


model = BACKBONES.build(backbone)


