import torch.nn as nn
from mmdet3d.registry import MODELS
from mmdet3d.utils import OptMultiConfig
from mmengine.model import BaseModule
from torch import Tensor


@MODELS.register_module()
class UpsampleHead(BaseModule):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 scale_factor: int,
                 mode: str = 'bilinear',
                 align_corners: bool = True,
                 init_cfg: OptMultiConfig = None) -> None:
        super(UpsampleHead, self).__init__(init_cfg=init_cfg)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Upsample(
                scale_factor=scale_factor,
                mode=mode,
                align_corners=align_corners))

    def forward(self, x: Tensor) -> Tensor:
        return self.head(x)
