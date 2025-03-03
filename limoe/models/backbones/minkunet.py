from typing import Optional, Sequence

import torch.nn as nn
import torchsparse
import torchsparse.nn as spnn
from mmdet3d.registry import MODELS
from mmdet3d.utils import OptMultiConfig
from mmengine.model import BaseModule
from torchsparse.tensor import SparseTensor


class TorchsparseConvModule(BaseModule):

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 kernel_size: int,
                 stride: int = 1,
                 dilation: int = 1,
                 bn_momentum: float = 0.1,
                 transposed: bool = False,
                 activate: bool = True,
                 init_cfg: OptMultiConfig = None) -> None:
        super(TorchsparseConvModule, self).__init__(init_cfg=init_cfg)

        self.conv = spnn.Conv3d(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=False,
            transposed=transposed)
        self.norm = spnn.BatchNorm(planes, momentum=bn_momentum)
        if activate:
            self.relu = spnn.ReLU(inplace=True)
        else:
            self.relu = None

    def forward(self, x: SparseTensor) -> SparseTensor:
        out = self.conv(x)
        out = self.norm(out)
        if self.relu is not None:
            out = self.relu(out)
        return out


class TorchsparseBasicBlock(BaseModule):
    expansion = 1

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 dilation: int = 1,
                 downsample: Optional[nn.Module] = None,
                 bn_momentum: float = 0.1,
                 init_cfg: OptMultiConfig = None) -> None:
        super(TorchsparseBasicBlock, self).__init__(init_cfg=init_cfg)

        self.conv1 = spnn.Conv3d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            bias=False)
        self.norm1 = spnn.BatchNorm(planes, momentum=bn_momentum)

        self.conv2 = spnn.Conv3d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            dilation=dilation,
            bias=False)
        self.norm2 = spnn.BatchNorm(planes, momentum=bn_momentum)
        self.relu = spnn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: SparseTensor) -> SparseTensor:
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class TorchsparseBottleneck(BaseModule):
    expansion = 4

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 dilation: int = 1,
                 downsample: Optional[nn.Module] = None,
                 bn_momentum: float = 0.1,
                 init_cfg: OptMultiConfig = None) -> None:
        super(TorchsparseBottleneck, self).__init__(init_cfg=init_cfg)

        self.conv1 = spnn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.norm1 = spnn.BatchNorm(planes, momentum=bn_momentum)

        self.conv2 = spnn.Conv3d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            bias=False)
        self.norm2 = spnn.BatchNorm(planes, momentum=bn_momentum)

        self.conv3 = spnn.Conv3d(
            planes, planes * self.expansion, kernel_size=1, bias=False)
        self.norm3 = spnn.BatchNorm(
            planes * self.expansion, momentum=bn_momentum)

        self.relu = spnn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: SparseTensor) -> SparseTensor:
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


@MODELS.register_module()
class MinkUNetBackbone(BaseModule):

    def __init__(self,
                 in_channels: int = 4,
                 base_channels: int = 32,
                 layers: Sequence[int] = [2, 3, 4, 6, 2, 2, 2, 2],
                 planes: Sequence[int] = [32, 64, 128, 256, 256, 128, 96, 96],
                 block_type: str = 'basic',
                 bn_momentum: float = 0.1,
                 init_cfg: OptMultiConfig = None) -> None:
        super(MinkUNetBackbone, self).__init__(init_cfg=init_cfg)
        assert block_type in ['basic', 'bottleneck']

        conv_module = TorchsparseConvModule
        if block_type == 'basic':
            block = TorchsparseBasicBlock
        elif block_type == 'bottleneck':
            block = TorchsparseBottleneck

        self.conv0 = nn.Sequential(
            conv_module(
                in_channels,
                base_channels,
                kernel_size=3,
                bn_momentum=bn_momentum),
            conv_module(
                base_channels,
                base_channels,
                kernel_size=3,
                bn_momentum=bn_momentum))

        self.inplanes = base_channels

        self.conv1 = conv_module(
            self.inplanes,
            self.inplanes,
            kernel_size=2,
            stride=2,
            dilation=1,
            bn_momentum=bn_momentum)
        self.block1 = self._make_layer(
            block, conv_module, planes[0], layers[0], bn_momentum=bn_momentum)

        self.conv2 = conv_module(
            self.inplanes,
            self.inplanes,
            kernel_size=2,
            stride=2,
            dilation=1,
            bn_momentum=bn_momentum)
        self.block2 = self._make_layer(
            block, conv_module, planes[1], layers[1], bn_momentum=bn_momentum)

        self.conv3 = conv_module(
            self.inplanes,
            self.inplanes,
            kernel_size=2,
            stride=2,
            dilation=1,
            bn_momentum=bn_momentum)
        self.block3 = self._make_layer(
            block, conv_module, planes[2], layers[2], bn_momentum=bn_momentum)

        self.conv4 = conv_module(
            self.inplanes,
            self.inplanes,
            kernel_size=2,
            stride=2,
            dilation=1,
            bn_momentum=bn_momentum)
        self.block4 = self._make_layer(
            block, conv_module, planes[3], layers[3], bn_momentum=bn_momentum)

        self.conv5 = conv_module(
            self.inplanes,
            planes[4],
            kernel_size=2,
            stride=2,
            dilation=1,
            bn_momentum=bn_momentum,
            transposed=True)
        self.inplanes = planes[4] + planes[2] * block.expansion
        self.block5 = self._make_layer(
            block, conv_module, planes[4], layers[4], bn_momentum=bn_momentum)

        self.conv6 = conv_module(
            self.inplanes,
            planes[5],
            kernel_size=2,
            stride=2,
            dilation=1,
            bn_momentum=bn_momentum,
            transposed=True)
        self.inplanes = planes[5] + planes[1] * block.expansion
        self.block6 = self._make_layer(
            block, conv_module, planes[5], layers[5], bn_momentum=bn_momentum)

        self.conv7 = conv_module(
            self.inplanes,
            planes[6],
            kernel_size=2,
            stride=2,
            dilation=1,
            bn_momentum=bn_momentum,
            transposed=True)
        self.inplanes = planes[6] + planes[0] * block.expansion
        self.block7 = self._make_layer(
            block, conv_module, planes[6], layers[6], bn_momentum=bn_momentum)

        self.conv8 = conv_module(
            self.inplanes,
            planes[7],
            kernel_size=2,
            stride=2,
            dilation=1,
            bn_momentum=bn_momentum,
            transposed=True)
        self.inplanes = planes[7] + base_channels
        self.block8 = self._make_layer(
            block, conv_module, planes[7], layers[7], bn_momentum=bn_momentum)

    def _make_layer(self,
                    block: nn.Module,
                    conv_module: nn.Module,
                    planes: int,
                    blocks: int,
                    stride: int = 1,
                    dilation: int = 1,
                    bn_momentum: float = 0.1) -> nn.Module:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = conv_module(
                self.inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=stride,
                bn_momentum=bn_momentum,
                activate=False)
        layers = []

        layers.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                dilation=dilation,
                bn_momentum=bn_momentum,
                downsample=downsample))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    stride=1,
                    dilation=dilation,
                    bn_momentum=bn_momentum))

        return nn.Sequential(*layers)

    def forward(self, feat_dict: dict) -> dict:
        voxel_features = feat_dict['voxels']
        coors = feat_dict['coors']
        x = torchsparse.SparseTensor(voxel_features, coors)

        out1 = self.conv0(x)

        out = self.conv1(out1)
        out2 = self.block1(out)

        out = self.conv2(out2)
        out3 = self.block2(out)

        out = self.conv3(out3)
        out4 = self.block3(out)

        out = self.conv4(out4)
        out5 = self.block4(out)

        out = self.conv5(out5)
        out = torchsparse.cat((out, out4))
        out = self.block5(out)

        out = self.conv6(out)
        out = torchsparse.cat((out, out3))
        out = self.block6(out)

        out = self.conv7(out)
        out = torchsparse.cat((out, out2))
        out = self.block7(out)

        out = self.conv8(out)
        out = torchsparse.cat((out, out1))
        out = self.block8(out)

        feat_dict['voxel_feats'] = out.F
        return feat_dict
