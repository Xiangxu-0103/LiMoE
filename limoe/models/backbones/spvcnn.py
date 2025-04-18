from typing import Optional, Sequence

import torch
import torch.nn as nn
import torchsparse
import torchsparse.nn.functional as F
from mmdet3d.registry import MODELS
from mmdet3d.utils import OptMultiConfig
from torchsparse import PointTensor, SparseTensor
from torchsparse.nn.utils import get_kernel_offsets

from .minkunet import MinkUNetBackbone


@MODELS.register_module()
class SPVCNNBackbone(MinkUNetBackbone):

    def __init__(self,
                 in_channels: int = 4,
                 base_channels: int = 32,
                 layers: Sequence[int] = [2, 3, 4, 6, 2, 2, 2, 2],
                 planes: Sequence[int] = [32, 64, 128, 256, 256, 128, 96, 96],
                 block_type: str = 'basic',
                 bn_momentum: float = 0.1,
                 drop_ratio: float = 0.3,
                 init_cfg: OptMultiConfig = None) -> None:
        super(SPVCNNBackbone, self).__init__(
            in_channels=in_channels,
            base_channels=base_channels,
            layers=layers,
            planes=planes,
            block_type=block_type,
            bn_momentum=bn_momentum,
            init_cfg=init_cfg)

        self.point_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(base_channels, planes[3]), nn.BatchNorm1d(planes[3]),
                nn.ReLU(True)),
            nn.Sequential(
                nn.Linear(planes[3], planes[5]), nn.BatchNorm1d(planes[5]),
                nn.ReLU(True)),
            nn.Sequential(
                nn.Linear(planes[5], planes[7]), nn.BatchNorm1d(planes[7]),
                nn.ReLU(True)),
        ])
        self.dropout = nn.Dropout(drop_ratio, True)

    def forward(self, feat_dict: dict) -> dict:
        voxel_features = feat_dict['voxels']
        coors = feat_dict['coors']

        # x: SparseTensor z: PointTensor
        x = SparseTensor(voxel_features, coors)
        z = PointTensor(x.F, x.C.float())
        x = initial_voxelize(z)

        out0 = self.conv0(x)
        z0 = voxel_to_point(out0, z)
        out0 = point_to_voxel(out0, z0)

        out1 = self.conv1(out0)
        out1 = self.block1(out1)

        out2 = self.conv2(out1)
        out2 = self.block2(out2)

        out3 = self.conv3(out2)
        out3 = self.block3(out3)

        out4 = self.conv4(out3)
        out4 = self.block4(out4)

        z1 = voxel_to_point(out4, z0, self.point_transforms[0])
        out4 = point_to_voxel(out4, z1)
        out4.F = self.dropout(out4.F)

        out = self.conv5(out4)
        out = torchsparse.cat((out, out3))
        out = self.block5(out)

        out = self.conv6(out)
        out = torchsparse.cat((out, out2))
        out = self.block6(out)

        z2 = voxel_to_point(out, z1, self.point_transforms[1])
        out = point_to_voxel(out, z2)
        out.F = self.dropout(out.F)

        out = self.conv7(out)
        out = torchsparse.cat((out, out1))
        out = self.block7(out)

        out = self.conv8(out)
        out = torchsparse.cat((out, out0))
        out = self.block8(out)

        out = voxel_to_point(out, z2, self.point_transforms[2])
        feat_dict['voxel_feats'] = out.F
        return feat_dict


def initial_voxelize(points: PointTensor) -> SparseTensor:
    """Voxelize again based on input PointTensor.

    Args:
        points (PointTensor): Input points after voxelization.

    Returns:
        SparseTensor: New voxels.
    """
    pc_hash = F.sphash(torch.floor(points.C).int())
    sparse_hash = torch.unique(pc_hash)
    idx_query = F.sphashquery(pc_hash, sparse_hash)
    counts = F.spcount(idx_query.int(), len(sparse_hash))

    inserted_coords = F.spvoxelize(torch.floor(points.C), idx_query, counts)
    inserted_coords = torch.round(inserted_coords).int()
    inserted_feat = F.spvoxelize(points.F, idx_query, counts)

    new_tensor = SparseTensor(inserted_feat, inserted_coords, 1)
    new_tensor.cmaps.setdefault(new_tensor.stride, new_tensor.coords)
    points.additional_features['idx_query'][1] = idx_query
    points.additional_features['counts'][1] = counts
    return new_tensor


def voxel_to_point(voxels: SparseTensor,
                   points: PointTensor,
                   point_transform: Optional[nn.Module] = None,
                   nearest: bool = False) -> PointTensor:
    """Fead voxel features to points.

    Args:
        voxels (SparseTensor): Input voxels.
        points (PointTensor): Input points.
        nearest (bool): Whether to use nearest neighbor interpolation.
            Defaults to False.

    Returns:
        PointTensor: Points with new features.
    """
    if points.idx_query is None or points.weights is None or \
            points.idx_query.get(voxels.s) is None or \
            points.weights.get(voxels.s) is None:
        offsets = get_kernel_offsets(2, voxels.s, 1, device=points.F.device)
        old_hash = F.sphash(
            torch.cat([
                torch.floor(points.C[:, :3] / voxels.s[0]).int() * voxels.s[0],
                points.C[:, -1].int().view(-1, 1)
            ], 1), offsets)
        pc_hash = F.sphash(voxels.C.to(points.F.device))
        idx_query = F.sphashquery(old_hash, pc_hash)
        weights = F.calc_ti_weights(
            points.C, idx_query, scale=voxels.s[0]).transpose(0,
                                                              1).contiguous()
        idx_query = idx_query.transpose(0, 1).contiguous()
        if nearest:
            weights[:, 1:] = 0.
            idx_query[:, 1:] = -1
        new_features = F.spdevoxelize(voxels.F, idx_query, weights)
        new_tensor = PointTensor(
            new_features,
            points.C,
            idx_query=points.idx_query,
            weights=points.weights)
        new_tensor.additional_features = points.additional_features
        new_tensor.idx_query[voxels.s] = idx_query
        new_tensor.weights[voxels.s] = weights
        points.idx_query[voxels.s] = idx_query
        points.weights[voxels.s] = weights
    else:
        new_features = F.spdevoxelize(voxels.F, points.idx_query.get(voxels.s),
                                      points.weights.get(voxels.s))
        new_tensor = PointTensor(
            new_features,
            points.C,
            idx_query=points.idx_query,
            weights=points.weights)
        new_tensor.additional_features = points.additional_features

    if point_transform is not None:
        new_tensor.F = new_tensor.F + point_transform(points.F)

    return new_tensor


def point_to_voxel(voxels: SparseTensor, points: PointTensor) -> SparseTensor:
    """Feed point features to voxels.

    Args:
        voxels (SparseTensor): Input voxels.
        points (PointTensor): Input points.

    Returns:
        SparseTensor: Voxels with new features.
    """
    if points.additional_features is None or \
            points.additional_features.get('idx_query') is None or \
            points.additional_features['idx_query'].get(voxels.s) is None:
        pc_hash = F.sphash(
            torch.cat([
                torch.floor(points.C[:, :3] / voxels.s[0]).int() * voxels.s[0],
                points.C[:, -1].int().view(-1, 1)
            ], 1))
        sparse_hash = F.sphash(voxels.C)
        idx_query = F.sphashquery(pc_hash, sparse_hash)
        counts = F.spcount(idx_query.int(), voxels.C.shape[0])
        points.additional_features['idx_query'][voxels.s] = idx_query
        points.additional_features['counts'][voxels.s] = counts
    else:
        idx_query = points.additional_features['idx_query'][voxels.s]
        counts = points.additional_features['counts'][voxels.s]

    inserted_features = F.spvoxelize(points.F, idx_query, counts)
    new_tensor = SparseTensor(inserted_features, voxels.C, voxels.s)
    new_tensor.cmaps = voxels.cmaps
    new_tensor.kmaps = voxels.kmaps

    return new_tensor
