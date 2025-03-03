from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import (ForwardResults,
                                                  OptSampleList, SampleList)
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from mmengine.model import BaseModel
from torch import Tensor


class ContrastiveLoss(nn.Module):

    def __init__(self, temperature: float) -> None:
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, k: Tensor, q: Tensor) -> Tensor:
        logits = torch.mm(k, q.transpose(1, 0))
        target = torch.arange(k.shape[0], device=k.device).long()
        out = torch.div(logits, self.temperature)
        out = out.contiguous()
        loss = self.criterion(out, target)
        return loss


@MODELS.register_module()
class SLidR(BaseModel):

    def __init__(self,
                 backbone_3d: ConfigType,
                 head_3d: ConfigType,
                 backbone_2d: ConfigType,
                 head_2d: ConfigType,
                 superpixel_size: int,
                 temperature: float,
                 voxel_encoder_3d: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 train_cfg: ConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super(SLidR, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.backbone_2d = MODELS.build(backbone_2d)
        self.head_2d = MODELS.build(head_2d)

        self.backbone_3d = MODELS.build(backbone_3d)
        self.head_3d = MODELS.build(head_3d)
        if voxel_encoder_3d is not None:
            self.voxel_encoder_3d = MODELS.build(voxel_encoder_3d)
            self.range = True
        else:
            self.voxel_encoder_3d = None
            self.range = False

        self.superpixel_size = superpixel_size
        self.contrastive_loss = ContrastiveLoss(temperature)
        self.train_cfg = train_cfg

    def extract_3d_feature(self, feat_dict: dict) -> Tensor:
        if self.range:
            feat_dict = self.voxel_encoder_3d(feat_dict)
        feat_dict = self.backbone_3d(feat_dict)
        features = self.head_3d(feat_dict)['logits']
        features = F.normalize(features, p=2, dim=1)
        return features

    def extract_2d_feature(self, images: Tensor) -> Tensor:
        features = self.backbone_2d(images)
        features = self.head_2d(features)
        features = F.normalize(features, p=2, dim=1)

    def loss(self, inputs: dict,
             data_samples: SampleList) -> Dict[str, Tensor]:

        # forward
        features_2d = self.extract_2d_feature(inputs['imgs'])

        feat_dict = inputs['ranges'].copy(
        ) if self.range else inputs['voxels'].copy()
        features_3d = self.extract_3d_feature(feat_dict)

        superpixels = []
        pairing_images = []
        pairing_points = []
        offset = 0

        if self.range:
            coors = feat_dict['coors']
            for i, data_sample in enumerate(data_samples):
                superpixel = data_sample.gt_pts_seg.superpixels
                pairing_image = data_sample.gt_pts_seg.pairing_images
                pairing_image[:, 0] += i * superpixel.shape[0]
                pairing_point = data_sample.gt_pts_seg.pairing_points
                pairing_point = pairing_point.long() + offset
                offset += sum(coors[:, 0] == i)

                superpixels.append(superpixel)
                pairing_images.append(pairing_image)
                pairing_points.append(pairing_point)

        else:
            for i, data_sample in enumerate(data_samples):
                superpixel = data_sample.gt_pts_seg.superpixels
                pairing_image = data_sample.gt_pts_seg.pairing_images
                pairing_image[:, 0] += i * superpixel.shape[0]
                pairing_point = data_sample.gt_pts_seg.pairing_points
                inverse_map = feat_dict['point2voxel_maps'][i]
                pairing_point = inverse_map[pairing_point].long() + offset
                offset += feat_dict['voxel_inds'][i].shape[0]

                superpixels.append(superpixel)
                pairing_images.append(pairing_image)
                pairing_points.append(pairing_point)

        superpixels = torch.cat(superpixels)
        pairing_images = torch.cat(pairing_images)
        pairing_points = torch.cat(pairing_points)

        superpixels = (
            torch.arange(
                0,
                features_2d.shape[0] * self.superpixel_size,
                self.superpixel_size,
                device=features_2d.device)[:, None, None] + superpixels)

        m = tuple(pairing_images.cpu().T.long())
        superpixels_I = superpixels.flatten()
        idx_P = torch.arange(
            pairing_points.shape[0], device=features_2d.device)
        total_pixels = superpixels_I.shape[0]
        idx_I = torch.arange(total_pixels, device=features_2d.device)

        with torch.no_grad():
            one_hot_P = torch.sparse_coo_tensor(
                torch.stack((superpixels[m], idx_P), dim=0),
                torch.ones(pairing_points.shape[0], device=features_2d.device),
                (superpixels.shape[0] * self.superpixel_size,
                 pairing_points.shape[0]))
            one_hot_I = torch.sparse_coo_tensor(
                torch.stack((superpixels_I, idx_I), dim=0),
                torch.ones(total_pixels, device=features_2d.device),
                (superpixels.shape[0] * self.superpixel_size, total_pixels))

        k = one_hot_P @ features_3d[pairing_points]
        k = k / (torch.sparse.sum(one_hot_P, 1).to_dense()[:, None] + 1e-6)
        q = one_hot_I @ features_2d.permute(0, 2, 3, 1).flatten(0, 2)
        q = q / (torch.sparse.sum(one_hot_I, 1).to_dense()[:, None] + 1e-6)

        mask = torch.where(k[:, 0] != 0)
        valid_k = k[mask]
        valid_q = q[mask]

        loss = dict()
        loss['loss_spatial'] = self.contrastive_loss(valid_k, valid_q)

        return loss

    def forward(self,
                inputs: dict,
                data_samples: OptSampleList = None,
                mode: str = 'tensor') -> ForwardResults:
        if mode == 'loss':
            return self.loss(inputs, data_samples)
