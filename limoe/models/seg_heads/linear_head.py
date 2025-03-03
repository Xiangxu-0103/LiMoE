from typing import Dict, List

import torch
import torch.nn as nn
from mmdet3d.models import Base3DDecodeHead
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils import ConfigType, OptConfigType
from torch import Tensor


@MODELS.register_module()
class LinearHead(Base3DDecodeHead):

    def __init__(self, loss_lovasz: OptConfigType = None, **kwargs) -> None:
        super(LinearHead, self).__init__(**kwargs)

        if loss_lovasz is not None:
            self.loss_lovasz = MODELS.build(loss_lovasz)
        else:
            self.loss_lovasz = None

    def build_conv_seg(self, channels: int, num_classes: int,
                       kernel_size: int) -> nn.Module:
        return nn.Linear(channels, num_classes)

    def forward(self, feat_dict: dict) -> dict:
        logits = self.cls_seg(feat_dict['voxel_feats'])
        feat_dict['logits'] = logits
        return feat_dict

    def loss_by_feat(self, feat_dict: dict,
                     batch_data_samples: SampleList) -> Dict[str, Tensor]:
        voxel_semantic_segs = []
        voxel_inds = feat_dict['voxel_inds']
        for batch_idx, data_sample in enumerate(batch_data_samples):
            pts_semantic_mask = data_sample.gt_pts_seg.pts_semantic_mask
            voxel_semantic_mask = pts_semantic_mask[voxel_inds[batch_idx]]
            voxel_semantic_segs.append(voxel_semantic_mask)
        seg_label = torch.cat(voxel_semantic_segs)
        seg_logit_feat = feat_dict['logits']
        loss = dict()
        loss['loss_ce'] = self.loss_decode(
            seg_logit_feat, seg_label, ignore_index=self.ignore_index)
        if self.loss_lovasz is not None:
            loss['loss_lovasz'] = self.loss_lovasz(
                seg_logit_feat, seg_label, ignore_index=self.ignore_index)
        return loss

    def predict(self, feat_dict: dict, batch_input_metas: List[dict],
                test_cfg: ConfigType) -> List[Tensor]:
        feat_dict = self.forward(feat_dict)
        seg_pred_list = self.predict_by_feat(feat_dict, batch_input_metas)
        return seg_pred_list

    def predict_by_feat(self, feat_dict: dict,
                        batch_input_metas: List[dict]) -> List[Tensor]:
        seg_logits = feat_dict['logits']

        seg_pred_list = []
        coors = feat_dict['coors']
        for batch_idx in range(len(batch_input_metas)):
            batch_mask = coors[:, -1] == batch_idx
            seg_logits_sample = seg_logits[batch_mask]
            point2voxel_map = feat_dict['point2voxel_maps'][batch_idx].long()
            point_seg_predicts = seg_logits_sample[point2voxel_map]
            seg_pred_list.append(point_seg_predicts)

        return seg_pred_list
