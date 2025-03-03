from typing import Sequence

import torch
from mmcv.transforms import BaseTransform
from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures import Det3DDataSample, PointData
from mmdet3d.structures.points import BasePoints


@TRANSFORMS.register_module()
class LiMoEInputs(BaseTransform):

    def __init__(self, keys: Sequence[str] = None):
        self.keys = keys

    def transform(self, results: dict) -> dict:
        if 'points' in results:
            if isinstance(results['points'], BasePoints):
                results['points'] = results['points'].tensor

        if 'pairing_points' in results:
            results['pairing_points'] = torch.tensor(results['pairing_points'])

        if 'pairing_images' in results:
            results['pairing_images'] = torch.tensor(results['pairing_images'])

        data_sample = Det3DDataSample()
        gt_pts_seg = PointData()

        inputs = {}
        for key in self.keys:
            if key in ('points', 'imgs'):
                inputs[key] = results[key]
            elif key in ('pairing_points', 'pairing_images', 'superpixels',
                         'pts_semantic_mask'):
                gt_pts_seg[key] = results[key]

        data_sample.gt_pts_seg = gt_pts_seg

        packed_results = dict()
        packed_results['data_samples'] = data_sample
        packed_results['inputs'] = inputs

        return packed_results
