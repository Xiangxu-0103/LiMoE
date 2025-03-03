import os.path as osp
from typing import Callable, List, Optional, Union

import numpy as np
from mmdet3d.datasets import Seg3DDataset
from mmdet3d.registry import DATASETS


@DATASETS.register_module()
class NuScenesSegDataset(Seg3DDataset):

    METAINFO = {
        'classes': ('barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
                    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer',
                    'truck', 'driveable_surface', 'other_flat', 'sidewalk',
                    'terrain', 'manmade', 'vegetation'),
        'palette': [[255, 120, 50], [255, 192, 203], [255, 255, 0],
                    [0, 150, 245], [0, 255, 255], [255, 127, 0], [255, 0, 0],
                    [255, 240, 150], [135, 60, 0], [160, 32,
                                                    240], [255, 0, 255],
                    [139, 137, 137], [75, 0, 75], [150, 240, 80],
                    [230, 230, 250], [0, 175, 0]],
        'seg_valid_class_ids':
        tuple(range(16)),
        'seg_all_class_ids':
        tuple(range(16)),
    }

    def __init__(self,
                 data_root: Optional[str] = None,
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 data_prefix: dict = dict(
                     pts='',
                     img='',
                     pts_instance_mask='',
                     pts_semantic_mask=''),
                 pipeline: List[Union[dict, Callable]] = [],
                 modality: dict = dict(use_lidar=True, use_camera=False),
                 ignore_index: Optional[int] = None,
                 scene_idxs: Optional[Union[str, np.ndarray]] = None,
                 test_mode: bool = False,
                 serialize_data: bool = True,
                 **kwargs) -> None:
        super(NuScenesSegDataset, self).__init__(
            data_root=data_root,
            ann_file=ann_file,
            metainfo=metainfo,
            data_prefix=data_prefix,
            pipeline=pipeline,
            modality=modality,
            ignore_index=ignore_index,
            scene_idxs=scene_idxs,
            test_mode=test_mode,
            serialize_data=serialize_data,
            **kwargs)

    def get_seg_label_mapping(self, metainfo: dict) -> np.ndarray:
        seg_label_mapping = np.zeros(metainfo['max_label'] + 1, dtype=np.int64)
        for idx in metainfo['seg_label_mapping']:
            seg_label_mapping[idx] = metainfo['seg_label_mapping'][idx]
        return seg_label_mapping

    def parse_data_info(self, info: dict) -> dict:
        if self.modality['use_lidar']:
            info['lidar_points']['lidar_path'] = \
                osp.join(
                    self.data_prefix.get('pts', ''),
                    info['lidar_points']['lidar_path'])
            if 'num_pts_feats' in info['lidar_points']:
                info['num_pts_feats'] = info['lidar_points']['num_pts_feats']
            info['lidar_path'] = info['lidar_points']['lidar_path']

        if self.modality['use_camera']:
            for cam_id, img_info in info['images'].items():
                if 'img_path' in img_info:
                    if cam_id in self.data_prefix:
                        cam_prefix = self.data_prefix[cam_id]
                    else:
                        cam_prefix = self.data_prefix.get('img', '')
                    img_info['img_path'] = osp.join(cam_prefix,
                                                    img_info['img_path'])

        if 'pts_instance_mask_path' in info:
            info['pts_instance_mask_path'] = \
                osp.join(self.data_prefix.get('pts_instance_mask', ''),
                         info['pts_instance_mask_path'])

        if 'pts_semantic_mask_path' in info:
            info['pts_semantic_mask_path'] = \
                osp.join(self.data_prefix.get('pts_semantic_mask', ''),
                         info['pts_semantic_mask_path'])

        info['seg_label_mapping'] = self.seg_label_mapping

        if self.test_mode and self.load_eval_anns:
            info['eval_ann_info'] = dict()

        return info
