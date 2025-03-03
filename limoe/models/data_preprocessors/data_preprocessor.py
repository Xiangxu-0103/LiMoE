from typing import List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from mmengine.model import ImgDataPreprocessor
from mmengine.utils import is_seq_of
from torch import Tensor


@MODELS.register_module()
class LiMoEDataPreprocessor(ImgDataPreprocessor):

    def __init__(self,
                 H: int,
                 W: int,
                 fov_up: float,
                 fov_down: float,
                 ignore_index: int,
                 voxel_size: Sequence[float],
                 voxel_type: str = 'cubic',
                 mean: Optional[Sequence[Union[float, int]]] = None,
                 std: Optional[Sequence[Union[float, int]]] = None,
                 pad_size_divisor: int = 1,
                 pad_value: Union[float, int] = 0,
                 to_rgb: bool = False,
                 bgr_to_rgb: bool = False,
                 rgb_to_bgr: bool = False,
                 non_blocking: bool = False) -> None:
        super(LiMoEDataPreprocessor, self).__init__(
            mean=mean,
            std=std,
            pad_size_divisor=pad_size_divisor,
            pad_value=pad_value,
            bgr_to_rgb=bgr_to_rgb,
            rgb_to_bgr=rgb_to_bgr,
            non_blocking=non_blocking)
        self._channel_conversion = to_rgb or bgr_to_rgb or rgb_to_bgr
        self.H = H
        self.W = W
        self.fov_up = fov_up / 180 * np.pi
        self.fov_down = fov_down / 180 * np.pi
        self.fov = abs(self.fov_down) + abs(self.fov_up)
        self.ignore_index = ignore_index
        self.voxel_size = voxel_size
        self.voxel_type = voxel_type

    def forward(self, data: dict, training: bool = False) -> dict:
        data = self.collate_data(data)
        inputs, data_samples = data['inputs'], data['data_samples']
        batch_inputs = dict()

        if 'points' in inputs:
            batch_inputs['points'] = inputs['points']
            voxel_dict = self.voxelize(inputs['points'], data_samples)
            range_dict = self.frustum_region_group(inputs['points'],
                                                   data_samples)
            batch_inputs['voxels'] = voxel_dict
            batch_inputs['ranges'] = range_dict

        if 'imgs' in inputs:
            imgs = inputs['imgs']

            if data_samples is not None:
                batch_input_shape = tuple(imgs[0].size()[-2:])
                for data_sample in data_samples:
                    data_sample.set_metainfo(
                        {'batch_input_shape': batch_input_shape})

            batch_inputs['imgs'] = imgs

        return {'inputs': batch_inputs, 'data_samples': data_samples}

    def preprocess_img(self, _batch_img: Tensor) -> Tensor:
        if self._channel_conversion:
            _batch_img = _batch_img[[2, 1, 0], ...]
        _batch_img = _batch_img.float()
        if self._enable_normalize:
            if self.mean.shape[0] == 3:
                assert _batch_img.dim() == 3 and _batch_img.shape[0] == 3
            _batch_img = (_batch_img - self.mean) / self.std
        return _batch_img

    def collate_data(self, data: dict) -> dict:
        data = self.cast_data(data)

        if 'imgs' in data['inputs']:
            _batch_imgs = data['inputs']['imgs']
            assert is_seq_of(_batch_imgs, Tensor)

            batch_imgs = []
            for _batch_img in _batch_imgs:
                _batch_img = [self.preprocess_img(_img) for _img in _batch_img]
                _batch_img = torch.stack(_batch_img, dim=0)
                batch_imgs.append(_batch_img)

            batch_imgs = torch.concat(batch_imgs, dim=0)
            data['inputs']['imgs'] = batch_imgs

        data.setdefault('data_samples', None)
        return data

    @torch.no_grad()
    def voxelize(self, points: List[Tensor], data_samples: SampleList) -> dict:
        voxel_dict = dict()

        voxels = []
        coors = []
        point2voxel_maps = []
        voxel_inds = []

        voxel_size = points[0].new_tensor(self.voxel_size)

        for i, res in enumerate(points):
            if self.voxel_type == 'cubic':
                res_coors = torch.round(res[:, :3] / voxel_size).int()
            elif self.voxel_type == 'cylinder':
                rho = torch.sqrt(res[:, 0]**2 + res[:, 1]**2)
                phi = torch.atan2(res[:, 1], res[:, 0]) * 180 / np.pi
                polar_res = torch.stack((rho, phi, res[:, 2]), dim=1)
                res_coors = torch.round(polar_res[:, :3] / voxel_size).int()

            res_coors -= res_coors.min(0)[0]

            res_coors_numpy = res_coors.cpu().numpy()
            inds, point2voxel_map = self.sparse_quantize(
                res_coors_numpy, return_index=True, return_inverse=True)
            point2voxel_map = torch.from_numpy(point2voxel_map).cuda()
            inds = torch.from_numpy(inds).cuda()
            res_voxel_coors = res_coors[inds]
            res_voxels = res[inds]
            res_voxel_coors = F.pad(
                res_voxel_coors, (0, 1), mode='constant', value=i)
            voxels.append(res_voxels)
            coors.append(res_voxel_coors)
            point2voxel_maps.append(point2voxel_map)
            voxel_inds.append(inds)

        voxels = torch.cat(voxels, dim=0)
        coors = torch.cat(coors, dim=0)

        voxel_dict['voxels'] = voxels
        voxel_dict['coors'] = coors
        voxel_dict['point2voxel_maps'] = point2voxel_maps
        voxel_dict['voxel_inds'] = voxel_inds

        return voxel_dict

    def ravel_hash(self, x: np.ndarray) -> np.ndarray:
        assert x.ndim == 2, x.shape

        x = x - np.min(x, axis=0)
        x = x.astype(np.uint64, copy=False)
        xmax = np.max(x, axis=0).astype(np.uint64) + 1

        h = np.zeros(x.shape[0], dtype=np.uint64)
        for k in range(x.shape[1] - 1):
            h += x[:, k]
            h *= xmax[k + 1]
        h += x[:, -1]
        return h

    def sparse_quantize(self,
                        coords: np.ndarray,
                        return_index: bool = False,
                        return_inverse: bool = False) -> List[np.ndarray]:
        _, indices, inverse_indices = np.unique(
            self.ravel_hash(coords), return_index=True, return_inverse=True)

        outputs = []
        if return_index:
            outputs += [indices]
        if return_inverse:
            outputs += [inverse_indices]
        return outputs

    @torch.no_grad()
    def frustum_region_group(self, points: List[Tensor],
                             data_samples: SampleList) -> dict:
        range_dict = dict()

        coors = []
        voxels = []

        for i, res in enumerate(points):
            depth = torch.linalg.norm(res[:, :3], 2, dim=1)
            yaw = -torch.atan2(res[:, 1], res[:, 0])
            pitch = torch.arcsin(res[:, 2] / depth)

            coors_x = 0.5 * (yaw / np.pi + 1.0)
            coors_y = 1.0 - (pitch + abs(self.fov_down)) / self.fov

            # scale to image size using angular resolution
            coors_x *= self.W
            coors_y *= self.H

            # round and clamp for use as index
            coors_x = torch.floor(coors_x)
            coors_x = torch.clamp(
                coors_x, min=0, max=self.W - 1).type(torch.int64)

            coors_y = torch.floor(coors_y)
            coors_y = torch.clamp(
                coors_y, min=0, max=self.H - 1).type(torch.int64)

            res_coors = torch.stack([coors_y, coors_x], dim=1)
            res_coors = F.pad(res_coors, (1, 0), mode='constant', value=i)
            coors.append(res_coors)
            voxels.append(res)

            if 'pts_semantic_mask' in data_samples[i].gt_pts_seg:
                import torch_scatter
                pts_semantic_mask = data_samples[
                    i].gt_pts_seg.pts_semantic_mask
                seg_label = torch.ones(
                    (self.H, self.W),
                    dtype=torch.long,
                    device=pts_semantic_mask.device) * self.ignore_index
                res_voxel_coors, inverse_map = torch.unique(
                    res_coors, return_inverse=True, dim=0)
                voxel_semantic_mask = torch_scatter.scatter_mean(
                    F.one_hot(pts_semantic_mask).float(), inverse_map, dim=0)
                voxel_semantic_mask = torch.argmax(voxel_semantic_mask, dim=-1)
                seg_label[res_voxel_coors[:, 1],
                          res_voxel_coors[:, 2]] = voxel_semantic_mask
                data_samples[i].gt_pts_seg.semantic_seg = seg_label

        voxels = torch.cat(voxels, dim=0)
        coors = torch.cat(coors, dim=0)
        range_dict['voxels'] = voxels
        range_dict['coors'] = coors

        return range_dict
