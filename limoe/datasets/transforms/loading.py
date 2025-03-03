import copy
import os.path as osp

import numpy as np
import torch
from mmcv.transforms import BaseTransform
from mmdet3d.registry import TRANSFORMS
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from PIL import Image


@TRANSFORMS.register_module()
class LoadMultiModalityData(BaseTransform):

    def __init__(self,
                 superpixel_root: str,
                 num_cameras: int = 6,
                 min_dist: float = 1.0) -> None:
        self.superpixel_root = superpixel_root
        self.min_dist = min_dist
        self.num_cameras = num_cameras

    def transform(self, results: dict) -> dict:
        points = results['points'].numpy()
        pc_original = LidarPointCloud(points.T)
        pairing_points = np.empty(0, dtype=np.int64)
        pairing_images = np.empty((0, 3), dtype=np.int64)

        images = []
        superpixels = []

        camera_list = [
            'CAM_FRONT',
            'CAM_FRONT_LEFT',
            'CAM_FRONT_RIGHT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT',
        ]
        camera_list = np.random.choice(
            camera_list, size=self.num_cameras, replace=False)
        np.random.shuffle(camera_list)

        for i, cam in enumerate(camera_list):
            # load point clouds
            pc = copy.deepcopy(pc_original)

            # load camera images
            img = np.array(Image.open(results['images'][cam]['img_path']))

            # load superpixels
            sp_path = osp.join(
                self.superpixel_root,
                results['images'][cam]['sample_data_token'] + '.png')
            sp = np.array(Image.open(sp_path))

            # transform the point cloud to the vehicle frame for the
            # timestamp of the sweep.
            pc.rotate(results['lidar2ego_rotation'])
            pc.translate(results['lidar2ego_translation'])

            # transform from ego to the global frame.
            pc.rotate(results['ego2global_rotation'])
            pc.translate(results['ego2global_translation'])

            # transform from global frame to the ego vehicle frame for the
            # timestamp of the image.
            pc.translate(-results['images'][cam]['ego2global_translation'])
            pc.rotate(results['images'][cam]['ego2global_rotation'.T])

            # transform from ego to the camera.
            pc.translate(-results['images'][cam]['sensor2ego_translation'])
            pc.rotate(results['images'][cam]['sensor2ego_rotation'].T)

            # camera frame z axis points away from the camera
            depths = pc.points[2, :]

            # matrix multiplication with camera-matrix + renormalization.
            points = view_points(
                pc.points[:3, :],
                results['images'][cam]['cam_intrinsic'],
                normalize=True)

            # Remove points that are either outside or behind the camera.
            # Also make sure points are at least 1m in front of the camera to
            # avoid seeing the lidar points on the camera.
            points = points[:2].T
            mask = np.ones(depths.shape[0], dtype=bool)
            mask = np.logical_and(mask, depths > self.min_dist)
            mask = np.logical_and(mask, points[:, 0] > 0)
            mask = np.logical_and(mask, points[:, 0] < img.shape[1] - 1)
            mask = np.logical_and(mask, points[:, 1] > 0)
            mask = np.logical_and(mask, points[:, 1] < img.shape[0] - 1)

            matching_points = np.where(mask)[0]
            matching_pixels = np.round(
                np.flip(points[matching_points], axis=1)).astype(np.int64)

            images.append(img / 255.)
            superpixels.append(sp)
            pairing_points = np.concatenate((pairing_points, matching_points))
            pairing_images = np.concatenate(
                (pairing_images,
                 np.concatenate((np.ones(
                     (matching_pixels.shape[0], 1), dtype=np.int64) * i,
                                 matching_pixels),
                                axis=1)))

        results['imgs'] = torch.tensor(
            np.array(images, dtype=np.float32).transpose(0, 3, 1, 2))
        results['superpixels'] = torch.tensor(np.stack(superpixels))
        results['pairing_points'] = pairing_points
        results['pairing_images'] = pairing_images
        return results
