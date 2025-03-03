import random
from typing import Sequence

import numpy as np
import torch
from mmcv.transforms import BaseTransform
from mmdet3d.registry import TRANSFORMS
from torchvision.transforms import InterpolationMode, RandomResizedCrop
from torchvision.transforms.functional import hflip, resize, resized_crop


@TRANSFORMS.register_module()
class ResizedCrop(BaseTransform):

    def __init__(self,
                 image_crop_size: Sequence[int] = (224, 416),
                 image_crop_range: Sequence[float] = (0.3, 1.0),
                 image_crop_ratio: Sequence[float] = (14.0 / 9.0, 17.0 / 9.0),
                 crop_center: bool = False) -> None:
        self.crop_size = image_crop_size
        self.crop_range = image_crop_range
        self.crop_ratio = image_crop_ratio
        self.crop_center = crop_center

    def transform(self, results: dict) -> dict:
        images = results['imgs']
        superpixels = results['superpixels'].unsqueeze(1)
        pairing_points = results['pairing_points']
        pairing_images = results['pairing_images']

        imgs = torch.empty(
            (images.shape[0], 3) + tuple(self.crop_size), dtype=torch.float32)
        sps = torch.empty(
            (images.shape[0], ) + tuple(self.crop_size), dtype=torch.uint8)
        pairing_points_out = np.empty(0, dtype=np.int64)
        pairing_images_out = np.empty((0, 3), dtype=np.int64)

        if self.crop_center:
            pairing_points_out = pairing_points

            _, _, h, w = images.shape
            for id, img in enumerate(images):
                mask = pairing_images[:, 0] == id
                p2 = pairing_images[mask]
                p2 = np.round(
                    np.multiply(
                        p2,
                        [1.0, self.crop_size[0] / h, self.crop_size[1] / w
                         ])).astype(np.int64)
                imgs[id] = resize(img, self.crop_size,
                                  InterpolationMode.BILINEAR)
                sps[id] = resize(superpixels[id], self.crop_size,
                                 InterpolationMode.NEAREST)
                p2[:, 1] = np.clip(0, self.crop_size[0] - 1, p2[:, 1])
                p2[:, 2] = np.clip(0, self.crop_size[1] - 1, p2[:, 2])
                pairing_images_out = np.concatenate((pairing_images_out, p2))
        else:
            for id, img in enumerate(images):
                successful = False
                mask = pairing_images[:, 0] == id
                P1 = pairing_points[mask]
                P2 = pairing_images[mask]
                while not successful:
                    i, j, h, w = RandomResizedCrop.get_params(
                        img, self.crop_range, self.crop_ratio)
                    p1 = P1.copy()
                    p2 = P2.copy()
                    p2 = np.round(
                        np.multiply(p2 - [0, i, j], [
                            1.0, self.crop_size[0] / h, self.crop_size[1] / w
                        ])).astype(np.int64)
                    valid_indexes_0 = np.logical_and(
                        p2[:, 1] < self.crop_size[0], p2[:, 1] >= 0)
                    valid_indexes_1 = np.logical_and(
                        p2[:, 2] < self.crop_size[1], p2[:, 2] >= 0)
                    valid_indexes = np.logical_and(valid_indexes_0,
                                                   valid_indexes_1)
                    sum_indexes = valid_indexes.sum()
                    len_indexes = len(valid_indexes)
                    if sum_indexes > 1024 or sum_indexes / len_indexes > 0.75:
                        successful = True
                imgs[id] = resized_crop(img, i, j, h, w, self.crop_size,
                                        InterpolationMode.BILINEAR)
                sps[id] = resized_crop(superpixels[id], i, j, h, w,
                                       self.crop_size,
                                       InterpolationMode.NEAREST)
                pairing_points_out = np.concatenate(
                    (pairing_points_out, p1[valid_indexes]))
                pairing_images_out = np.concatenate(
                    (pairing_images_out, p2[valid_indexes]))

        results['imgs'] = imgs
        results['superpixels'] = sps
        results['pairing_points'] = pairing_points_out
        results['pairing_images'] = pairing_images_out
        return results


@TRANSFORMS.register_module()
class FlipHorizontal(BaseTransform):

    def __init__(self, flip_ratio: float = 0.5) -> None:
        self.flip_ratio = flip_ratio

    def transform(self, results: dict) -> dict:
        images = results['imgs']
        superpixels = results['superpixels']
        pairing_images = results['pairing_images']

        w = images.shape[3]
        for i, img in enumerate(images):
            if random.random() < self.flip_ratio:
                images[i] = hflip(img)
                superpixels[i] = hflip(superpixels[i:i + 1])
                mask = pairing_images[:, 0] == i
                pairing_images[mask, 2] = w - 1 - pairing_images[mask, 2]

        results['imgs'] = images
        results['superpixels'] = superpixels
        results['pairing_images'] = pairing_images
        return results
