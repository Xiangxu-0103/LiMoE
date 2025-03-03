dataset_type = 'NuScenesSegDataset'
data_root = 'data/nuscenes/'
class_names = [
    'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle',
    'pedestrian', 'traffic_cone', 'trailer', 'truck', 'driveable_surface',
    'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation'
]
labels_map = {
    0: 16,
    1: 16,
    2: 6,
    3: 6,
    4: 6,
    5: 16,
    6: 6,
    7: 16,
    8: 16,
    9: 0,
    10: 16,
    11: 16,
    12: 7,
    13: 16,
    14: 1,
    15: 2,
    16: 2,
    17: 3,
    18: 4,
    19: 16,
    20: 16,
    21: 5,
    22: 8,
    23: 9,
    24: 10,
    25: 11,
    26: 12,
    27: 13,
    28: 14,
    29: 16,
    30: 15,
    31: 16
}

metainfo = dict(
    classes=class_names, seg_label_mapping=labels_map, max_label=31)
input_modality = dict(use_lidar=True, use_camera=True)

data_prefix = dict(
    pts='samples/LIDAR_TOP',
    CAM_FRONT='samples/CAM_FRONT',
    CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
    CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
    CAM_BACK='samples/CAM_BACK',
    CAM_BACK_LEFT='samples/CAM_BACK_LEFT',
    CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
    pts_semantic_mask='lidarseg/v1.0-trainval')

train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=4),
    dict(
        type='LoadMultiModalityData',
        superpixel_root='data/superpixels/slic/',
        num_cameras=6),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.55),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[0, 3.14159265359],
        scale_ratio_range=[0.95, 1.05]),
    dict(
        type='ResizedCrop',
        image_crop_size=[224, 448],
        image_crop_ratio=[1.5555555555555556, 1.8888888888888888],
        crop_center=True),
    dict(type='FlipHorizontal'),
    dict(
        type='SuperflowInputs',
        keys=[
            'points', 'imgs', 'pairing_points', 'pairing_images', 'superpixels'
        ])
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='NuScenesSegDataset',
        data_root=data_root,
        ann_file='superflow_nus_info.pkl',
        data_prefix=data_prefix,
        pipeline=train_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        ignore_index=16))

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')
