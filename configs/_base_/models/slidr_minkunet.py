model = dict(
    type='SLidR',
    data_preprocessor=dict(
        type='LiMoEDataPreprocessor',
        H=32,
        W=480,
        fov_up=10.0,
        fov_down=-30.0,
        ignore_index=16,
        voxel_size=[0.1, 1, 0.1],
        voxel_type='cylinder',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    backbone_3d=dict(
        type='MinkUNetBackbone',
        in_channels=4,
        base_channels=32,
        layers=[2, 3, 4, 6, 2, 2, 2, 2],
        planes=[32, 64, 128, 256, 256, 128, 96, 96],
        block_type='basic',
        bn_momentum=0.05),
    head_3d=dict(
        type='LinearHead', channels=96, num_classes=64, dropout_ratio=0),
    backbone_2d=dict(type='ViT', images_encoder='dinov2_vit_base_p14'),
    head_2d=dict(
        type='UpsampleHead', in_channels=768, out_channels=64,
        scale_factor=14),
    superpixel_size=150,
    temperature=0.07)
