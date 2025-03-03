import torch
import torch.nn as nn
from mmdet3d.registry import MODELS
from torch import Tensor

from .dinov2 import dinov2_vision_transformer as dinov2_vit

DINOv2_MODELS = {
    'dinov2_vit_small_p14': ('dinov2_vits14', 14, 384),
    'dinov2_vit_base_p14': ('dinov2_vitb14', 14, 768),
    'dinov2_vit_large_p14': ('dinov2_vitl14', 14, 1024)
}


@MODELS.register_module()
class ViT(nn.Module):

    def __init__(self,
                 images_encoder: str,
                 feat: str = 'x_pre_norm',
                 height: int = 224,
                 width: int = 448) -> None:
        super(ViT, self).__init__()

        # ViT parameters
        model_name, patch_size, embed_dim = DINOv2_MODELS.get(images_encoder)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.which_feature = feat

        # Compute feature size
        assert (height % self.patch_size) == 0
        assert (width % self.patch_size) == 0
        self.f_height = height // self.patch_size
        self.f_width = width // self.patch_size

        # Load ViT
        self.encoder = dinov2_vit.__dict__[model_name](
            patch_size=patch_size, pretrained=True)

        # Teacher must stay frozen
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

    def forward(self, x: Tensor) -> Tensor:

        # Go through frozen encoder
        with torch.no_grad():
            batch_size = x.shape[0]

            output = self.encoder.forward_get_last_n(x)
            feat = output[self.which_feature]
            x = torch.cat(feat, dim=2)

            # Remove the CLS token and reshape the patch token features.
            x = (
                x[:, 1:, :].transpose(1, 2).view(batch_size, self.embed_dim,
                                                 self.f_height, self.f_width))

        return x
