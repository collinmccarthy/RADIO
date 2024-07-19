import math
from typing import List, Optional

import torch
from mmengine.model import BaseModule
from timm.models.vision_transformer import VisionTransformer
from mmseg.models.builder import BACKBONES

from ..radio.radio_model import RADIOModel


@BACKBONES.register_module()
class MMSegRadio(RADIOModel, BaseModule):

    def __init__(self, *args, init_cfg: Optional[dict] = None, **kwargs) -> None:
        RADIOModel.__init__(self, *args, **kwargs)  # Normal model init
        BaseModule.__init__(
            self, init_cfg=init_cfg
        )  # Pass in init_cfg for init_weights()

        # Unique to radio, always eval
        self.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass."""
        _, _, H, W = x.shape

        # Scale inputs to the range [0, 1].
        x = x / 255.0

        _, features = self(x)

        if isinstance(self.model, VisionTransformer):
            # Reshape
            B, _, C = features.shape

            if hasattr(self.model, "patch_generator"):
                # Cropped Positional Embedding (CPE) case.
                patch_height = patch_width = self.model.patch_generator.patch_size
            else:
                # Standard ViT case.
                patch_height, patch_width = self.model.patch_embed.patch_size
            features = (
                features.reshape(
                    B, math.ceil(H / patch_height), math.ceil(W / patch_width), C
                )
                .permute(0, 3, 1, 2)
                .contiguous()
            )

        # IMPORTANT: prevent gradients from flowing back towards the backbone.
        features = features.detach()

        return [features]

    def train(self, mode=True):
        """Intercept call."""
        raise NotImplementedError(
            "Training mode not implemented, only eval mode (frozen weights)"
        )

    # def init_weights(self):
    #     # This is a no-op as the model weights are loaded during instantiation.
    #     if (isinstance(self.init_cfg, dict)
    #             and self.init_cfg.get('type') == 'Pretrained'):
    #         pass
    #     else:
    #         raise ValueError(f"Unhandled case: {self.init_cfg}")
