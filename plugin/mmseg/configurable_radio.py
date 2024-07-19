import math
import sys
from pathlib import Path
from typing import List, Optional, Union
from argparse import Namespace

import torch
from mmengine.dist import is_distributed, get_rank, barrier
from mmengine.model import BaseModule
from timm.models.vision_transformer import VisionTransformer
from mmseg.models.builder import BACKBONES

# Hack: add top-level module to path until setup.py exists or PYTHON_PATH has been updated
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # Top-level `RADIO` project dir

from plugin.radio.configurable_radio_model import ConfigurableRADIOModel


@BACKBONES.register_module()
class ConfigurableRADIO(ConfigurableRADIOModel, BaseModule):
    """Wrapper ConfigurableRADIOModel for mmsegmentation.

    NOTE: Inputs should NOT be normalized outside of this class, e.g. in SegDataPreProcessor or
    DetDataPreProcessor. They will be normalized to [0,1] here then the conditioner will normalize
    using the default mean/std which is in this 0-1 scale. For the default mean/std values, see
    ConfigurableRADIOModel which uses timm.data.constants.OPENAI_CLIP_MEAN and OPEN_AI_CLIP_STD just
    like hubconf.radio_model() does via get_default_conditioner().

    Similar to radio.mmseg.radio.RADIO wtih a few changes:
    - Uses kwarg-based initialization to directly initialize ConfigurableRADIOModel
    - Removes teachers and re-uses RADIOModel.forward()
    - Initializes weights via load_radio_model_from_state_dict instead of directly in __init__, to
      maintain standard mmengine workflow
    """

    def __init__(
        self,
        *args,
        frozen: bool = True,
        skip_conditioner: bool = False,
        teachers: list[dict],
        ignore_teachers: bool = True,
        cast_outputs_to_fp32: bool = False,
        init_cfg: Optional[dict] = None,
        **kwargs,
    ) -> None:
        assert len(teachers) == 0 or ignore_teachers, (
            "Using ignore_teachers=False is not currently supported because the summary outputs"
            " are not being returned from forward(). To use teachers, extend this class"
            " and override both __init__() and forward()"
        )
        ConfigurableRADIOModel.__init__(
            self,
            *args,
            teachers=teachers,
            ignore_teachers=ignore_teachers,
            cast_outputs_to_fp32=cast_outputs_to_fp32,
            **kwargs,
        )
        BaseModule.__init__(self, init_cfg=init_cfg)

        conditioner_mean = self.input_conditioner.norm_mean
        conditioner_std = self.input_conditioner.norm_std
        if not (0 < conditioner_mean < 255 and 0 < conditioner_std < 255):
            raise RuntimeError(
                f"Expected conditioner_mean={conditioner_mean} and "
                f"conditioner_std={conditioner_std} between 0 and 1 because we are normalizing"
                f" the inputs in this range always."
            )

        self._frozen_features = frozen
        self._skip_conditioner = skip_conditioner

    def make_preprocessor_external(self):
        raise RuntimeError(
            "ConfigurableRADIO scales inputs to range [0,1] before using input_conditioner."
            " If making preprocessor external, the inputs will still be scaled in this method which"
            " will require changing the input conditioner. Recommend keeping the default"
            " conditioner and passing in unnormalized inputs."
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass, overall the implementation as radio.mmseg.RADIO.forward().

        A few differences:
        - We only detach if self._frozen_features=True, so we can try fine-tuning the features
        - We use ConfigurableRADIOModel.forward() which doesn't automatically cast to fp32, and
          we set cast_outputs_to_fp32 as False by default in __init__()
        - We assert features shape is 4-dimensional, e.g. are NHWC
        """
        _, _, H, W = x.shape

        # Scale inputs to the range [0,1] for default conditioner
        x = x / 255.0

        _, features = self(x)

        if isinstance(self.model, VisionTransformer):
            # Reshape
            B, _, C = features.shape

            if hasattr(self.model, "patch_generator"):
                # Cropped Positional Embedding (CPE) case
                patch_height = patch_width = self.model.patch_generator.patch_size
            else:
                # Standard ViT case.
                patch_height, patch_width = self.model.patch_embed.patch_size

            # We only need to reshape and permute for RADIO (ViT); E-RADIO is already NHWC
            features = (
                features.reshape(B, math.ceil(H / patch_height), math.ceil(W / patch_width), C)
                .permute(0, 3, 1, 2)
                .contiguous()
            )

        torch._assert(
            features.ndim == 4, f"Expected NHWC features tensor, found ndim={features.ndim}"
        )

        if self._frozen_features:
            # Prevent gradients from flowing back of features are frozen
            # If we allow features to be updated using solely the downstream task loss we will lose
            # the ability to use any of the adaptors for other VLM tasks
            features = features.detach()

        return [features]

    def train(self, mode=True):
        """Intercept call."""
        raise NotImplementedError("Training mode not implemented, only eval mode (frozen weights)")

    def init_weights(self):
        # This is a no-op as the model weights are loaded during instantiation.
        if isinstance(self.init_cfg, dict) and self.init_cfg.get("type") == "Pretrained":
            checkpoint = self.init_cfg.get("checkpoint", None)
            if checkpoint is None:
                raise RuntimeError(
                    f"ConfigurableRADIO expects checkpoint URL, e.g."
                    f" https://huggingface.co/nvidia/RADIO/resolve/main/radio_v2.1_bf16.pth.tar?download=true"
                )

            # Following radio.mmseg.radio.RADIO for pulling on rank 0 first then the rest
            if not is_distributed() or get_rank() == 0:
                self.load_pretrained_state_dict(url=checkpoint)  # Download on rank 0 to cache
            if is_distributed():
                barrier()
                if get_rank() > 0:
                    self.load_pretrained_state_dict(url=checkpoint)  # Download on other ranks

        else:
            raise ValueError(f"Unhandled case: {self.init_cfg}")
