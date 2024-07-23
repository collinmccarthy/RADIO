import sys
from pathlib import Path
from typing import List, Union, Sequence, Optional, Tuple

import torch
from timm.models.registry import register_model

# Hack: add top-level module to path until setup.py exists or PYTHON_PATH has been updated
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # Top-level `RADIO` project dir

from radio.eradio_model import FasterViT


class MultiResERADIOModel(FasterViT):

    def _intermediate_stages(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,
    ) -> List[torch.Tensor]:

        # TODO: Then go back and just implement the non-intermediate layer versions and test with a forward
        #       pass via generate_configurable_kwargs.py (and rename this to verify_custom_models.py?)

        outputs, num_levels = [], len(self.levels)
        take_indices = set(range(num_levels - n, num_levels) if isinstance(n, int) else n)

        torch._assert(
            all([0 <= idx < len(self.levels) for idx in take_indices]),
            f"Indicies for extracting intermediate layers must be between 0 and num stages - 1"
            f" = {len(self.levels) - 1}. Requested n={n} which produced indices={take_indices},"
            f" which is invalid.",
        )

        # Follows FasterViT.forward_features() but without output norm
        _, _, H, W = x.shape
        if H % 32 != 0 or W % 32 != 0:
            raise ValueError(
                f"E-RADIO requires input dimensions to be divisible by 32 but got H x W: {H} x {W}"
            )

        x = self.patch_embed(x)
        full_features = None
        for il, level in enumerate(self.levels):
            x, pre_downsample_x = level(x)

            if self.return_full_features or self.use_neck:
                full_features = self.high_res_neck(pre_downsample_x, il, full_features)
                outputs.append(full_features)
            else:
                outputs.append(x)

        outputs = [out for idx, out in enumerate(outputs) if idx in take_indices]
        return outputs

    def get_intermediate_stages(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,
        norm: bool = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        """Intermediate layer accessor following timm.models.VisionTransformer.

        NOTE: Here 'n' refers to stages (or 'levels' in FasterViT) not blocks. We assert that the
        indices are in range.
        """
        # take last n levels if n is an int, if in is a sequence, select by matching indices
        outputs = self._intermediate_stages(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]

        return tuple(outputs)


@register_model
def configurable_eradio_xxxtiny(pretrained=False, **kwargs):
    model = MultiResERADIOModel(
        depths=[1, 3, 4, 5],
        num_heads=[2, 4, 8, 16],
        window_size=[None, None, [16, 16], 16],
        dim=32,
        in_dim=32,
        mlp_ratio=4,
        drop_path_rate=0.0,
        sr_ratio=[1, 1, [2, 1], 1],
        use_swiglu=False,
        yolo_arch=True,
        shuffle_down=False,
        conv_base=True,
        use_neck=True,
        full_features_head_dim=256,
        neck_start_stage=2,
        **kwargs,
    )
    if pretrained:
        model.load_state_dict(torch.load(pretrained))
    return model


@register_model
def configurable_eradio_xxxtiny_8x_ws12(pretrained=False, **kwargs):
    model = MultiResERADIOModel(
        depths=[1, 3, 4, 5],
        num_heads=[2, 4, 8, 16],
        window_size=[None, None, [12, 12], 12],
        dim=32,
        in_dim=32,
        mlp_ratio=4,
        drop_path_rate=0.0,
        sr_ratio=[1, 1, [2, 1], 1],
        use_swiglu=False,
        downsample_shuffle=False,
        yolo_arch=True,
        shuffle_down=False,
        cpb_mlp_hidden=64,
        use_neck=True,
        full_features_head_dim=256,
        neck_start_stage=2,
        conv_groups_ratio=1,
        **kwargs,
    )
    if pretrained:
        model.load_state_dict(torch.load(pretrained)["state_dict"])
    return model


@register_model
def configurable_eradio_xxxtiny_8x_ws16(pretrained=False, **kwargs):
    model = MultiResERADIOModel(
        depths=[1, 3, 4, 5],
        num_heads=[2, 4, 8, 16],
        window_size=[None, None, [16, 16], 16],
        dim=32,
        in_dim=32,
        mlp_ratio=4,
        drop_path_rate=0.0,
        sr_ratio=[1, 1, [2, 1], 1],
        use_swiglu=False,
        downsample_shuffle=False,
        yolo_arch=True,
        shuffle_down=False,
        cpb_mlp_hidden=64,
        use_neck=True,
        full_features_head_dim=256,
        neck_start_stage=1,
        conv_groups_ratio=1,
        **kwargs,
    )
    if pretrained:
        model.load_state_dict(torch.load(pretrained)["state_dict"])
    return model


@register_model
def configurable_fastervit2_large_fullres_ws16(pretrained=False, **kwargs):
    model = MultiResERADIOModel(
        depths=[3, 3, 5, 5],
        num_heads=[2, 4, 8, 16],
        window_size=[None, None, [16, 16], 16],
        dim=192,
        in_dim=64,
        mlp_ratio=4,
        drop_path_rate=0.0,
        sr_ratio=[1, 1, [2, 1], 1],
        use_swiglu=False,
        yolo_arch=True,
        shuffle_down=False,
        conv_base=True,
        use_neck=True,
        full_features_head_dim=1536,
        neck_start_stage=2,
        **kwargs,
    )
    if pretrained:
        model.load_state_dict(torch.load(pretrained)["state_dict"])
    return model


@register_model
def configurable_eradio(pretrained=False, **kwargs):
    return configurable_fastervit2_large_fullres_ws16(pretrained=pretrained, **kwargs)
