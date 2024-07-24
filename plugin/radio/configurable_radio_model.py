from typing import List, Optional, Union, Tuple
from argparse import Namespace
from pathlib import Path
import warnings
import inspect
import sys

import torch
from torch import Tensor
from torch.hub import load_state_dict_from_url
from timm.models import clean_state_dict, VisionTransformer
from timm.data.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD


# Hack: add top-level module to path until setup.py exists or PYTHON_PATH has been updated
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # Top-level `RADIO` project dir

from radio import eradio_model
from radio.radio_model import RADIOModel, Resolution, create_model_from_args
from radio.input_conditioner import InputConditioner
from radio.enable_spectral_reparam import disable_spectral_reparam
from radio.adaptor_base import RadioOutput, AdaptorInput
from radio.adaptor_registry import adaptor_registry
from radio.vitdet import apply_vitdet_arch, VitDetArgs
from hubconf import get_prefix_state_dict

from plugin.radio.multires_eradio_model import MultiResERADIOModel


class ConfigurableRADIOModel(RADIOModel):
    """
    A wrapper around RADIOModel that takes in only primitive data types to initialize the model.
    Modified from radio.hubconf.radio_model() to use explicit kwargs instead of checkpoint args.

    This is useful for when we need to make modifications to the model, wrap it, or just build it
    entirely via a config file without relying on specific checkpoints.

    There are very few defaults here because the original RADIO models do not have any exposed
    default values, so we're not assuming any specific values except for a few defaults implicitly
    used in the building process in hbconf.py.

    See radio/scripts/generate_configurable_kwargs.py for how to generate the kwargs here for
    existing radio checkpoints, while verifying the model is the exact same as initializing via
    checkpoints in the normal way with hubconf.radio_model().
    """

    def __init__(
        self,
        # Kwargs for RADIOModel
        patch_size: int,
        max_resolution: int,
        preferred_resolution: Union[dict, Resolution],
        adaptor_cfgs: Optional[dict[str, Union[dict, Namespace]]],  # instead of adaptor_names
        vitdet_window_size: Optional[int],
        vitdet_num_windowed: Optional[int],
        vitdet_num_global: Optional[int],
        # Kwargs for create_model(), from checkpoint['args'] with same names unless specified
        create_model_kwargs: dict,
        # model: str,
        # in_chans: Optional[int],
        # input_size: Optional[int],
        # pretrained: bool,
        # num_classes: int,
        # drop: float,
        # drop_path: Optional[float],
        # drop_block: Optional[float],
        # gp: Optional[str],  # aka global pool
        # bn_momentum: Optional[float],
        # bn_eps: Optional[float],
        # initial_checkpoint: str,
        # torchscript: bool,
        # cls_token_per_teacher: bool,
        # cpe_max_size: Optional[int],
        # model_kwargs: dict,  #  e.g. {'return_full_features': True} for E-RADIO
        # teachers: List[dict],
        # register_multiple: int,
        # spectral_reparam: bool,
        # model_norm: bool,
        # Kwargs for conditioner
        dtype: Union[str, torch.dtype],
        input_scale: float = 1.0,
        norm_mean: tuple[float, float, float] = OPENAI_CLIP_MEAN,
        norm_std: tuple[float, float, float] = OPENAI_CLIP_STD,
        # Other kwargs
        cast_outputs_to_fp32: bool = True,  # Default behavior for RADIOModel
        disable_spectral_reparam: bool = True,  # Default for radio_model_from_args()
        ignore_teachers: bool = False,  # Set to True when only using as a backbone
        pretrained_url: Optional[str] = None,  # Pass in url to load pretrained model here
        out_indices_layers: Optional[list[int]] = None,  # For get_intermediate_layers() (ViT)
        out_indices_stages: Optional[list[int]] = None,  # For get_intermediate_stages() (FasterViT)
    ) -> None:
        # Store all kwargs for analyzing later (following https://stackoverflow.com/a/73158104/12422298)
        func_params = inspect.signature(ConfigurableRADIOModel.__init__).parameters
        self._all_kwargs = {
            key: val for key, val in locals().items() if key in func_params and key != "self"
        }

        self._cast_outputs_to_fp32 = cast_outputs_to_fp32

        if ignore_teachers:
            teachers = []

        # self._create_model_kwargs = dict(
        #     model=model,
        #     in_chans=in_chans,
        #     input_size=input_size,
        #     pretrained=pretrained,
        #     num_classes=num_classes,
        #     drop=drop,
        #     drop_path=drop_path,
        #     drop_block=drop_block,
        #     gp=gp,
        #     bn_momentum=bn_momentum,
        #     bn_eps=bn_eps,
        #     initial_checkpoint=initial_checkpoint,
        #     torchscript=torchscript,
        #     cls_token_per_teacher=cls_token_per_teacher,
        #     cpe_max_size=cpe_max_size,
        #     model_kwargs=model_kwargs,
        #     teachers=teachers,
        #     register_multiple=register_multiple,
        #     spectral_reparam=spectral_reparam,
        #     model_norm=model_norm,
        # )
        self._create_model_kwargs = create_model_kwargs
        model = create_model_from_args(Namespace(**create_model_kwargs))

        # Currently we use out_indices_layers for ViT and out_indices_stages for FasterViT
        # Any other backbones should be configured independently of RADIO
        assert not (
            out_indices_layers is not None and out_indices_stages is not None
        ), "Cannot set both out_indices_layers and out_indices_stages"

        if out_indices_layers is not None:
            assert isinstance(model, VisionTransformer) and model.hasattr(
                "get_intermediate_layers"
            ), f"Expected VisionTransformer model when out_indices_layers is not None"
        self._out_indices_layers = out_indices_layers

        if out_indices_stages is not None:
            assert isinstance(model, MultiResERADIOModel) and model.hasattr(
                "get_intermediate_stages"
            ), f"Expected MultiResERADIOModel when out_indices_stages is not None"
        self._out_indices_stages = out_indices_stages

        spectral_reparam = create_model_kwargs.get("spectral_reparam", False)
        if spectral_reparam and disable_spectral_reparam:
            # Spectral reparametrization uses PyTorch's "parametrizations" API. The idea behind
            # the method is that instead of there being a `weight` tensor for certain Linear layers
            # in the model, we make it a dynamically computed function. During training, this
            # helps stabilize the model. However, for downstream use cases, it shouldn't be necessary.
            # Disabling it in this context means that instead of having `w' = f(w)`, we just compute `w' = f(w)`
            # once, during this function call, and replace the parametrization with the realized weights.
            # This makes the model run faster, and also use less memory.
            disable_spectral_reparam(model)
            spectral_reparam = False

        if isinstance(dtype, str):
            if dtype == "float32":
                dtype = torch.float32
            elif dtype == "float16":
                dtype = torch.float16
            elif dtype == "bfloat16":
                dtype = torch.bfloat16
            else:
                raise RuntimeError(f"Unsupported dtype str value: {dtype}. ")

        self._spectral_reparam = spectral_reparam
        self._conditioner_kwargs = dict(
            input_scale=input_scale,
            norm_mean=norm_mean,
            norm_std=norm_std,
            dtype=dtype,
        )
        conditioner = InputConditioner(**self._conditioner_kwargs)

        model.to(dtype=dtype)
        assert conditioner.dtype == dtype, "Failed to maintain correct dtype"

        if adaptor_cfgs is None:
            adaptor_cfgs = {}
        assert isinstance(
            adaptor_cfgs, dict
        ), "Expected adaptor_cfgs to be dict from adaptor names to per-adaptor configs"

        adaptors = dict()
        teachers = create_model_kwargs.get("teachers", [])

        summary_idxs = torch.tensor(
            [i for i, t in enumerate(teachers) if t.get("use_summary", True)], dtype=torch.int64
        )

        for adaptor_name, adaptor_cfg in adaptor_cfgs.items():
            for tidx, tconf in enumerate(teachers):
                if tconf["name"] == adaptor_name:
                    break
            else:
                raise ValueError(
                    f'Unable to find the specified adaptor name. Known names: {list(t["name"] for t in teachers)}'
                )

            if isinstance(adaptor_cfg, dict):
                adaptor_cfg = Namespace(**adaptor_cfg)
            assert isinstance(adaptor_cfg, Namespace), (
                f"Expected adaptor_cfgs to be a dictionary with values containing either Namespace"
                f" objects or dicts. Found adaptor_cfg[{adaptor_name}] with type"
                f" {type(adaptor_cfg).__name__}: {adaptor_cfg}"
            )

            ttype = tconf["type"]
            adaptor = adaptor_registry.create_adaptor(
                name=ttype,
                main_cfg=adaptor_cfg,
                adaptor_config=tconf,
                state=dict(),  # No adaptor_state here, moved to load_from_state_dict()
            )
            adaptor.head_idx = tidx
            adaptors[adaptor_name] = adaptor

        # Re-factor RADIOModel args into radio_kwargs dict to return
        preferred_resolution_error_msg = (
            f"Expected preferred_resolution to be of type Resolution or dict with keys 'height' and"
            f" 'width'. Found type {type(preferred_resolution).__name__}: {preferred_resolution}"
        )
        if isinstance(preferred_resolution, dict):
            assert (
                len(preferred_resolution) == 2
                and "height" in preferred_resolution
                and "width" in preferred_resolution
            ), f"{preferred_resolution_error_msg}"
            preferred_resolution = Resolution(
                height=preferred_resolution["height"],
                width=preferred_resolution["width"],
            )

        assert isinstance(preferred_resolution, Resolution), preferred_resolution_error_msg

        self._radio_kwargs = dict(
            patch_size=patch_size,
            max_resolution=max_resolution,
            preferred_resolution=preferred_resolution,
            summary_idxs=summary_idxs,
            window_size=vitdet_window_size,
            adaptors=adaptors,
        )

        # Now build RADIOModel using model, conditioner and radio kwargs
        # Not including model/conditioner in wargs b/c these are separate modules initialized above
        super().__init__(model=model, input_conditioner=conditioner, **self._radio_kwargs)

        if vitdet_window_size is not None:
            apply_vitdet_arch(
                model,
                VitDetArgs(
                    vitdet_window_size,
                    self.num_summary_tokens,
                    num_windowed=vitdet_num_windowed,
                    num_global=vitdet_num_global,
                ),
            )

        if pretrained_url is not None:
            # Load state dict after, so the creation and loading are independent (for mmdetection)
            self.load_pretrained_state_dict(url=pretrained_url)

    def forward(self, x: torch.Tensor) -> Union[RadioOutput, list[RadioOutput]]:
        """Same as RADIOModel.forward() but don't automatically cast the features to fp32.

        Using fp32 is prohibitive for super high-resolution images.
        """
        res_step = self.min_resolution_step
        if res_step is not None and (x.shape[-2] % res_step != 0 or x.shape[-1] % res_step != 0):
            raise ValueError(
                "The input resolution must be a multiple of `self.min_resolution_step`. "
                "`self.get_nearest_supported_resolution(<height>, <width>) is provided as a convenience API. "
                f"Input: {x.shape[-2:]}, Nearest: {self.get_nearest_supported_resolution(*x.shape[-2:])}"
            )

        x = self.input_conditioner(x)

        # Return multi-resolution features matching self.out_indices
        if self._out_indices_layers is not None:
            self.model: VisionTransformer
            torch._assert(isinstance(self.model, VisionTransformer))
            features: list[Tensor] = self.model.get_intermediate_layers(
                x,
                n=self._out_indices_layers,
                norm=self._intermediate_features_norm,
            )
        elif self._out_indices_stages is not None:
            self.model: MultiResERADIOModel
            torch._assert(isinstance(self.model, MultiResERADIOModel))
            features: list[Tensor] = self.model.get_intermediate_stages(
                x,
                n=self._out_indices_stages,
                norm=self._intermediate_features_norm,
            )
        else:
            y = self.model.forward_features(x)
            features = [y]

        outputs: list[RadioOutput] = []
        for y in features:

            if isinstance(self.model, VisionTransformer):
                patch_gen = getattr(self.model, "patch_generator", None)
                if patch_gen is not None:
                    all_summary = y[:, : patch_gen.num_cls_tokens]
                    if self.summary_idxs is not None:
                        bb_summary = all_summary[:, self.summary_idxs]
                    else:
                        bb_summary = all_summary
                    all_feat = y[:, patch_gen.num_skip :]
                elif self.model.global_pool == "avg":
                    all_summary = y[:, self.model.num_prefix_tokens :].mean(dim=1)
                    bb_summary = all_summary
                    all_feat = y
                else:
                    all_summary = y[:, 0]
                    bb_summary = all_summary
                    all_feat = y[:, 1:]
            elif isinstance(self.model, eradio_model.FasterViT):
                _, f = y
                all_feat = f.flatten(2).transpose(1, 2)
                all_summary = all_feat.mean(dim=1)
                bb_summary = all_summary
            elif isinstance(y, (list, tuple)):
                all_summary, all_feat = y
                bb_summary = all_summary
            else:
                raise ValueError("Unsupported model type")

            ret = RadioOutput(bb_summary.flatten(1), all_feat)

            if self._cast_outputs_to_fp32:
                all_feat = all_feat.float()
                ret = ret.to(torch.float32)

            if self.adaptors:
                ret = dict(backbone=ret)
                for name, adaptor in self.adaptors.items():
                    if all_summary.ndim == 3:
                        summary = all_summary[:, adaptor.head_idx]
                    else:
                        summary = all_summary

                    if self._cast_outputs_to_fp32:
                        summary = summary.float()

                    ada_input = AdaptorInput(images=x, summary=summary, features=all_feat)
                    v = adaptor(ada_input).to(torch.float32)
                    ret[name] = v

            outputs.append(ret)

        # Return a list only if out_indices is not None
        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs

    @property
    def create_model_kwargs(self) -> dict:
        return self._create_model_kwargs

    @property
    def conditioner_kwargs(self) -> dict:
        return self._conditioner_kwargs

    @property
    def radio_kwargs(self) -> dict:
        return self._radio_kwargs

    @property
    def all_kwargs(self) -> dict:
        return self._all_kwargs

    @property
    def spectral_reparam(self) -> bool:
        return self._spectral_reparam

    def load_pretrained_state_dict(self, url: str, progress: bool = True) -> None:
        """Modified from radio.hubconf.radio_model() to separate model initialization and loading"""
        chk = load_state_dict_from_url(url, progress=progress, map_location="cpu")

        if "state_dict_ema" in chk:
            state_dict = chk["state_dict_ema"]
            force_disable_spectral_reparam = True
        else:
            state_dict = chk["state_dict"]

        state_dict = clean_state_dict(state_dict)

        # Load base model state (not sure why this is necessary to do separately, besides prefix?)
        base_model_state_dict = get_prefix_state_dict(state_dict, "base_model.")
        key_warn = self.model.load_state_dict(base_model_state_dict, strict=False)
        if key_warn.missing_keys:
            warnings.warn(f"Missing keys in state dict: {key_warn.missing_keys}")
        if key_warn.unexpected_keys:
            warnings.warn(f"Unexpected keys in state dict: {key_warn.unexpected_keys}")

        chkpt_spectral_reparam = "args" in chk and getattr(chk["args"], "spectral_reparam", False)
        if chkpt_spectral_reparam and self.spectral_reparam and force_disable_spectral_reparam:
            disable_spectral_reparam(self.model)

        # Load conditioner state (not sure why this is necessary to do separately?)
        conditioner_state_dict = get_prefix_state_dict(state_dict, "input_conditioner.")
        self.input_conditioner.load_state_dict(conditioner_state_dict)

        # Load adaptor states
        for adaptor_name, adaptor_module in self.adaptors.items():
            tidx = adaptor_module.head_idx
            pf_idx_head = f"_heads.{tidx}"
            pf_name_head = f"_heads.{adaptor_name}"
            pf_idx_feat = f"_feature_projections.{tidx}"
            pf_name_feat = f"_feature_projections.{adaptor_name}"

            adaptor_state = dict()
            for k, v in state_dict.items():
                if k.startswith(pf_idx_head):
                    adaptor_state["summary" + k[len(pf_idx_head) :]] = v
                elif k.startswith(pf_name_head):
                    adaptor_state["summary" + k[len(pf_name_head) :]] = v
                elif k.startswith(pf_idx_feat):
                    adaptor_state["feature" + k[len(pf_idx_feat) :]] = v
                elif k.startswith(pf_name_feat):
                    adaptor_state["feature" + k[len(pf_name_feat) :]] = v

            adaptor_module.load_state_dict(adaptor_state)
