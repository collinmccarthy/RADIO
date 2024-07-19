# fmt: off
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

dependencies = ["torch", "timm", "einops"]

import os
from typing import Dict, Any, Optional, Union, List
from argparse import Namespace
import warnings

import torch
from torch.hub import load_state_dict_from_url

from timm.models import clean_state_dict
from timm.data.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

from radio.adaptor_registry import adaptor_registry
from radio.common import DEFAULT_VERSION, RadioResource, RESOURCE_MAP
from radio.enable_spectral_reparam import disable_spectral_reparam
from radio.radio_model import RADIOModel, Resolution, create_model_from_args, create_model_from_kwargs
from radio.input_conditioner import get_default_conditioner, get_default_conditioner_from_kwargs
from radio.vitdet import apply_vitdet_arch, VitDetArgs


def radio_model(
    version: str = "",
    progress: bool = True,
    adaptor_names: Union[str, List[str]] = None,
    vitdet_window_size: Optional[int] = None,
    **kwargs,
) -> RADIOModel:
    if not version:
        version = DEFAULT_VERSION

    if os.path.isfile(version):
        chk = torch.load(version, map_location="cpu")
        resource = RadioResource(version, patch_size=None, max_resolution=None, preferred_resolution=None)
    else:
        resource = RESOURCE_MAP[version]
        chk = load_state_dict_from_url(
            resource.url, progress=progress, map_location="cpu"
        )

    if "state_dict_ema" in chk:
        state_dict = chk["state_dict_ema"]
        chk['args'].spectral_reparam = False
    else:
        state_dict = chk["state_dict"]

    mod = create_model_from_args(chk["args"])

    state_dict = clean_state_dict(state_dict)

    key_warn = mod.load_state_dict(get_prefix_state_dict(state_dict, "base_model."), strict=False)
    if key_warn.missing_keys:
        warnings.warn(f'Missing keys in state dict: {key_warn.missing_keys}')
    if key_warn.unexpected_keys:
        warnings.warn(f'Unexpected keys in state dict: {key_warn.unexpected_keys}')

    if getattr(chk['args'], "spectral_reparam", False):
        # Spectral reparametrization uses PyTorch's "parametrizations" API. The idea behind
        # the method is that instead of there being a `weight` tensor for certain Linear layers
        # in the model, we make it a dynamically computed function. During training, this
        # helps stabilize the model. However, for downstream use cases, it shouldn't be necessary.
        # Disabling it in this context means that instead of having `w' = f(w)`, we just compute `w' = f(w)`
        # once, during this function call, and replace the parametrization with the realized weights.
        # This makes the model run faster, and also use less memory.
        disable_spectral_reparam(mod)
        chk['args'].spectral_reparam = False

    conditioner = get_default_conditioner()
    conditioner.load_state_dict(get_prefix_state_dict(state_dict, "input_conditioner."))

    dtype = getattr(chk['args'], 'dtype', torch.float32)
    mod.to(dtype=dtype)
    conditioner.dtype = dtype

    summary_idxs = torch.tensor([
        i
        for i, t in enumerate(chk["args"].teachers)
        if t.get("use_summary", True)
    ], dtype=torch.int64)

    if adaptor_names is None:
        adaptor_names = []
    elif isinstance(adaptor_names, str):
        adaptor_names = [adaptor_names]

    teachers = chk["args"].teachers
    adaptors = dict()
    for adaptor_name in adaptor_names:
        for tidx, tconf in enumerate(teachers):
            if tconf["name"] == adaptor_name:
                break
        else:
            raise ValueError(f'Unable to find the specified adaptor name. Known names: {list(t["name"] for t in teachers)}')

        ttype = tconf["type"]

        pf_idx_head = f'_heads.{tidx}'
        pf_name_head = f'_heads.{adaptor_name}'
        pf_idx_feat = f'_feature_projections.{tidx}'
        pf_name_feat = f'_feature_projections.{adaptor_name}'

        adaptor_state = dict()
        for k, v in state_dict.items():
            if k.startswith(pf_idx_head):
                adaptor_state['summary' + k[len(pf_idx_head):]] = v
            elif k.startswith(pf_name_head):
                adaptor_state['summary' + k[len(pf_name_head):]] = v
            elif k.startswith(pf_idx_feat):
                adaptor_state['feature' + k[len(pf_idx_feat):]] = v
            elif k.startswith(pf_name_feat):
                adaptor_state['feature' + k[len(pf_name_feat):]] = v

        adaptor = adaptor_registry.create_adaptor(ttype, chk["args"], tconf, adaptor_state)
        adaptor.head_idx = tidx
        adaptors[adaptor_name] = adaptor

    radio = RADIOModel(
        mod,
        conditioner,
        summary_idxs=summary_idxs,
        patch_size=resource.patch_size,
        max_resolution=resource.max_resolution,
        window_size=vitdet_window_size,
        preferred_resolution=resource.preferred_resolution,
        adaptors=adaptors,
    )

    if vitdet_window_size is not None:
        apply_vitdet_arch(mod, VitDetArgs(vitdet_window_size, radio.num_summary_tokens))

    return radio


# fmt: on
def radio_model_from_kwargs(
    # Kwargs for RADIOModel
    patch_size: int,
    max_resolution: int,
    preferred_resolution: Union[dict, Resolution],
    adaptor_cfgs: Optional[dict[str, Union[dict, Namespace]]],  # instead of adaptor_names
    vitdet_window_size: Optional[int],
    # Kwargs for create_model()
    model: str,  # chk['args'].model
    in_chans: Optional[int],  # chk['args'].in_chans
    input_size: Optional[int],  # chk['args'].input_size
    pretrained: bool,  # chk['args'].pretrained
    num_classes: int,  # chk['args'].num_classes
    drop: float,  # chk['args'].drop
    drop_path: Optional[float],  # chk['args'].drop_path
    drop_block: Optional[float],  # chk['args'].drop_block
    global_pool: Optional[str],  # chk['args'].gp
    bn_momentum: Optional[float],  # chk['args'].bn_momentum
    bn_eps: Optional[float],  # chk['args'].bn_eps
    initial_checkpoint: str,  # chk['args'].initial_checkpoint
    torchscript: bool,  # chk['args'].torchscript
    cls_token_per_teacher: bool,  # chk['args'].cls_token_per_teacher
    cpe_max_size: Optional[int],  # chk['args'].cpe_max_size
    model_kwargs: dict,  # chk['args'].model_kwargs, e.g. {'return_full_features': True} for E-RADIO
    teachers: List[dict],  # chk['args'].teachers
    register_multiple: int,  # chk['args'].register_multiple
    spectral_reparam: bool,  # chk['args'].spectral_reparam
    model_norm: bool,  # chk['args'].model_norm
    # Kwargs for conditioner
    dtype: torch.dtype,
    # Other kwargs
    disable_spectral_reparam: bool = True,  # Default for radio_model_from_args()
) -> Union[RADIOModel, tuple[RADIOModel, dict, dict]]:
    create_model_kwargs = dict(
        model=model,
        in_chans=in_chans,
        input_size=input_size,
        pretrained=pretrained,
        num_classes=num_classes,
        drop=drop,
        drop_path=drop_path,
        drop_block=drop_block,
        global_pool=global_pool,
        bn_momentum=bn_momentum,
        bn_eps=bn_eps,
        initial_checkpoint=initial_checkpoint,
        torchscript=torchscript,
        cls_token_per_teacher=cls_token_per_teacher,
        cpe_max_size=cpe_max_size,
        model_kwargs=model_kwargs,
        num_teachers=len(teachers),
        register_multiple=register_multiple,
        spectral_reparam=spectral_reparam,
        model_norm=model_norm,
    )
    mod = create_model_from_kwargs(**create_model_kwargs)

    if spectral_reparam and disable_spectral_reparam:
        # Spectral reparametrization uses PyTorch's "parametrizations" API. The idea behind
        # the method is that instead of there being a `weight` tensor for certain Linear layers
        # in the model, we make it a dynamically computed function. During training, this
        # helps stabilize the model. However, for downstream use cases, it shouldn't be necessary.
        # Disabling it in this context means that instead of having `w' = f(w)`, we just compute `w' = f(w)`
        # once, during this function call, and replace the parametrization with the realized weights.
        # This makes the model run faster, and also use less memory.
        disable_spectral_reparam(mod)
        spectral_reparam = False

    conditioner_kwargs = dict(
        input_scale=1.0, norm_mean=OPENAI_CLIP_MEAN, norm_std=OPENAI_CLIP_STD, dtype=dtype
    )
    conditioner = get_default_conditioner_from_kwargs(**conditioner_kwargs)

    mod.to(dtype=dtype)
    assert conditioner.dtype == dtype, "Failed to maintain correct dtype"

    summary_idxs = torch.tensor(
        [i for i, t in enumerate(teachers) if t.get("use_summary", True)], dtype=torch.int64
    )

    if adaptor_cfgs is None:
        adaptor_cfgs = {}
    assert isinstance(
        adaptor_cfgs, dict
    ), "Expected adaptor_cfgs to be dict from adaptor names to per-adaptor configs"

    teachers = teachers
    adaptors = dict()
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
            state=dict(),  # No adaptor_state here, moved to radio_model_load_state_dict()
        )
        adaptor.head_idx = tidx
        adaptors[adaptor_name] = adaptor

    # Re-factor RADIOModel args into radio_kwargs dict to return
    preferred_resolution_error_msg = (
        f"Expected preferred_resolution to be of type Resolution or dict, with keys 'height'"
        f" and 'width'. Found type {type(preferred_resolution).__name__}: {preferred_resolution}"
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

    radio_kwargs = dict(
        input_conditioner=conditioner,
        patch_size=patch_size,
        max_resolution=max_resolution,
        preferred_resolution=preferred_resolution,
        summary_idxs=summary_idxs,
        window_size=vitdet_window_size,
        adaptors=adaptors,
    )

    radio = RADIOModel(mod, **radio_kwargs)

    if vitdet_window_size is not None:
        apply_vitdet_arch(mod, VitDetArgs(vitdet_window_size, radio.num_summary_tokens))

    inner_kwargs = dict(
        create_model_kwargs=create_model_kwargs,
        conditioner_kwargs=conditioner_kwargs,
        radio_kwargs=radio_kwargs,
    )
    return radio, inner_kwargs


def radio_model_load_state_dict(
    model: RADIOModel, url: str, model_has_spectral_reaparm: bool = False, progress: bool = True
) -> None:
    chk = load_state_dict_from_url(url, progress=progress, map_location="cpu")

    if "state_dict_ema" in chk:
        state_dict = chk["state_dict_ema"]
        force_disable_spectral_reparam = True
    else:
        state_dict = chk["state_dict"]

    state_dict = clean_state_dict(state_dict)

    # Load base model state
    base_model_state_dict = get_prefix_state_dict(state_dict, "base_model.")
    key_warn = model.model.load_state_dict(base_model_state_dict, strict=False)
    if key_warn.missing_keys:
        warnings.warn(f"Missing keys in state dict: {key_warn.missing_keys}")
    if key_warn.unexpected_keys:
        warnings.warn(f"Unexpected keys in state dict: {key_warn.unexpected_keys}")

    chkpt_spectral_reparam = "args" in chk and getattr(chk["args"], "spectral_reparam", False)
    if (model_has_spectral_reaparm or chkpt_spectral_reparam) and force_disable_spectral_reparam:
        disable_spectral_reparam(model.model)

    # Load conditioner state
    conditioner_state_dict = get_prefix_state_dict(state_dict, "input_conditioner.")
    model.input_conditioner.load_state_dict(conditioner_state_dict)

    for adaptor_name, adaptor_module in model.adaptors.items():
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


# fmt: off

def get_prefix_state_dict(state_dict: Dict[str, Any], prefix: str):
    mod_state_dict = {
        k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)
    }
    return mod_state_dict
