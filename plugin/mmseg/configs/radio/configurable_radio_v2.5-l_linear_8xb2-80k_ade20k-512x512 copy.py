# fmt: off
import os
import torch

_base_ = [
    "./configurable_radio_v2.1_linear_8xb2-80k_ade20k-512x512.py"
]

model = dict(
    backbone=dict(
        type="MMDetRADIO",
        frozen=True,
        init_cfg=dict(
            type="Pretrained",
            checkpoint="https://huggingface.co/nvidia/RADIO/resolve/main/radio-v2.5-l_half.pth.tar?download=true",
        ),
        # Remaining args from output of test_configurable_radio(), with a few changes:
        # - Set `teachers` to empty list in create_model_kwargs()
        # - Change preferred_resolution to dict instead of Resolution
        # - Change `dtype` to string, e.g. `"float32"` instead of `torch.float32`
        # - Drop conditioner defaults (`input_scale`, `norm_mean`, `norm_std`)
        # - Change other kwargs:
        #   - Move `pretrained_url` to `init_cfg.checkpoint` above, don't specify `pretrained_url`
        #   - Set `ignore_teachers=True` (not technically necessary but more self-documenting)
        patch_size=16,
        max_resolution=2048,
        preferred_resolution=dict(height=768, width=768),
        adaptor_cfgs=None,
        vitdet_window_size=None,
        vitdet_num_windowed=None,
        vitdet_num_global=4,
        create_model_kwargs=dict(
            model='vit_large_patch16_224',
            in_chans=None,
            input_size=None,
            pretrained=False,
            num_classes=None,
            drop=0.0,
            drop_path=None,
            drop_block=None,
            gp=None,
            bn_momentum=None,
            bn_eps=None,
            initial_checkpoint='',
            torchscript=False,
            cls_token_per_teacher=True,
            cpe_max_size=2048,
            model_kwargs=dict(),
            teachers=[],
            register_multiple=8,
            spectral_reparam=False,
            model_norm=False,
        ),
        dtype="float32",
        cast_outputs_to_fp32=True,  # Radio default, let amp re-cast if using mixed precision
        disable_spectral_reparam=True,
        ignore_teachers=True,  # No teachers/adapters for backbone-only
        pretrained_url=None,  # Using init_cfg.checkpoint instead
        out_indices_layers=None,
        out_indices_stages=None,
    ),
)

# Checkpoint 'args' have amp=True, amp_dtype='bfloat16'
amp_dtype = "bfloat16"

# Checkpoint 'args' have sync_bn = False
sync_bn = None

# Other defaults almost always specified on command line
launcher = "pytorch"
resume = True
