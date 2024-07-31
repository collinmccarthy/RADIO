# fmt: off
import os

_base_ = ["./radio_linear_8xb2-80k_ade20k-512x512.py"]

model = dict(
    # Uses timm.models.vision_transformer.py, vit_base_patch16_224 with embed_dim=768
    # See https://github.com/huggingface/pytorch-image-models/blob/8b14fc7bb6d42d67da6c33ac9bfaf3c024dbaff8/timm/models/vision_transformer.py#L2099
    backbone=dict(
        repo_id="nvidia/RADIO-B",
    ),
    decode_head=dict(
        in_channels=[768],
        channels=768,
    )
)

optim_wrapper = dict(  # Change LR from 1e-3 to 1e-4 and add clip_grad
    optimizer=dict(lr=0.0001),
    clip_grad=dict(
        max_norm=0.01,
        norm_type=2,
    ),
)

param_scheduler = [  # Change PolyLR eta_min from 0 to 1e-6, and end from 80k to 160k
    dict(type="LinearLR", start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type="PolyLR",
        eta_min=1e-6,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    ),
]
