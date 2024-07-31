# fmt: off
import os

_base_ = ["./radio_linear_8xb2-80k_ade20k-512x512.py"]

model = dict(
    # Uses `timm.models.vision_transformer.py, vit_large_patch16_224 with embed_dim=1024
    # See https://github.com/huggingface/pytorch-image-models/blob/8b14fc7bb6d42d67da6c33ac9bfaf3c024dbaff8/timm/models/vision_transformer.py#L2148
    backbone=dict(
        repo_id="nvidia/RADIO-L",
    ),
    decode_head=dict(
        in_channels=[1024],
        channels=1024,
    )
)
