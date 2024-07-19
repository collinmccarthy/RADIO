from mmengine.config import read_base

with read_base():
    from .eradio_linear_8xb2_80k_ade20k_512x512 import *

train_dataloader.update(batch_size=4)
