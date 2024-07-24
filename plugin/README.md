# RADIO Modifications

This document outlines the additions to RADIO we've made to support more generalized fine-tuning, benchmarking and creating new RADIO models. This 'plugin' directory exists to more easily separate
our changes with the official repository so we can easily pull new changes as necessary.

## `ConfigurableRADIOModel`

- See `plugin/radio/configurable_radio_model.py`
- Adds support for explicit kwargs, which is necessary for creating new RADIO models while supporting current models and their pre-trained checkpoints

## RADIO, ADE20K

### Radio, 2x GPUs w/ BS=8, No Sync BN, No AMP

- Running: https://wandb.ai/hd-vision/radio-baselines/runs/tpqvyh2p

```bash
DATA_DIR=/home/cmccarth/data/radio/ADEChallengeData2016
WORK_DIR=/home/cmccarth/results/highres-av/ext_repos/radio/radio_linear_2xb8_80k_ade20k_512x512

SBATCH=/home/cmccarth/highres-av/tools/launch/sbatch_simple.sh
CONDA_ENV=radio
NODE=wario
NUM_GPUS=2
GPUS=v100:$NUM_GPUS
QOS=high

LAUNCH="\
python -m torch.distributed.launch --master_port=29501 --nnodes=1 --nproc_per_node=$NUM_GPUS \
/home/cmccarth/highres-av/ext_repos/radio/mmseg/train_custom.py \
/home/cmccarth/highres-av/ext_repos/radio/mmseg/configs/radio/radio_linear_8xb2-80k_ade20k-512x512.py \
--cfg-options train_dataloader.batch_size=8 \
--cfg-options train_dataloader.dataset.data_root=$DATA_DIR \
--cfg-options val_dataloader.dataset.data_root=$DATA_DIR \
--cfg-options visualizer.vis_backends.1.init_kwargs.entity='hd-vision' \
--cfg-options visualizer.vis_backends.1.init_kwargs.project='radio-baselines' \
--cfg-options visualizer.vis_backends.1.save_dir=$WORK_DIR \
--cfg-options default_hooks.checkpoint.interval=1000 \
--cfg-options train_cfg.val_interval=2000 \
--work-dir=$WORK_DIR \
--launcher=pytorch \
--resume \
--train"

mkdir -p $WORK_DIR/slurm
COMMAND="sbatch -p $NODE --gpus=$GPUS -o $WORK_DIR/slurm/slurm-%j.txt --qos=$QOS $SBATCH $CONDA_ENV $LAUNCH"

# echo "------"
# echo $COMMAND
$COMMAND
```

### Radio, 2x GPUs w/ BS=8, No Sync BN, AMP

- https://wandb.ai/hd-vision/radio-baselines/runs/x3n27i2k

```bash
DATA_DIR=/home/cmccarth/data/radio/ADEChallengeData2016
WORK_DIR=/home/cmccarth/results/highres-av/ext_repos/radio/radio_linear_2xb8_80k_ade20k_512x512_amp

SBATCH=/home/cmccarth/highres-av/tools/launch/sbatch_simple.sh
CONDA_ENV=radio
NODE=wario
NUM_GPUS=2
GPUS=v100:$NUM_GPUS
QOS=high

LAUNCH="\
python -m torch.distributed.launch --master_port=29502 --nnodes=1 --nproc_per_node=$NUM_GPUS \
/home/cmccarth/highres-av/ext_repos/radio/mmseg/train_custom.py \
/home/cmccarth/highres-av/ext_repos/radio/mmseg/configs/radio/radio_linear_8xb2-80k_ade20k-512x512.py \
--cfg-options train_dataloader.batch_size=8 \
--cfg-options train_dataloader.dataset.data_root=$DATA_DIR \
--cfg-options val_dataloader.dataset.data_root=$DATA_DIR \
--cfg-options visualizer.vis_backends.1.init_kwargs.entity='hd-vision' \
--cfg-options visualizer.vis_backends.1.init_kwargs.project='radio-baselines' \
--cfg-options visualizer.vis_backends.1.save_dir=$WORK_DIR \
--cfg-options default_hooks.checkpoint.interval=1000 \
--cfg-options train_cfg.val_interval=2000 \
--work-dir=$WORK_DIR \
--launcher=pytorch \
--resume \
--amp \
--train"

mkdir -p $WORK_DIR/slurm
COMMAND="sbatch -p $NODE --gpus=$GPUS -o $WORK_DIR/slurm/slurm-%j.txt --qos=$QOS $SBATCH $CONDA_ENV $LAUNCH"

# echo "------"
# echo $COMMAND
$COMMAND
```

### Radio, 2x GPUs w/ BS=8, Sync BN, No AMP

- Running: https://wandb.ai/hd-vision/radio-baselines/runs/2z84uepp

```bash
DATA_DIR=/home/cmccarth/data/radio/ADEChallengeData2016
WORK_DIR=/home/cmccarth/results/highres-av/ext_repos/radio/radio_linear_2xb8_80k_ade20k_512x512_syncbn

SBATCH=/home/cmccarth/highres-av/tools/launch/sbatch_simple.sh
CONDA_ENV=radio
NODE=daisy
NUM_GPUS=2
GPUS=$NUM_GPUS
QOS=medium

LAUNCH="\
python -m torch.distributed.launch --master_port=29503 --nnodes=1 --nproc_per_node=$NUM_GPUS \
/home/cmccarth/highres-av/ext_repos/radio/mmseg/train_custom.py \
/home/cmccarth/highres-av/ext_repos/radio/mmseg/configs/radio/radio_linear_8xb2-80k_ade20k-512x512.py \
--cfg-options train_dataloader.batch_size=8 \
--cfg-options train_dataloader.dataset.data_root=$DATA_DIR \
--cfg-options val_dataloader.dataset.data_root=$DATA_DIR \
--cfg-options visualizer.vis_backends.1.init_kwargs.entity='hd-vision' \
--cfg-options visualizer.vis_backends.1.init_kwargs.project='radio-baselines' \
--cfg-options visualizer.vis_backends.1.save_dir=$WORK_DIR \
--cfg-options default_hooks.checkpoint.interval=1000 \
--cfg-options train_cfg.val_interval=2000 \
--cfg-options sync_bn=torch \
--work-dir=$WORK_DIR \
--launcher=pytorch \
--resume \
--train"

mkdir -p $WORK_DIR/slurm
COMMAND="sbatch -p $NODE --gpus=$GPUS -o $WORK_DIR/slurm/slurm-%j.txt --qos=$QOS $SBATCH $CONDA_ENV $LAUNCH"

# echo "------"
# echo $COMMAND
$COMMAND
```

### Radio, 2x GPUs w/ BS=8, Sync BN, AMP

- Running: https://wandb.ai/hd-vision/radio-baselines/runs/t4laja8i

```bash
DATA_DIR=/home/cmccarth/data/radio/ADEChallengeData2016
WORK_DIR=/home/cmccarth/results/highres-av/ext_repos/radio/radio_linear_2xb8_80k_ade20k_512x512_syncbn_amp

SBATCH=/home/cmccarth/highres-av/tools/launch/sbatch_simple.sh
CONDA_ENV=radio
NODE=daisy
NUM_GPUS=2
GPUS=$NUM_GPUS
QOS=medium

LAUNCH="\
python -m torch.distributed.launch --master_port=29504 --nnodes=1 --nproc_per_node=$NUM_GPUS \
/home/cmccarth/highres-av/ext_repos/radio/mmseg/train_custom.py \
/home/cmccarth/highres-av/ext_repos/radio/mmseg/configs/radio/radio_linear_8xb2-80k_ade20k-512x512.py \
--cfg-options train_dataloader.batch_size=8 \
--cfg-options train_dataloader.dataset.data_root=$DATA_DIR \
--cfg-options val_dataloader.dataset.data_root=$DATA_DIR \
--cfg-options visualizer.vis_backends.1.init_kwargs.entity='hd-vision' \
--cfg-options visualizer.vis_backends.1.init_kwargs.project='radio-baselines' \
--cfg-options visualizer.vis_backends.1.save_dir=$WORK_DIR \
--cfg-options default_hooks.checkpoint.interval=1000 \
--cfg-options train_cfg.val_interval=2000 \
--cfg-options sync_bn=torch \
--work-dir=$WORK_DIR \
--launcher=pytorch \
--resume \
--amp \
--train"

mkdir -p $WORK_DIR/slurm
COMMAND="sbatch -p $NODE --gpus=$GPUS -o $WORK_DIR/slurm/slurm-%j.txt --qos=$QOS $SBATCH $CONDA_ENV $LAUNCH"

# echo "------"
# echo $COMMAND
$COMMAND
```

## E-RAIO, ADE20K

### E-Radio, 2x GPUs w/ BS=8, No Sync BN, No AMP

```bash
DATA_DIR=/home/cmccarth/data/radio/ADEChallengeData2016
WORK_DIR=/home/cmccarth/results/highres-av/ext_repos/radio/eradio_linear_2xb8_80k_ade20k_512x512

SBATCH=/home/cmccarth/highres-av/tools/launch/sbatch_simple.sh
CONDA_ENV=radio
NODE=bowser
NUM_GPUS=2
GPUS=$NUM_GPUS
QOS=high

LAUNCH="\
python -m torch.distributed.launch  --master_port=29505 --nnodes=1 --nproc_per_node=$NUM_GPUS \
/home/cmccarth/highres-av/ext_repos/radio/mmseg/train_custom.py \
/home/cmccarth/highres-av/ext_repos/radio/mmseg/configs/radio/eradio_linear_8xb2-80k_ade20k-512x512.py \
--cfg-options train_dataloader.batch_size=8 \
--cfg-options train_dataloader.dataset.data_root=$DATA_DIR \
--cfg-options val_dataloader.dataset.data_root=$DATA_DIR \
--cfg-options visualizer.vis_backends.1.init_kwargs.entity='hd-vision' \
--cfg-options visualizer.vis_backends.1.init_kwargs.project='radio-baselines' \
--cfg-options visualizer.vis_backends.1.save_dir=$WORK_DIR \
--cfg-options default_hooks.checkpoint.interval=1000 \
--cfg-options train_cfg.val_interval=2000 \
--work-dir=$WORK_DIR \
--launcher=pytorch \
--resume \
--train"

mkdir -p $WORK_DIR/slurm
COMMAND="sbatch -p $NODE --gpus=$GPUS -o $WORK_DIR/slurm/slurm-%j.txt --qos=$QOS $SBATCH $CONDA_ENV $LAUNCH"

# echo "------"
# echo $COMMAND
$COMMAND
```

### E-Radio, 2x GPUs w/ BS=8, No Sync BN, AMP

```bash
DATA_DIR=/home/cmccarth/data/radio/ADEChallengeData2016
WORK_DIR=/home/cmccarth/results/highres-av/ext_repos/radio/eradio_linear_2xb8_80k_ade20k_512x512_amp

SBATCH=/home/cmccarth/highres-av/tools/launch/sbatch_simple.sh
CONDA_ENV=radio
NODE=bowser
NUM_GPUS=2
GPUS=$NUM_GPUS
QOS=high

LAUNCH="\
python -m torch.distributed.launch  --master_port=29506 --nnodes=1 --nproc_per_node=$NUM_GPUS \
/home/cmccarth/highres-av/ext_repos/radio/mmseg/train_custom.py \
/home/cmccarth/highres-av/ext_repos/radio/mmseg/configs/radio/eradio_linear_8xb2-80k_ade20k-512x512.py \
--cfg-options train_dataloader.batch_size=8 \
--cfg-options train_dataloader.dataset.data_root=$DATA_DIR \
--cfg-options val_dataloader.dataset.data_root=$DATA_DIR \
--cfg-options visualizer.vis_backends.1.init_kwargs.entity='hd-vision' \
--cfg-options visualizer.vis_backends.1.init_kwargs.project='radio-baselines' \
--cfg-options visualizer.vis_backends.1.save_dir=$WORK_DIR \
--cfg-options default_hooks.checkpoint.interval=1000 \
--cfg-options train_cfg.val_interval=2000 \
--work-dir=$WORK_DIR \
--launcher=pytorch \
--resume \
--amp \
--train"

mkdir -p $WORK_DIR/slurm
COMMAND="sbatch -p $NODE --gpus=$GPUS -o $WORK_DIR/slurm/slurm-%j.txt --qos=$QOS $SBATCH $CONDA_ENV $LAUNCH"

# echo "------"
# echo $COMMAND
$COMMAND
```

### E-Radio, 2x GPUs w/ BS=8, Sync BN, No AMP

- Running: https://wandb.ai/hd-vision/radio-baselines/runs/i3cqkcwm

```bash
DATA_DIR=/home/cmccarth/data/radio/ADEChallengeData2016
WORK_DIR=/home/cmccarth/results/highres-av/ext_repos/radio/eradio_linear_2xb8_80k_ade20k_512x512_syncbn

SBATCH=/home/cmccarth/highres-av/tools/launch/sbatch_simple.sh
CONDA_ENV=radio
NODE=toad
NUM_GPUS=2
GPUS=$NUM_GPUS
QOS=medium

LAUNCH="\
python -m torch.distributed.launch --master_port=29507 --nnodes=1 --nproc_per_node=$NUM_GPUS \
/home/cmccarth/highres-av/ext_repos/radio/mmseg/train_custom.py \
/home/cmccarth/highres-av/ext_repos/radio/mmseg/configs/radio/eradio_linear_8xb2-80k_ade20k-512x512.py \
--cfg-options train_dataloader.batch_size=8 \
--cfg-options train_dataloader.dataset.data_root=$DATA_DIR \
--cfg-options val_dataloader.dataset.data_root=$DATA_DIR \
--cfg-options visualizer.vis_backends.1.init_kwargs.entity='hd-vision' \
--cfg-options visualizer.vis_backends.1.init_kwargs.project='radio-baselines' \
--cfg-options visualizer.vis_backends.1.save_dir=$WORK_DIR \
--cfg-options default_hooks.checkpoint.interval=1000 \
--cfg-options train_cfg.val_interval=2000 \
--cfg-options sync_bn=torch \
--work-dir=$WORK_DIR \
--launcher=pytorch \
--resume \
--train"

mkdir -p $WORK_DIR/slurm
COMMAND="sbatch -p $NODE --gpus=$GPUS -o $WORK_DIR/slurm/slurm-%j.txt --qos=$QOS $SBATCH $CONDA_ENV $LAUNCH"

# echo "------"
# echo $COMMAND
$COMMAND
```

### E-Radio, 2x GPUs w/ BS=8, Sync BN, AMP

```bash
DATA_DIR=/home/cmccarth/data/radio/ADEChallengeData2016
WORK_DIR=/home/cmccarth/results/highres-av/ext_repos/radio/eradio_linear_2xb8_80k_ade20k_512x512_syncbn_amp

SBATCH=/home/cmccarth/highres-av/tools/launch/sbatch_simple.sh
CONDA_ENV=radio
NODE=mario
NUM_GPUS=2
GPUS=v100:$NUM_GPUS
QOS=medium

LAUNCH="\
python -m torch.distributed.launch --master_port=29508 --nnodes=1 --nproc_per_node=$NUM_GPUS \
/home/cmccarth/highres-av/ext_repos/radio/mmseg/train_custom.py \
/home/cmccarth/highres-av/ext_repos/radio/mmseg/configs/radio/eradio_linear_8xb2-80k_ade20k-512x512.py \
--cfg-options train_dataloader.batch_size=8 \
--cfg-options train_dataloader.dataset.data_root=$DATA_DIR \
--cfg-options val_dataloader.dataset.data_root=$DATA_DIR \
--cfg-options visualizer.vis_backends.1.init_kwargs.entity='hd-vision' \
--cfg-options visualizer.vis_backends.1.init_kwargs.project='radio-baselines' \
--cfg-options visualizer.vis_backends.1.save_dir=$WORK_DIR \
--cfg-options default_hooks.checkpoint.interval=1000 \
--cfg-options train_cfg.val_interval=2000 \
--cfg-options sync_bn=torch \
--work-dir=$WORK_DIR \
--launcher=pytorch \
--resume \
--amp \
--train"

mkdir -p $WORK_DIR/slurm
COMMAND="sbatch -p $NODE --gpus=$GPUS -o $WORK_DIR/slurm/slurm-%j.txt --qos=$QOS $SBATCH $CONDA_ENV $LAUNCH"

# echo "------"
# echo $COMMAND
$COMMAND
```

