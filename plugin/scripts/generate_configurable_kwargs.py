"""
Example runs:

```
conda activate radio
cd radio/scripts

# Create output/generate_configurable_kwargs.txt without full args
python generate_configurable_kwargs.py

# Create output/generate_version_kwargs_with_args.txt with full args
python generate_configurable_kwargs.py --full-args --log-filename=generate_version_kwargs_with_args.txt
```
"""

import sys
import argparse
import pprint
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch
from torch.hub import load_state_dict_from_url

# Hack: add top-level module to path until setup.py exists or PYTHON_PATH has been updated
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # Top-level `RADIO` project dir

from hubconf import radio_model
from radio.common import RESOURCE_MAP
from radio.radio_model import RADIOModel

from plugin.scripts.common.diff_modules import diff_model
from plugin.radio.configurable_radio_model import (
    ConfigurableRADIOModel,
)

logger = logging.getLogger(Path(__file__).name)


def setup_logger(log_dir: str, log_filename: str, append_log: bool = False):
    log_filepath = Path(log_dir, log_filename)

    # Reset handlers
    formatter = logging.Formatter(
        "[%(asctime)s %(name)s %(levelname)s] %(message)s", datefmt="%m/%d %H:%M:%S"
    )

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(filename=log_filepath, mode="a" if append_log else "w")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    logger.setLevel(logging.INFO)


def get_num_params_str(model: RADIOModel) -> str:
    num_params = sum([p.numel() for p in model.parameters()])
    num_params_m = num_params / 1e6
    return f"{num_params_m:.2f}M"


def parse_args():
    parser = argparse.ArgumentParser(description="Print all model configurations")
    parser.add_argument(
        "--versions",
        type=str,
        nargs="+",
        default=list(RESOURCE_MAP.keys()),
        help="Model versions to print out. Default: all versions in RESOURCE_MAP",
    )
    parser.add_argument(
        "--full-args",
        "--full_args",
        default=False,
        action="store_true",
        help="Print full args in checkpoint. Default: False.",
    )

    default_log_dir = str(Path(__file__).parent.joinpath("output"))
    parser.add_argument(
        "--log-dir",
        "--log_dir",
        default=default_log_dir,
        type=str,
        help=f"Directory for output log. Default: {default_log_dir}",
    )

    default_log_filename = Path(__file__).name.replace(".py", ".txt")
    parser.add_argument(
        "--log-filename",
        "--log_filename",
        default=default_log_filename,
        type=str,
        help=f"Filename for output log. Default: {default_log_filename}",
    )
    parser.add_argument(
        "--log-append",
        "--log_append",
        action="store_true",
        default=False,
        help="Append to output log instead of overwriting. Default: False",
    )
    parser.add_argument(
        "--diff-skip-forward",
        "--diff_skip_forward",
        action="store_true",
        default=False,
        help="Skip forward pass when diff'ing original and new radio modules. Default: False",
    )
    args = parser.parse_args()
    return args


def main(adaptor_cfgs: Optional[dict] = None):
    cmdline_args = parse_args()
    setup_logger(
        log_dir=cmdline_args.log_dir,
        log_filename=cmdline_args.log_filename,
        append_log=cmdline_args.log_append,
    )

    for version in cmdline_args.versions:
        assert (
            version in RESOURCE_MAP
        ), f"Version '{version}' not found in resource map. Available versions: {RESOURCE_MAP.keys()}"

        resource = RESOURCE_MAP[version]

        # Convert resource RadioResource (dataclass) to dict, then Resolution (NamedTuple) to dict
        resource_dict = asdict(resource)
        resource_dict["preferred_resolution"] = resource_dict["preferred_resolution"]._asdict()

        # Download model if doesn't exist
        checkpoint = load_state_dict_from_url(resource.url, progress=True, map_location="cpu")
        radio_args = checkpoint["args"]
        radio_args_dict = vars(radio_args)

        # Generate the original model twice, verify the diff with itself is empty
        orig_model = radio_model(
            version=version,
            progress=True,
            adaptor_names=None,
            vitdet_window_size=getattr(radio_args, "vitdet_window_size", None),
        )

        # Verify diff with itself is empty
        diff_results = diff_model(
            curr_model=orig_model,
            orig_model=orig_model,
            resolution=resource.preferred_resolution,
            skip_forward_pass=cmdline_args.diff_skip_forward,
        )
        if len(diff_results) > 0:
            msg = (
                f"Diff for RADIOModel with version '{version}' failed. Two consecutive invocations"
                f" of the same model produced different results. Diff results:"
                f"\n{pprint.pformat(diff_results, sort_dicts=False)}"
            )
            logger.error(msg)
            raise RuntimeError(msg)

        # Generate another version and verify diff with this is also empty
        # This could be non-empty if the deepcopy in diff_model() isn't working correctly
        orig_model_v2 = radio_model(
            version=version,
            progress=True,
            adaptor_names=None,
            vitdet_window_size=getattr(radio_args, "vitdet_window_size", None),
        )
        diff_results = diff_model(
            curr_model=orig_model,
            orig_model=orig_model_v2,
            resolution=resource.preferred_resolution,
            skip_forward_pass=cmdline_args.diff_skip_forward,
        )
        if len(diff_results) > 0:
            msg = (
                f"Diff for RADIOModel with version '{version}' failed. Two consecutive invocations"
                f" of the same model (after one diff) produced different results. Diff results:"
                f"\n{pprint.pformat(diff_results, sort_dicts=False)}"
            )
            logger.error(msg)
            raise RuntimeError(msg)

        # Generate the model independent from the checkpoint, and log these args for mmdet configs
        # Then diff with original model to verify the architecture hasn't changed
        model = ConfigurableRADIOModel(
            # Kwargs for RADIOModel
            patch_size=resource.patch_size,
            max_resolution=resource.max_resolution,
            preferred_resolution=resource.preferred_resolution,
            adaptor_cfgs=adaptor_cfgs,
            vitdet_window_size=getattr(radio_args, "vitdet_window_size", None),
            # Kwargs for create_model()
            model=radio_args.model,
            in_chans=radio_args.in_chans,
            input_size=radio_args.input_size,
            pretrained=radio_args.pretrained,
            num_classes=radio_args.num_classes,
            drop=radio_args.drop,
            drop_path=radio_args.drop_path,
            drop_block=radio_args.drop_block,
            gp=radio_args.gp,
            bn_momentum=radio_args.bn_momentum,
            bn_eps=radio_args.bn_eps,
            initial_checkpoint=radio_args.initial_checkpoint,
            torchscript=radio_args.torchscript,
            cls_token_per_teacher=radio_args.cls_token_per_teacher,
            cpe_max_size=radio_args.cpe_max_size,
            model_kwargs=radio_args.model_kwargs,
            teachers=radio_args.teachers,
            register_multiple=radio_args.register_multiple,
            spectral_reparam=getattr(radio_args, "spectral_reparam", False),  # Not in every ckpt
            model_norm=getattr(radio_args, "model_norm", False),  # Not in every ckpt
            # Kwargs for conditioner
            dtype=getattr(radio_args, "dtype", torch.float32),  # Not in every ckpt
            # Other kwargs
            pretrained_url=resource.url,
        )

        diff_results = diff_model(
            curr_model=model,
            orig_model=orig_model,
            resolution=resource.preferred_resolution,
            skip_forward_pass=cmdline_args.diff_skip_forward,
            expected_type_diffs={"": ConfigurableRADIOModel},  # Top-level, empty prefix
        )
        if len(diff_results) > 0:
            msg = (
                f"Diff for RADIOModel with version '{version}' failed, does not match original"
                f" RADIOModel. Implementation of radio_model() has changed or a bug has been"
                f" introduced. Diff results:"
                f"\n{pprint.pformat(diff_results, sort_dicts=False)}"
            )
            logger.error(msg)
            raise RuntimeError(msg)

        logger.info("-" * 80)
        logger.info(
            f"Radio Version: {version}, Model: {radio_args.model},"
            f" Num Params: {get_num_params_str(model)}, Verification: SUCCESS"
        )

        logger.info("- " * 40)
        logger.info("Resource kwargs")
        logger.info("- " * 40)
        logger.info("\n" + pprint.pformat(resource_dict, sort_dicts=False))

        logger.info("- " * 40)
        logger.info("Conditioner kwargs")
        logger.info("- " * 40)
        logger.info("\n" + pprint.pformat(model.conditioner_kwargs, sort_dicts=False))

        logger.info("- " * 40)
        logger.info("Create model kwargs")
        logger.info("- " * 40)
        logger.info("\n" + pprint.pformat(model.model_kwargs, sort_dicts=False))

        logger.info("- " * 40)
        logger.info("Radio kwargs")
        logger.info("- " * 40)
        logger.info("\n" + pprint.pformat(model.radio_kwargs, sort_dicts=False))

        # The important one, the kwargs for our create_radio_model_from_kwargs() method
        logger.info("- " * 40)
        logger.info("Configurable radio kwargs")
        logger.info("- " * 40)
        logger.info("\n" + pprint.pformat(model.all_kwargs, sort_dicts=False))

        if cmdline_args.full_args:
            logger.info("- " * 40)
            logger.info("Full checkpoint kwargs")
            logger.info("- " * 40)
            logger.info("\n" + pprint.pformat(radio_args_dict, sort_dicts=True))

        logger.info("-" * 80 + "\n")


if __name__ == "__main__":
    main()
