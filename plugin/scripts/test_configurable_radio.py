"""
Example runs:

```
conda activate radio
cd radio/scripts

# Verify and output kwargs used into output/test_configurable_radio.txt
python test_configurable_radio.py

# NOTE: This requires a GPU for full verification (forward pass), otherwise add --diff-skip-forward
```
"""

import copy
import sys
import argparse
import pprint
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import timm
import torch
from torch.hub import load_state_dict_from_url
from transformers import AutoModel

# Hack: add top-level module to path until setup.py exists or PYTHONPATH has been updated
# sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # Top-level `RADIO` project dir

# NOTE: Expect PYTHONPATH to include RADIO project dir for these imports
from hubconf import radio_model
from radio.common import RESOURCE_MAP
from radio.radio_model import RADIOModel, Resolution

from plugin.scripts.common.diff_modules import diff_model
from plugin.radio.configurable_radio_model import (
    ConfigurableRADIOModel,
)

logger = logging.getLogger(Path(__file__).name)
default_log_dir = str(Path(__file__).parent.joinpath("output"))
default_log_filename = Path(__file__).name.replace(".py", ".txt")

HF_RESOURCE_MAP = {  # Version key in RESOURCE_MAP to HuggingFace repo id
    # RADIOv2.5
    "radio_v2.5-b": "NVIDIA/RADIO-B",
    "radio_v2.5-l": "NVIDIA/RADIO-L",
    # RADIO
    "radio_v2.1": "NVIDIA/RADIO",
    "radio_v2": None,
    "radio_v1": None,
    # E-RADIO
    "e-radio_v2": "NVIDIA/E-RADIO",
}


def _finalize_log_filename(log_filename: str, version: str) -> str:
    if "radio" in log_filename:
        log_filename = log_filename.replace("radio", version)  # e.g. 'radio' -> 'radio_v2.5'
    else:
        suffix = Path(log_filename).suffix
        log_filename = log_filename.replace(suffix, f"_{version}{suffix}")

    return log_filename


def setup_logger(
    log_dir: str,
    log_filename: str,
):
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s %(name)s %(levelname)s] %(message)s", datefmt="%m/%d %H:%M:%S"
    )

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    log_filename = _finalize_log_filename(log_filename=log_filename, version="all")
    log_filepath = Path(log_dir, log_filename)
    file_handler = logging.FileHandler(filename=log_filepath, mode="w")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)


def get_version_specific_handler(
    log_dir: str,
    log_filename: str,
    version: str,
):
    log_filename = _finalize_log_filename(log_filename=log_filename, version=version)
    log_filepath = Path(log_dir, log_filename)

    # Skip any formatting prefix so diffs between versions are easy
    formatter = logging.Formatter("%(message)s")

    file_handler = logging.FileHandler(filename=log_filepath, mode="w")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    return file_handler


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
        "--skip-full-args",
        "--skip_full_args",
        default=False,
        action="store_true",
        help="Skip printing full args in checkpoint. Default: False.",
    )

    parser.add_argument(
        "--log-dir",
        "--log_dir",
        default=default_log_dir,
        type=str,
        help=f"Directory for output log. Default: {default_log_dir}",
    )

    parser.add_argument(
        "--log-filename",
        "--log_filename",
        default=default_log_filename,
        type=str,
        help=f"Filename for output log. Default: {default_log_filename}",
    )
    parser.add_argument(
        "--diff-skip-forward",
        "--diff_skip_forward",
        action="store_true",
        default=False,
        help="Skip forward pass when diff'ing original and new radio modules. Default: False",
    )
    args = parser.parse_args()

    if (
        not (torch.cuda.is_available() and torch.cuda.device_count() > 0)
        and not args.diff_skip_forward
    ):
        raise RuntimeError(
            f"Found 0 CUDA devices, cannot perform forward pass check in diff_model()."
            f" Set --diff-skip-forward to skip checking the model via a forward_pass."
        )

    return args


def _log_error_messages(error_msgs: list[str], all: bool) -> None:
    logger.info("*" * 60)
    success_str = "ALL SUCCESS" if all else "SUCCESS"
    verification_str = success_str if len(error_msgs) == 0 else "FAILED"
    logger.info(f"Verification: {verification_str}")

    if len(error_msgs) > 0:
        for idx, error_msg in enumerate(error_msgs):
            logger.info("* " * 30)
            logger.info(f"Error message {idx + 1} of {len(error_msgs)}")
            logger.info("* " * 30)
            logger.error(error_msg)

    logger.info("*" * 60)


def main(adaptor_cfgs: Optional[dict] = None):
    cmdline_args = parse_args()
    setup_logger(log_dir=cmdline_args.log_dir, log_filename=cmdline_args.log_filename)

    all_error_messages: list[str] = []
    for version in cmdline_args.versions:
        version_error_messages: list[str] = []

        # Add a second file handler to output these results to a version specific file for diffs
        version_file_handler = get_version_specific_handler(
            log_dir=cmdline_args.log_dir,
            log_filename=cmdline_args.log_filename,
            version=version,
        )
        logger.addHandler(version_file_handler)

        assert (
            version in RESOURCE_MAP
        ), f"Version '{version}' not found in resource map. Available versions: {RESOURCE_MAP.keys()}"
        resource = RESOURCE_MAP[version]

        # Convert resource RadioResource (dataclass) to dict, then Resolution (NamedTuple) to dict
        # For RADIOv2.5 preferred_resolution is a normal tuple, so handle that differently
        preferred_resolution = resource.preferred_resolution
        if not isinstance(resource.preferred_resolution, Resolution):
            assert isinstance(preferred_resolution, tuple) and len(preferred_resolution) == 2, (
                f"Expected RadioResource.preferred_resolution={preferred_resolution} to have type"
                f" Resolution or 2-tuple, found type={type(preferred_resolution).__name__}"
            )
            resource.preferred_resolution = Resolution(*resource.preferred_resolution)

        resource_dict = asdict(resource)
        resource_dict["preferred_resolution"] = resource.preferred_resolution._asdict()

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
            version_error_messages.append(msg)
            # logger.error(msg)
            # raise RuntimeError(msg)

        # Generate another version and verify diff with this is also empty
        # This could be non-empty if the deepcopy in diff_model() isn't working correctly
        orig_model_v2 = radio_model(
            version=version,
            progress=True,
            adaptor_names=None,
            vitdet_window_size=getattr(radio_args, "vitdet_window_size", None),
        )
        diff_results = diff_model(
            curr_model=orig_model_v2,
            orig_model=orig_model,
            resolution=resource.preferred_resolution,
            skip_forward_pass=cmdline_args.diff_skip_forward,
        )
        if len(diff_results) > 0:
            msg = (
                f"Diff for RADIOModel with version '{version}' failed. Two consecutive invocations"
                f" of the same model (after one diff) produced different results. Diff results:"
                f"\n{pprint.pformat(diff_results, sort_dicts=False)}"
            )
            version_error_messages.append(msg)
            # logger.error(msg)
            # raise RuntimeError(msg)

        # Verify diff with HF model is empty if HF repo exists
        # NOTE: This will overwrite the timm registry in the process of loading the HF model,
        #       which will change the entrypoint for create_model() inside the RADIOModel class
        #       so we will backup the registery and then reload it
        prev_timm_entrypoints = copy.deepcopy(timm.models._registry._model_entrypoints)

        assert (
            version in HF_RESOURCE_MAP
        ), f"Missing version {version} in HF_RESOURCE_MAP to look up hugging face repo id"
        hf_repo = HF_RESOURCE_MAP[version]
        if hf_repo is not None:
            # HF model is wrapper around the normal RADIOModel; real model is hf_model.radio_model
            hf_model = AutoModel.from_pretrained(hf_repo, trust_remote_code=True)
            hf_model = hf_model.radio_model

            # The type of hf_model components are things like
            #   <class 'transformers_modules.NVIDIA.RADIO-B.39bbff9882746141c2ab6adf679321cf9d8ac4a5.radio_model.RADIOModel'>
            # which will not match standard / original types like
            #   <class 'radio.radio_model.RADIOModel'>
            # so use string-based type checking, e.g. both have type(model).__name__ == 'RADIOModel'

            # Also for RADIO v2.1 the HF model has buffer 'summary_idxs': tensor([0, 2]) while the
            #   torchhub model has 'summary_idxs': tensor([0, 1, 2, 3]). Since HF models don't
            #   support adaptors we will ignore this?
            diff_results = diff_model(
                curr_model=hf_model,
                orig_model=orig_model,
                resolution=resource.preferred_resolution,
                skip_forward_pass=cmdline_args.diff_skip_forward,
                type_checking_use_name=True,
            )
            if len(diff_results) > 0:
                msg = (
                    f"Diff for RADIOModel with version '{version}' failed. Torchhub version and"
                    f" HuggingFace version using repo id '{hf_repo}' produced different results."
                    f" Diff results:"
                    f"\n{pprint.pformat(diff_results, sort_dicts=False)}"
                )
                version_error_messages.append(msg)
                # logger.error(msg)
                # raise RuntimeError(msg)

        timm.models._registry._model_entrypoints = prev_timm_entrypoints

        # Generate the model independent from the checkpoint, and log these args for mmdet configs
        # Then diff with original model to verify the architecture hasn't changed
        model = ConfigurableRADIOModel(
            # Kwargs for RADIOModel
            patch_size=resource.patch_size,
            max_resolution=resource.max_resolution,
            preferred_resolution=resource.preferred_resolution,
            adaptor_cfgs=adaptor_cfgs,
            vitdet_window_size=getattr(radio_args, "vitdet_window_size", None),  # Not in every ckpt
            vitdet_num_windowed=resource.vitdet_num_windowed,
            vitdet_num_global=resource.vitdet_num_global,
            # Kwargs for create_model()
            create_model_kwargs=dict(
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
                spectral_reparam=getattr(radio_args, "spectral_reparam", False),  # Not in every ckp
                model_norm=getattr(radio_args, "model_norm", False),  # Not in every ckpt
            ),
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
            expected_curr_type_diffs={"": ConfigurableRADIOModel},  # Top-level, empty prefix
        )
        if len(diff_results) > 0:
            msg = (
                f"Diff for RADIOModel with version '{version}' failed, does not match original"
                f" RADIOModel. Implementation of radio_model() has changed or a bug has been"
                f" introduced. Diff results:"
                f"\n{pprint.pformat(diff_results, sort_dicts=False)}"
            )
            version_error_messages.append(msg)
            # logger.error(msg)
            # raise RuntimeError(msg)

        logger.info("-" * 60)
        logger.info(
            f"Radio Version: {version}, Model: {radio_args.model},"
            f" Num Params: {get_num_params_str(model)}"
        )

        logger.info("- " * 30)
        logger.info("Resource kwargs")
        logger.info("- " * 30)
        logger.info(pprint.pformat(resource_dict, sort_dicts=False))

        logger.info("- " * 30)
        logger.info("Conditioner kwargs")
        logger.info("- " * 30)
        logger.info(pprint.pformat(model.conditioner_kwargs, sort_dicts=False))

        logger.info("- " * 30)
        logger.info("Create model kwargs")
        logger.info("- " * 30)
        logger.info(pprint.pformat(model.create_model_kwargs, sort_dicts=False))

        logger.info("- " * 30)
        logger.info("Radio kwargs")
        logger.info("- " * 30)
        logger.info(pprint.pformat(model.radio_kwargs, sort_dicts=False))

        # The important one, the kwargs for our create_radio_model_from_kwargs() method
        logger.info("- " * 30)
        logger.info("Configurable radio kwargs")
        logger.info("- " * 30)
        logger.info(pprint.pformat(model.all_kwargs, sort_dicts=False))

        if not cmdline_args.skip_full_args:
            logger.info("- " * 30)
            logger.info("Full checkpoint kwargs")
            logger.info("- " * 30)
            logger.info(pprint.pformat(radio_args_dict, sort_dicts=True))

        _log_error_messages(error_msgs=version_error_messages, all=False)
        logger.removeHandler(version_file_handler)

        all_error_messages.extend(version_error_messages)

    # Log all error messages at the end once more
    if len(cmdline_args.versions) > 1:
        _log_error_messages(error_msgs=all_error_messages, all=True)


if __name__ == "__main__":
    main()
