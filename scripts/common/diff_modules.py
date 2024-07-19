import sys
import copy
from pathlib import Path
from typing import Callable, Union

import torch
from torch import nn

# Hack for path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # Top-level `radio` folder

from radio.radio_model import RADIOModel, RadioOutput, Resolution


def remove_empty_subdicts(results: dict) -> None:
    for key in list(results.keys()):
        val = results[key]
        if isinstance(val, dict):
            remove_empty_subdicts(val)
            if len(val) == 0:
                results.pop(key)  # Remove


def diff_model(
    curr_model: Union[RADIOModel, nn.Module],
    orig_model: Union[RADIOModel, nn.Module],
    resolution: Resolution,
    skip_forward_pass: bool = False,
) -> dict:
    diff_results = {}  # Pass in dict to add to, so the results are "flattened"
    _diff_module(curr_module=curr_model, orig_module=orig_model, diff_results=diff_results)

    # Perform forward pass and diff result if everything else lines up
    if not skip_forward_pass:
        if not (torch.cuda.is_available() and torch.cuda.device_count() > 0):
            raise RuntimeError(
                f"Found 0 CUDA devices, cannot perform forward pass check in diff_model()."
                f" Set skip_forward_pass=True to skip this check."
            )

        # Create a buffer with random values between 0-1
        buff = torch.rand(
            1, 3, resolution.height, resolution.width, dtype=torch.float32, device="cuda"
        )

        def _get_deterministic_results(
            model: RADIOModel, training: bool, deploy: bool
        ) -> dict[str, torch.Tensor]:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.cuda.empty_cache()

            # Deepcopy model since switching between train/eval and deploy will change some params
            curr_model = copy.deepcopy(model)

            # Move model to GPU, run forward pass, move results to CPU
            curr_model.cuda()

            if training:
                curr_model.train()
            else:
                curr_model.eval()

            if deploy:
                curr_model.switch_to_deploy()

            with torch.no_grad():  # Even if "training" mode, not performing grad updates
                output: Union[RadioOutput, torch.Tensor] = curr_model(buff)

            if isinstance(output, RadioOutput):
                output_dict = output._asdict()
            elif isinstance(output, torch.Tensor):
                output_dict = {"features": output}
            else:
                raise RuntimeError(
                    f"Expected model to output RadioOutput or torch.tensor, found output type"
                    f" {type(output)} with value: {output}"
                )

            for key in list(output_dict.keys()):
                output_dict[key] = output_dict[key].cpu()

            # Delete the model and clear GPU cache
            del curr_model
            torch.cuda.empty_cache()

            return output_dict

        # For some models we get different results with training for the same exact model, even
        #   with deepcopy, not sure why. Skipping training=True and just checking eval mode for now.
        training_flags = [False]
        for training in training_flags:
            deploy_flags = [True, False] if not training else [False]
            for deploy in deploy_flags:
                orig_results_dict = _get_deterministic_results(
                    model=orig_model, training=training, deploy=deploy
                )
                curr_results_dict = _get_deterministic_results(
                    model=curr_model, training=training, deploy=deploy
                )

                # Verify shape
                assert sorted(list(curr_results_dict.keys())) == sorted(
                    list(orig_results_dict.keys())
                ), "Expected both models to have same keys in output dict"

                output_diff = dict()
                for key in orig_results_dict.keys():
                    orig_result = orig_results_dict[key]
                    curr_result = curr_results_dict[key]

                    output_diff[key] = {}
                    if curr_result.shape != orig_result.shape:
                        output_diff[key]["shape"] = dict(
                            orig_model=tuple(orig_result.shape),
                            curr_model=tuple(curr_result.shape),
                        )
                    else:  # Verify values
                        # If training, can't use exact for some models, not exactly sure why
                        output_diff[key]["values"] = _diff_tensor(
                            curr_tensor=curr_result, orig_tensor=orig_result, exact=not training
                        )

                diff_results_key = "forward"
                if training:
                    diff_results_key += "_train"
                else:
                    diff_results_key += "_eval"

                if deploy:
                    diff_results_key += "_deploy"

                diff_results[diff_results_key] = output_diff

    remove_empty_subdicts(diff_results)
    return diff_results


def _diff_tensor(curr_tensor: torch.Tensor, orig_tensor: torch.Tensor, exact: bool = True) -> dict:
    diff_result = {}

    if exact:  # Try exact comparsion, requires deterministic functions for forward pass outputs
        tensor_equal = curr_tensor == orig_tensor
    else:
        tensor_equal = torch.isclose(orig_tensor, curr_tensor)

    tensor_not_equal = ~tensor_equal
    if not torch.all(tensor_equal):
        numel_true = torch.sum(tensor_equal)
        numel_false = torch.sum(tensor_not_equal)
        diff_result.update(
            numel_compare_true=numel_true,
            numel_compare_true_perc=100 * (numel_true / (numel_true + numel_false)),
            numel_compare_false=torch.sum(tensor_not_equal),
            numel_compare_false_perc=100 * (numel_false / (numel_true + numel_false)),
            curr_mean=torch.mean(curr_tensor),
            orig_mean=torch.mean(orig_tensor),
        )

    return diff_result


def _diff_module_hooks(curr_hooks: dict[int, Callable], orig_hooks: dict[int, Callable]) -> dict:
    diff_results = {}

    if len(curr_hooks) == 0 and len(orig_hooks) == 0:
        return diff_results
    elif len(curr_hooks) != len(orig_hooks):
        diff_results["mismatched_num_hooks"] = dict(
            orig_num_hooks=len(orig_hooks),
            orig_hook_types=[type(hook) for hook in orig_hooks.values()],
            curr_num_hooks=len(curr_hooks),
            curr_hook_types=[type(hook) for hook in curr_hooks.values()],
        )
    else:
        assert sorted(list(curr_hooks.keys())) == sorted(
            list(orig_hooks.keys())
        ), "Expected hook ids to be the same if number of hooks is same"

        mismatched_types = {}
        for hook_id in list(curr_hooks.keys()):
            # If hooks are added in a different order the values will be different, but that
            #   shouldn't happen if we're building radio models in the same way
            curr_hook = curr_hooks[hook_id]
            orig_hook = orig_hooks[hook_id]
            if type(curr_hook) != type(orig_hook):
                mismatched_types[f"hook_id_{hook_id}"] = dict(
                    orig_hook=type(orig_hook),
                    curr_hook=type(curr_hook),
                )

        diff_results["mismatched_types"] = mismatched_types

    return diff_results


def _diff_module(
    curr_module: nn.Module, orig_module: nn.Module, diff_results: dict, prefix: str = ""
):
    # Rather than recursively adding, append to current dict so the results are shallower
    current_results = {}

    # Check immediate children, then call this recursively
    curr_children = dict(curr_module.named_children())
    curr_parameters = dict(curr_module.named_parameters(prefix=prefix, recurse=False))
    curr_buffers = dict(curr_module.named_buffers(prefix=prefix, recurse=False))

    orig_children = dict(orig_module.named_children())
    orig_parameters = dict(orig_module.named_parameters(prefix=prefix, recurse=False))
    orig_buffers = dict(orig_module.named_buffers(prefix=prefix, recurse=False))

    # Diff hooks
    # See https://discuss.pytorch.org/t/how-to-check-where-the-hooks-are-in-the-model/120120/9
    current_results["forward_hooks"] = _diff_module_hooks(
        curr_hooks=curr_module._forward_hooks, orig_hooks=orig_module._forward_hooks
    )
    current_results["forward_pre_hooks"] = _diff_module_hooks(
        curr_hooks=curr_module._forward_pre_hooks, orig_hooks=orig_module._forward_pre_hooks
    )
    current_results["backward_hooks"] = _diff_module_hooks(
        curr_hooks=curr_module._backward_hooks, orig_hooks=orig_module._backward_hooks
    )

    # Backward pre hooks are added in recent PR, shouldn't exist but they could in future
    current_results["backward_pre_hooks"] = _diff_module_hooks(
        curr_hooks=getattr(curr_module, "_backward_pre_hooks", {}),
        orig_hooks=getattr(orig_module, "_backward_pre_hooks", {}),
    )

    # Diff parameters
    param_diffs = dict()
    orig_param_keys = orig_parameters.keys()
    curr_param_keys = curr_parameters.keys()
    if sorted(list(orig_param_keys)) != sorted(list(curr_param_keys)):
        missing_keys = [key for key in orig_param_keys if key not in curr_param_keys]
        extra_keys = [key for key in curr_param_keys if key not in orig_param_keys]
        param_diffs.update(missing_keys=missing_keys, extra_keys=extra_keys)
    else:
        for key in orig_param_keys:
            param_diff = {}
            orig_param = orig_parameters[key]
            curr_param = curr_parameters[key]
            if orig_param.requires_grad != curr_param.requires_grad:
                param_diff["requires_grad"] = dict(
                    orig_requires_grad=orig_param.requires_grad,
                    curr_requires_grad=curr_param.requires_grad,
                )
            if orig_param.data.shape != curr_param.data.shape:
                param_diff["shape"] = dict(
                    orig_shape=orig_param.data.shape,
                    curr_shape=curr_param.data.shape,
                )
            else:
                param_diff["values"] = _diff_tensor(
                    curr_tensor=curr_param.data, orig_tensor=orig_param.data
                )

            param_diffs[key] = param_diff

    current_results["parameters"] = param_diffs

    # Diff buffers
    buff_diffs = dict()
    orig_buff_keys = orig_buffers.keys()
    curr_buff_keys = curr_buffers.keys()
    if sorted(list(orig_buff_keys)) != sorted(list(curr_buff_keys)):
        missing_keys = [key for key in orig_buff_keys if key not in curr_buff_keys]
        extra_keys = [key for key in curr_buff_keys if key not in orig_buff_keys]
        buff_diffs.update(missing_keys=missing_keys, extra_keys=extra_keys)
    else:
        for key in orig_buff_keys:
            buff_diff = {}
            orig_buff = orig_buffers[key]
            curr_buff = curr_buffers[key]
            if orig_buff.shape != curr_buff.shape:
                buff_diff["shape"] = dict(
                    orig_shape=orig_buff.shape,
                    curr_shape=curr_buff.shape,
                )
            else:
                buff_diff["values"] = _diff_tensor(
                    curr_tensor=curr_buff,
                    orig_tensor=orig_buff,
                )

            buff_diffs[key] = buff_diff

    current_results["buffers"] = buff_diffs

    # Diff current children
    children_diffs = dict()
    assert isinstance(curr_module, nn.Module) and isinstance(
        orig_module, nn.Module
    ), "Expected both curr_module and orig_module to be nn.Module's"

    if type(curr_module) != type(orig_module):
        children_diffs.update(
            module_types=dict(
                orig_module_type=type(orig_module),
                curr_module_type=type(curr_module),
            )
        )

    orig_module_keys = orig_children.keys()
    curr_module_keys = curr_children.keys()
    keys_match = sorted(list(curr_module_keys)) == sorted(list(orig_module_keys))
    if not keys_match:
        missing_keys = [key for key in orig_module_keys if key not in curr_module_keys]
        extra_keys = [key for key in curr_module_keys if key not in orig_module_keys]
        children_diffs.update(missing_keys=missing_keys, extra_keys=extra_keys)

    current_results["children"] = children_diffs

    curr_key = prefix if len(prefix) > 0 else "<self>"
    diff_results[curr_key] = current_results

    # Recurse if keys match
    if keys_match:
        for key in orig_module_keys:
            full_key = f"{prefix}.{key}" if len(prefix) > 0 else key
            curr_child_module = curr_children[key]
            orig_child_moduule = orig_children[key]

            _diff_module(
                curr_module=curr_child_module,
                orig_module=orig_child_moduule,
                prefix=full_key,
                diff_results=diff_results,
            )
