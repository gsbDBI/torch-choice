from copy import copy, deepcopy
from typing import Optional, Tuple

import torch
import torch.nn as nn


def parameter_std(model_trained: nn.Module, loss_fn: callable) -> Tuple[dict, Optional[torch.Tensor]]:
    """Compute standard errors of parameters via the inverse Hessian of the loss.

    This method computes the Hessian of loss_fn(model_trained) with respect to
    model_trained.parameters(), then returns standard errors derived from the Hessian.

    NOTE: The current implementation involves deletion of attributes in the model, which is an
    unsafe workaround. See https://github.com/pytorch/pytorch/issues/50138 for details.

    Args:
        model_trained (nn.Module): a trained PyTorch model. The Hessian-based std only makes sense
            if the model has been trained to optimum.
        loss_fn (callable): the negative log-likelihood (loss) function taking the model and
            returning a scalar loss tensor.

    Returns:
        Tuple[dict, Optional[torch.Tensor]]: A dictionary mapping keys from model_trained.state_dict()
            to the standard errors with the same shapes as the corresponding parameters; and the
            Hessian tensor if needed by callers in the future.
    """
    # Work on a copy to avoid mutating the original model.
    model = copy(model_trained)
    state_dict = deepcopy(model.state_dict())

    shape, start, end = dict(), dict(), dict()
    param_list = list()
    s = 0
    # Wrap state dict into a single one dimensional tensor.
    for k, v in state_dict.items():
        num_params = state_dict[k].numel()
        start[k], end[k] = (s, s + num_params)
        s += num_params
        shape[k] = v.shape
        param_list.append(v.clone().view(-1,))
    if len(param_list) == 0:
        raise ValueError('No parameters found in the model.')
    all_params = torch.cat(param_list)

    def func(input_tensor: torch.Tensor) -> torch.Tensor:
        # Unwrap parameters back into the cloned model.
        for k in state_dict.keys():
            src = input_tensor[start[k]: end[k]].view(*shape[k])

            if k == "lambda_weight":
                # Special handling in nested logit models
                del model.lambda_weight
                model.lambda_weight = src
            else:
                # Keys look like:
                #   - "coef_dict.x1[user].coef" for conditional logit models
                #   - "item_coef_dict.x1[user].coef" or "nest_coef_dict.x1[user].coef" for nested logit models
                coef_dict, variable_name = k.split(".")[0], k.split(".")[1]
                del getattr(model, coef_dict)[variable_name].coef
                getattr(model, coef_dict)[variable_name].coef = src

        return loss_fn(model)

    H = torch.autograd.functional.hessian(func, all_params)

    std_all = torch.sqrt(torch.diag(torch.inverse(H)))
    std_dict = dict()
    for k in state_dict.keys():
        std_dict[k] = std_all[start[k]: end[k]].view(*shape[k]).cpu()

    return std_dict
