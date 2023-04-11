from copy import copy, deepcopy
from typing import Optional, Tuple

import torch
import torch.nn as nn


def parameter_std(model_trained: nn.Module, loss_fn: callable) -> Tuple[dict, Optional[torch.Tensor]]:
    """This method firstly computes the Hessian of loss_fn(model_trained) with respect to
    model_trained.parameters(), then computes the standard error from the Hessian.

    NOTE: the current implementation involving deletion of attributes in model, this is an unsafe
    workaround for now. See https://github.com/pytorch/pytorch/issues/50138 for details.

    Args:
        model_trained (nn.Module): a trained pytorch model, the std estimated from Hessian only works
            if the model has been trained to optimal.
        loss_fn (callable): the negatigve log-likelihood function (loss function).
        return_hessian (bool): request to return hessian matrix as well.

    Returns:
        [dict]: a dictionary maps from keys in model_train.state_dict() to standard errors of esimations
            of each parameters in model_train.parameters(), shapes of values of returned dictionary
            is the same as shapes in model_train.state_dict().
        [torch.Tensor]: optionally return the Hessian of loss_fn(model_trained) w.r.t. model_trained.parameters()
    """
    # Need to make this safe.
    model = copy(model_trained)
    state_dict = deepcopy(model.state_dict())

    shape, start, end = dict(), dict(), dict()
    param_list = list()
    s = 0
    # wrap state dict into a single one dimensional tensor.
    for k, v in state_dict.items():
        num_params = state_dict[k].numel()
        start[k], end[k] = (s, s + num_params)
        s += num_params
        shape[k] = v.shape
        param_list.append(v.clone().view(-1,))
    all_params = torch.cat(param_list)

    def func(input_tensor):
        # unwrap parameters.
        for k in state_dict.keys():
            src = input_tensor[start[k]: end[k]].view(*shape[k])
            # NOTE: The removeprefix and removesuffix require Python >= 3.9!
            # variable_name = k.removeprefix("coef_dict.").removesuffix(".coef")
            # less elegant/robust but supports earlier versions of python.
            # examples: k = "coef_dict.x1[user].coef" for conditional logit models
            # k = "item_coef_dict.x1[user].coef" or "nest_coef_dict.x1[user].coef" for nested logit models.
            # prefix = "coef_dict."
            # suffix = ".coef"

            if k == "lambda_weight":
                # this is a special case in nested logit models.
                del model.lambda_weight
                model.lambda_weight = src
            else:
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
