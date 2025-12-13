from __future__ import annotations

from copy import deepcopy
from typing import Callable, Dict

import torch
import torch.nn as nn


def _resolve_parent(module: nn.Module, param_name: str) -> tuple[object, str]:
    """Resolve a dotted parameter name (from `named_parameters`) to its parent object and attribute.

    Supports `nn.ModuleDict` keys (which appear as path segments in the parameter name).
    """
    parts = param_name.split(".")
    current: object = module
    for part in parts[:-1]:
        # Support containers where children are accessed by key/index rather than attribute name.
        if isinstance(current, nn.ModuleDict):
            current = current[part]
        elif isinstance(current, (nn.ModuleList, nn.Sequential)) and part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current, parts[-1]


def parameter_std(
    model_trained: nn.Module,
    loss_fn: Callable[[nn.Module], torch.Tensor],
    *,
    damping: float = 1e-6,
    use_double: bool = True,
) -> Dict[str, torch.Tensor]:
    """Compute parameter standard errors via (regularized) inverse Hessian of the loss.

    This method computes the Hessian of loss_fn(model_trained) with respect to
    model_trained.parameters(), then returns standard errors derived from the Hessian.

    NOTE: The current implementation involves deletion of attributes in the model, which is an
    unsafe workaround. See https://github.com/pytorch/pytorch/issues/50138 for details.

    Args:
        model_trained (nn.Module): a trained PyTorch model. The Hessian-based std only makes sense
            if the model has been trained to optimum.
        loss_fn (callable): the negative log-likelihood (loss) function taking the model and
            returning a scalar loss tensor.
        damping (float): ridge term added to the Hessian diagonal before inversion for numerical stability.
        min_variance (float): lower bound applied to diagonal variances before taking square root to avoid NaNs.
        use_double (bool): compute Hessian/inverse in float64 for improved numerical stability.

    Returns:
        Dict[str, torch.Tensor]: mapping from parameter names (as in `model.named_parameters()`)
            to tensors of standard errors with the same shapes as the corresponding parameters.
    """
    # Standard-error computation is expensive; be safe and avoid mutating the caller's model.
    # (The Hessian routine below temporarily replaces parameters with tensors.)
    model = deepcopy(model_trained)

    # NestedLogitModel historically may create a parameter alias `lambdas` at runtime by assigning
    # `self.lambdas = self.lambda_weight` inside forward(). That registers `lambdas` as a Parameter,
    # and later forward passes will try to overwrite it with a Tensor when we replace parameters
    # during Hessian evaluation (especially in float64), causing:
    #   TypeError: cannot assign 'torch.*Tensor' as parameter 'lambdas'
    # For std-error computation we only need the canonical `lambda_weight`, so drop the alias.
    if hasattr(model, "_parameters") and isinstance(model._parameters, dict):
        if "lambdas" in model._parameters:
            del model._parameters["lambdas"]
    if hasattr(model, "lambdas"):
        try:
            delattr(model, "lambdas")
        except Exception:
            # Safe fallback: leave attribute; removing from _parameters above is the key piece.
            pass

    params = [
        (name, p)
        for name, p in model.named_parameters()
        if p.requires_grad and name != "lambdas"
    ]
    if not params:
        raise ValueError("No trainable parameters found in the model.")

    shape: Dict[str, torch.Size] = {}
    start: Dict[str, int] = {}
    end: Dict[str, int] = {}
    param_list = []
    s = 0
    # Wrap parameters into a single 1D tensor.
    for name, p in params:
        num_params = p.numel()
        start[name], end[name] = (s, s + num_params)
        s += num_params
        shape[name] = p.shape
        param_list.append(p.detach().clone().reshape(-1))
    all_params = torch.cat(param_list)
    if use_double:
        all_params = all_params.double()

    def func(input_tensor: torch.Tensor) -> torch.Tensor:
        # Unwrap parameters back into the cloned model.
        # We replace nn.Parameter leaves with tensors that are views into `input_tensor` so autograd
        # can compute derivatives w.r.t. `input_tensor`.
        for name, _ in params:
            src = input_tensor[start[name] : end[name]].view(*shape[name])
            parent, attr = _resolve_parent(model, name)
            # Drop the Parameter so the new tensor is used in forward.
            try:
                delattr(parent, attr)
            except AttributeError:
                # Some ModuleDict-managed parameters may not support delattr cleanly across versions;
                # fall back to overwriting.
                pass
            setattr(parent, attr, src)

        out = loss_fn(model)
        if out.dtype != input_tensor.dtype:
            out = out.to(input_tensor.dtype)
        return out

    H = torch.autograd.functional.hessian(func, all_params)
    # Symmetrize (Hessian should be symmetric; numerical noise can break this).
    H = 0.5 * (H + H.T)
    if use_double:
        H = H.double()

    # Add ridge damping to improve invertibility.
    dim = H.shape[0]
    if dim == 0:
        raise ValueError("Empty Hessian.")
    eye = torch.eye(dim, device=H.device, dtype=H.dtype)
    H_damped = H + float(damping) * eye

    # Invert (fall back to pseudo-inverse for near-singular Hessians).
    try:
        cov = torch.linalg.inv(H_damped)
    except torch.linalg.LinAlgError:
        cov = torch.linalg.pinv(H_damped)
    cov = 0.5 * (cov + cov.T)

    var = torch.diagonal(cov)
    # Guard against numerical issues: if a variance is non-finite or non-positive, report an
    # effectively infinite standard error (conservative; avoids NaNs in downstream tables).
    bad = (~torch.isfinite(var)) | (var <= 0)
    safe_var = torch.where(bad, torch.ones_like(var), var)
    std_all = torch.sqrt(safe_var)
    std_all = torch.where(bad, torch.full_like(std_all, float("inf")), std_all)

    std_dict: Dict[str, torch.Tensor] = {}
    for name, _ in params:
        std_dict[name] = std_all[start[name] : end[name]].view(*shape[name]).cpu()

    return std_dict
