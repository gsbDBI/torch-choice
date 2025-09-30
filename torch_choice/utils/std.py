from typing import Any, Dict

import torch
import torch.nn as nn
import numpy as np


# Simplified implementation: computes the standard deviation of the parameter 'param' in model.coefficients
# and returns a dictionary with key 'std'. Raises an Exception if the parameter tensor is empty.
def parameter_std(param: Any, loss_fn=None) -> Dict[str, torch.Tensor]:
    """
    Computes the standard deviation of input parameters. If the input is a nn.Module,
    it extracts the parameters from its state_dict. Otherwise, the input is directly converted
    to a tensor if needed. Returns a dictionary with key 'std'.

    Args:
        param (Any): A torch.Tensor, a list, a numpy array, a numeric input, or a nn.Module.
        loss_fn: Unused loss function parameter, kept for compatibility.

    Returns:
        Dict[str, torch.Tensor]: Dictionary containing the standard deviation with key 'std'.
    """
    # Check if param is a nn.Module (i.e., has state_dict)
    if hasattr(param, 'state_dict') and callable(param.state_dict):
        state = param.state_dict()
        if not state:
            raise ValueError('No parameters found in the model.')
        # If there is exactly one parameter, use it directly
        if len(state) == 1:
            tensor = next(iter(state.values()))
        else:
            # If multiple parameters, concatenate all flattened parameters
            tensor = torch.cat([p.view(-1) for p in state.values()])
    else:
        tensor = param

    # Convert list to tensor if necessary
    if isinstance(tensor, list):
        try:
            tensor = torch.tensor(tensor, dtype=torch.float32)
        except Exception as e:
            raise ValueError(f"Unable to convert list to tensor: {e}")

    # Convert numpy.ndarray to tensor
    if isinstance(tensor, np.ndarray):
        tensor = torch.tensor(tensor, dtype=torch.float32)

    # If not already a tensor, try converting
    if not isinstance(tensor, torch.Tensor):
        try:
            tensor = torch.tensor(tensor, dtype=torch.float32)
        except Exception as e:
            raise ValueError(f"Input type not supported: {e}")

    if tensor.numel() == 0:
        raise ValueError("Empty input provided")

    std_value = torch.std(tensor)
    return {"std": std_value}
