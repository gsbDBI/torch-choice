
import os
from typing import Union, List

import pandas as pd
import torch
from torch.utils.data.sampler import BatchSampler, SequentialSampler, RandomSampler


def pivot3d(df: pd.DataFrame, dim0: str, dim1: str, values: Union[str, List[str]]) -> torch.Tensor:
    """
    Creates a tensor of shape (df[dim0].nunique(), df[dim1].nunique(), len(values)) from the
    provided data frame.

    Example, if dim0 is the column of session ID, dim1 is the column of alternative names, then
        out[t, i, k] is the feature values[k] of item i in session t. The returned tensor
        has shape (num_sessions, num_items, num_params), which fits the purpose of conditioanl
        logit models.
    """
    if not isinstance(values, list):
        values = [values]

    dim1_list = sorted(df[dim1].unique())

    tensor_slice = list()
    for value in values:
        layer = df.pivot(index=dim0, columns=dim1, values=value)
        tensor_slice.append(torch.Tensor(layer[dim1_list].values))

    tensor = torch.stack(tensor_slice, dim=-1)
    assert tensor.shape == (df[dim0].nunique(), df[dim1].nunique(), len(values))
    return tensor


def create_data_loader(dataset, batch_size: int = -1, shuffle: bool = False, num_workers: int = 0):
    if batch_size == -1:
        # use full-batch.
        batch_size = len(dataset)

    sampler = BatchSampler(
        RandomSampler(dataset) if shuffle else SequentialSampler(dataset),
        batch_size=batch_size,
        drop_last=False)
    # feed a batch_sampler as sampler so that dataset.__getitem__ is called with a list of indices.
    # cannot use multiple workers if the entire dataset is already on GPU.
    dataloader = torch.utils.data.DataLoader(dataset,
                                             sampler=sampler,
                                             num_workers=num_workers,
                                             collate_fn=lambda x: x[0],
                                             pin_memory=(dataset.device == 'cpu'))
    return dataloader
