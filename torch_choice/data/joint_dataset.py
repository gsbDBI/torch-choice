"""
Constructors for joining multiple datasets together.
"""
from typing import Union
import torch


class JointDataset(torch.utils.data.Dataset):
    """A helper class for joining several pytorch datasets, using JointDataset
    and pytorch data loader allows for sampling the same batch index from several
    datasets.
    """
    def __init__(self, **datasets):
        super(JointDataset, self).__init__()
        self.datasets = datasets
        # check the length of sub-datasets are the same.
        assert len(set([len(d) for d in self.datasets.values()])) == 1

    def __len__(self) -> int:
        for d in self.datasets.values():
            return len(d)

    def __getitem__(self, indices: Union[int, torch.LongTensor]):
        return dict((name, d[indices]) for (name, d) in self.datasets.items())

    def __repr__(self) -> str:
        out = [f'JointDataset with {len(self.datasets)} sub-datasets: (']
        for name, dataset in self.datasets.items():
            out.append(f'\t{name}: {str(dataset)}')
        out.append(')')
        return '\n'.join(out)

    @property
    def device(self):
        for d in self.datasets.values():
            return d.device

    def to(self, device):
        for d in self.datasets.values():
            d = d.to(device)
        return self
