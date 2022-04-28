"""
The JointDataset class is a wrapper for the torch.utils.data.ChoiceDataset class, it is particularly useful when we
need to make prediction from multiple datasets. For example, you have data on consumer purchase records in a fast food
store, and suppose every customer will purchase exactly a single main food and a single drink. In this case, you have
two separate datasets: FoodDataset and DrinkDataset. You may want to use PyTorch sampler to sample them in a dependent
manner: you want to take the i-th sample from both datasets, so that you know what (food, drink) combo the i-th customer
purchased. You can do this by using the JointDataset class.

Author: Tianyu Du
Update: Apr. 28, 2022
"""
from typing import Union, Dict
import torch

from torch_choice.data.choice_dataset import ChoiceDataset


class JointDataset(torch.utils.data.Dataset):
    """A helper class for joining several pytorch datasets, using JointDataset
    and pytorch data loader allows for sampling the same batch index from several
    datasets.

    The JointDataset class is a wrapper for the torch.utils.data.ChoiceDataset class, it is particularly useful when we
    need to make prediction from multiple datasets. For example, you have data on consumer purchase records in a fast food
    store, and suppose every customer will purchase exactly a single main food and a single drink. In this case, you have
    two separate datasets: FoodDataset and DrinkDataset. You may want to use PyTorch sampler to sample them in a dependent
    manner: you want to take the i-th sample from both datasets, so that you know what (food, drink) combo the i-th customer
    purchased. You can do this by using the JointDataset class.
    """
    def __init__(self, **datasets) -> None:
        """The initialize methods.

        Args:
            Arbitrarily many datasets with arbitrary names as keys. In the example above, you can construct
            ```
            dataset = JointDataset(food=FoodDataset, drink=DrinkDataset)
            ```
            All datasets should have the same length.

        """
        super(JointDataset, self).__init__()
        self.datasets = datasets
        # check the length of sub-datasets are the same.
        assert len(set([len(d) for d in self.datasets.values()])) == 1

    def __len__(self) -> int:
        """Get the number of samples in the joint dataset.

        Returns:
            int: the number of samples in the joint dataset, which is the same as the number of samples in each dataset contained.
        """
        for d in self.datasets.values():
            return len(d)

    def __getitem__(self, indices: Union[int, torch.LongTensor]) -> Dict[str, ChoiceDataset]:
        """Queries samples from the dataset by index.

        Args:
            indices (Union[int, torch.LongTensor]): an integer or a 1D tensor of multiple indices.

        Returns:
            Dict[str, ChoiceDataset]: the subset of the dataset. Keys of the dictionary will be names of each dataset
                contained (the same as the keys of the ``datasets`` argument in the constructor). Values will be subsets
                of contained datasets, sliced using the provided indices.
        """
        return dict((name, d[indices]) for (name, d) in self.datasets.items())

    def __repr__(self) -> str:
        """A method to get a string representation of the dataset.

        Returns:
            str: the string representation of the dataset.
        """
        out = [f'JointDataset with {len(self.datasets)} sub-datasets: (']
        for name, dataset in self.datasets.items():
            out.append(f'\t{name}: {str(dataset)}')
        out.append(')')
        return '\n'.join(out)

    @property
    def device(self) -> str:
        """Returns the device of datasets contained in the joint dataset.

        Returns:
            str: the device of the dataset.
        """
        for d in self.datasets.values():
            return d.device

    def to(self, device: Union[str, torch.device]) -> "JointDataset":
        """Moves all datasets in this dataset to the specified PyTorch device.

        Args:
            device (Union[str, torch.device]): the destination device.

        Returns:
            ChoiceDataset: the modified dataset on the new device.
        """
        for d in self.datasets.values():
            d = d.to(device)
        return self
