"""
The dataset object for management large scale consumer choice datasets.
Please refer to the documentation and tutorials for more details on using `ChoiceDataset`.

Author: Tianyu Du
Update: Apr. 27, 2022
"""
import copy
from typing import Dict, List, Optional, Union

import torch


class ChoiceDataset(torch.utils.data.Dataset):
    def __init__(self,
                 item_index: torch.LongTensor,
                 label: Optional[torch.LongTensor] = None,
                 user_index: Optional[torch.LongTensor] = None,
                 session_index: Optional[torch.LongTensor] = None,
                 item_availability: Optional[torch.BoolTensor] = None,
                 **kwargs) -> None:
        """
        Initialization methods for the dataset object, researchers should supply all information about the dataset
        using this initialization method.

        The number of choice instances are called `batch_size` in the documentation. The `batch_size` corresponds to the
        file length in wide-format dataset, and often denoted using `N`. We call it `batch_size` to follow the convention
        in machine learning literature.
        A `choice instance` is a row of the dataset, so there are `batch_size` choice instances in each `ChoiceDataset`.

        The dataset consists of:
        (1) a collection of `batch_size` tuples (item_id, user_id, session_id, label), where each tuple is a choice instance.
        (2) a collection of `observables` associated with item, user, session, etc.

        Args:
            item_index (torch.LongTensor): a tensor of shape (batch_size) indicating the relevant item in each row
                of the dataset, the relevant item can be:
                (1) the item bought in this choice instance,
                (2) or the item reviewed by the user. In the later case, we need the `label` tensor to specify the rating score.
                NOTE: The support for second case is under-development, currently, we are only supporting binary label.

            label (Optional[torch.LongTensor], optional): a tensor of shape (batch_size) indicating the label for prediction in
                each choice instance. While you want to predict the item bought, you can leave the `label` argument
                as `None` in the initialization method, and the model will use `item_index` as the object to be predicted.
                But if you are, for example, predicting the rating an user gave an item, label must be provided.
                Defaults to None.

            user_index (Optional[torch.LongTensor], optional): a tensor of shape num_purchases (batch_size) indicating
                the ID of the user who was involved in each choice instance. If `None` user index is provided, it's assumed
                that the choice instances are from the same user.
                `user_index` is required if and only if there are multiple users in the dataset, for example:
                    (1) user-observables is involved in the utility form,
                    (2) and/or the coefficient is user-specific.
                This tensor is used to select the corresponding user observables and coefficients assigned to the
                user (like theta_user) for making prediction for that purchase.
                Defaults to None.

            session_index (Optional[torch.LongTensor], optional): a tensor of shape num_purchases (batch_size) indicating
                the ID of the session when that choice instance occurred. This tensor is used to select the correct
                session observables or price observables for making prediction for that choice instance. Therefore, if
                there is no session/price observables, you can leave this argument as `None`. In this case, the `ChoiceDataset`
                object will assume each choice instance to be in its own session.
                Defaults to None.

            item_availability (Optional[torch.BoolTensor], optional): A boolean tensor of shape (num_sessions, num_items)
                indicating the availability of each item in each session. Utilities of unavailable items would be set to -infinite,
                and hence these unavailable items will be set to 0 while making prediction.
                We assume all items are available if set to None.
                Defaults to None.

        Other Kwargs (Observables):
            One can specify the following types of observables, where * in shape denotes any positive
                integer. Typically * represents the number of observables.
            Please refer to the documentation for a detailed guide to use observables.
            1. user observables must start with 'user_' and have shape (num_users, *)
            2. item observables must start with 'item_' and have shape (num_items, *)
            3. session observables must start with 'session_' and have shape (num_sessions, *)
            4. taste observables (those vary by user and item) must start with `taste_` and have shape
                (num_users, num_items, *).
            NOTE: we don't recommend using taste observables, because num_users * num_items is potentially large.
            5. price observables (those vary by session and item) must start with `price_` and have
                shape (num_sessions, num_items, *)
        """
        # ENHANCEMENT(Tianyu): add item_names for summary.
        super(ChoiceDataset, self).__init__()
        self.label = label
        self.item_index = item_index
        self.user_index = user_index
        self.session_index = session_index

        if self.session_index is None:
            # if any([x.startswith('session_') or x.startswith('price_') for x in kwargs.keys()]):
            # if any session sensitive observable is provided, but session index is not,
            # infer each row in the dataset to be a session.
            # TODO: (design choice) should we assign unique session index to each choice instance or the same session index.
            print('No `session_index` is provided, assume each choice instance is in its own session.')
            self.session_index = torch.arange(len(self.item_index)).long()

        self.item_availability = item_availability

        for key, item in kwargs.items():
            setattr(self, key, item)

        # TODO: add a validation procedure to check the consistency of the dataset.

    def __getitem__(self, indices: Union[int, torch.LongTensor]) -> "ChoiceDataset":
        """Retrieves samples corresponding to the provided index or list of indices.

        Args:
            indices (Union[int, torch.LongTensor]): a single integer index or a tensor of indices.

        Returns:
            ChoiceDataset: a subset of the dataset.
        """
        if isinstance(indices, int):
            # convert single integer index to an array of indices.
            indices = torch.LongTensor([indices])
        new_dict = dict()
        new_dict['item_index'] = self.item_index[indices].clone()

        # copy optional attributes.
        new_dict['label'] = self.label[indices].clone() if self.label is not None else None
        new_dict['user_index'] = self.user_index[indices].clone() if self.user_index is not None else None
        new_dict['session_index'] = self.session_index[indices].clone() if self.session_index is not None else None
        # item_availability has shape (num_sessions, num_items), no need to re-index it.
        new_dict['item_availability'] = self.item_availability

        # copy other attributes.
        for key, val in self.__dict__.items():
            if key not in new_dict.keys():
                if torch.is_tensor(val):
                    new_dict[key] = val.clone()
                else:
                    new_dict[key] = copy.deepcopy(val)
        return self._from_dict(new_dict)

    def __len__(self) -> int:
        """Returns number of samples in this dataset.

        Returns:
            int: length of the dataset.
        """
        return len(self.item_index)

    def __contains__(self, key: str) -> bool:
        return key in self.keys

    def __eq__(self, other: "ChoiceDataset") -> bool:
        """Returns whether all tensor attributes of both ChoiceDatasets are equal."""
        if not isinstance(other, ChoiceDataset):
            raise TypeError('You can only compare with ChoiceDataset objects.')
        else:
            flag = True
            for key, val in self.__dict__.items():
                if torch.is_tensor(val):
                    # ignore NaNs while comparing.
                    if not torch.equal(torch.nan_to_num(val), torch.nan_to_num(other.__dict__[key])):
                        print('Attribute {} is not equal.'.format(key))
                        flag = False
            return flag

    @property
    def device(self) -> str:
        """Returns the device of the dataset.

        Returns:
            str: the device of the dataset.
        """
        for attr in self.__dict__.values():
            if torch.is_tensor(attr):
                return attr.device

    @property
    def num_users(self) -> int:
        """Returns number of users involved in this dataset, returns 1 if there is no user identity.

        Returns:
            int: the number of users involved in this dataset.
        """
        # query from user_index
        if self.user_index is not None:
            return len(torch.unique(self.user_index))
        else:
            return 1

        # for key, val in self.__dict__.items():
        #     if torch.is_tensor(val):
        #         if self._is_user_attribute(key) or self._is_taste_attribute(key):
        #             return val.shape[0]
        # return 1

    @property
    def num_items(self) -> int:
        """Returns the number of items involved in this dataset.

        Returns:
            int: the number of items involved in this dataset.
        """
        return len(torch.unique(self.item_index))

        # for key, val in self.__dict__.items():
        #     if torch.is_tensor(val):
        #         if self._is_item_attribute(key):
        #             return val.shape[0]
        #         elif self._is_taste_attribute(key) or self._is_price_attribute(key):
        #             return val.shape[1]
        # return 1

    @property
    def num_sessions(self) -> int:
        """Returns the number of sessions involved in this dataset.

        Returns:
            int: the number of sessions involved in this dataset.
        """
        return len(torch.unique(self.session_index))

        # if self.session_index is None:
        #     return 1

        # for key, val in self.__dict__.items():
        #     if torch.is_tensor(val):
        #         if self._is_session_attribute(key) or self._is_price_attribute(key):
        #             return val.shape[0]
        # return 1

    @property
    def x_dict(self) -> Dict[object, torch.Tensor]:
        """Formats attributes of in this dataset into shape (num_sessions, num_items, num_params) and returns in a dictionary format.
        Models in this package are expecting this dictionary based data format.

        Returns:
            Dict[object, torch.Tensor]: a dictionary with attribute names in the dataset as keys, and reshaped attribute
                tensors as values.
        """
        out = dict()
        for key, val in self.__dict__.items():
            if self._is_attribute(key):  # only include attributes.
                out[key] = self._expand_tensor(key, val)  # reshape to (num_sessions, num_items, num_params).
        return out

    @classmethod
    def _from_dict(cls, dictionary: Dict[str, torch.tensor]) -> "ChoiceDataset":
        """Creates an instance of ChoiceDataset from a dictionary of arguments.

        Args:
            dictionary (Dict[str, torch.tensor]): a dictionary with keys as argument names and values as arguments.

        Returns:
            ChoiceDataset: the created copy of dataset.
        """
        dataset = cls(**dictionary)
        for key, item in dictionary.items():
            setattr(dataset, key, item)
        return dataset

    def apply_tensor(self, func: callable) -> "ChoiceDataset":
        """This s a helper method to apply the provided function to all tensors and tensor values of all dictionaries.

        Args:
            func (callable): a callable function to be applied on tensors and tensor-values of dictionaries.

        Returns:
            ChoiceDataset: the modified dataset.
        """
        for key, item in self.__dict__.items():
            if torch.is_tensor(item):
                setattr(self, key, func(item))
            # boardcast func to dictionary of tensors as well.
            elif isinstance(getattr(self, key), dict):
                for obj_key, obj_item in getattr(self, key).items():
                    if torch.is_tensor(obj_item):
                        setattr(getattr(self, key), obj_key, func(obj_item))
        return self

    def to(self, device: Union[str, torch.device]) -> "ChoiceDataset":
        """Moves all tensors in this dataset to the specified PyTorch device.

        Args:
            device (Union[str, torch.device]): the destination device.

        Returns:
            ChoiceDataset: the modified dataset on the new device.
        """
        return self.apply_tensor(lambda x: x.to(device))

    def clone(self) -> "ChoiceDataset":
        """Creates a copy of self.

        Returns:
            ChoiceDataset: a copy of self.
        """
        dictionary = {}
        for k, v in self.__dict__.items():
            if torch.is_tensor(v):
                dictionary[k] = v.clone()
            else:
                dictionary[k] = copy.deepcopy(v)
        return self.__class__._from_dict(dictionary)

    def _check_device_consistency(self) -> None:
        """Checks if all tensors in this dataset are on the same device.

        Raises:
            Exception: an exception is raised if not all tensors are on the same device.
        """
        # assert all tensors are on the same device.
        devices = list()
        for val in self.__dict__.values():
            if torch.is_tensor(val):
                devices.append(val.device)
        if len(set(devices)) > 1:
            raise Exception(f'Found tensors on different devices: {set(devices)}.',
                            'Use dataset.to() method to align devices.')

    def _size_repr(self, value: object) -> List[int]:
        """A helper method to get the string-representation of object sizes, this is helpful while constructing the
        string representation of the dataset.

        Args:
            value (object): an object to examine its size.

        Returns:
            List[int]: list of integers representing the size of the object, length of the list is equal to dimension of `value`.
        """
        if torch.is_tensor(value):
            return list(value.size())
        elif isinstance(value, int) or isinstance(value, float):
            return [1]
        elif isinstance(value, list) or isinstance(value, tuple):
            return [len(value)]
        else:
            return []

    def __repr__(self) -> str:
        """A method to get a string representation of the dataset.

        Returns:
            str: the string representation of the dataset.
        """
        info = [
            f'{key}={self._size_repr(item)}' for key, item in self.__dict__.items()]
        return f"{self.__class__.__name__}({', '.join(info)}, device={self.device})"

    # ==================================================================================================================
    # methods for checking attribute categories.
    # ==================================================================================================================
    @staticmethod
    def _is_item_attribute(key: str) -> bool:
        return key.startswith('item_') and (key != 'item_availability') and (key != 'item_index')

    @staticmethod
    def _is_user_attribute(key: str) -> bool:
        return key.startswith('user_') and (key != 'user_index')

    @staticmethod
    def _is_session_attribute(key: str) -> bool:
        return key.startswith('session_') and (key != 'session_index')

    @staticmethod
    def _is_taste_attribute(key: str) -> bool:
        return key.startswith('taste_')

    @staticmethod
    def _is_price_attribute(key: str) -> bool:
        return key.startswith('price_')

    def _is_attribute(self, key: str) -> bool:
        return self._is_item_attribute(key) \
            or self._is_user_attribute(key) \
            or self._is_session_attribute(key) \
            or self._is_taste_attribute(key) \
            or self._is_price_attribute(key)

    def _expand_tensor(self, key: str, val: torch.Tensor) -> torch.Tensor:
        """Expands attribute tensor to (num_sessions, num_items, num_params) shape for prediction tasks, this method
        won't reshape the tensor at all if the `key` (i.e., name of the tensor) suggests its not an attribute of any kind.

        Args:
            key (str): name of the attribute used to determine the raw shape of the tensor. For example, 'item_obs' means
                the raw tensor is in shape (num_items, num_params).
            val (torch.Tensor): the attribute tensor to be reshaped.

        Returns:
            torch.Tensor: the reshaped tensor with shape (num_sessions, num_items, num_params).
        """
        if not self._is_attribute(key):
            print(f'Warning: the input key {key} is not an attribute of the dataset, will NOT modify the provided tensor.')
            # don't expand non-attribute tensors, if any.
            return val

        num_params = val.shape[-1]
        if self._is_user_attribute(key):
            # user_attribute (num_users, *)
            out = val[self.user_index, :].view(
                len(self), 1, num_params).expand(-1, self.num_items, -1)
        elif self._is_item_attribute(key):
            # item_attribute (num_items, *)
            out = val.view(1, self.num_items, num_params).expand(
                len(self), -1, -1)
        elif self._is_session_attribute(key):
            # session_attribute (num_sessions, *)
            out = val[self.session_index, :].view(
                len(self), 1, num_params).expand(-1, self.num_items, -1)
        elif self._is_taste_attribute(key):
            # taste_attribute (num_users, num_items, *)
            out = val[self.user_index, :, :]
        elif self._is_price_attribute(key):
            # price_attribute (num_sessions, num_items, *)
            out = val[self.session_index, :, :]

        assert out.shape == (len(self), self.num_items, num_params)
        return out
