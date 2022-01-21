"""
The dataset class for consumer choice datasets.
Supports for uit linux style naming for variables.
"""
import copy
from typing import List, Dict, Optional, Union
import torch


class ChoiceDataset(torch.utils.data.Dataset):
    def __init__(self,
                 label: torch.LongTensor,
                 user_index: Optional[torch.LongTensor] = None,
                 session_index: Optional[torch.LongTensor] = None,
                 item_availability: Optional[torch.BoolTensor] = None,
                 **kwargs) -> None:
        """

        Args:
            label (torch.LongTensor): a tensor of shape num_purchases (batch_size) indicating the ID
                of the item bought.
            user_index (Optional[torch.LongTensor], optional): used only if there are multiple users
                in the dataset, a tensor of shape num_purchases (batch_size) indicating the ID of the
                user who purchased. This tensor is used to select the corresponding user observables and
                coefficients assigned to the user (like theta_user) for making prediction for that
                purchase.
                Defaults to None.
            session_index (Optional[torch.LongTensor], optional): used only if there are multiple
                sessions in the dataset, a tensor of shape num_purchases (batch_size) indicating the
                ID of the session when that purchase occurred. This tensor is used to select the correct
                session observables or price observables for making prediction for that purchase.
                Defaults to None.
            item_availability (Optional[torch.BoolTensor], optional): assume all items are available
                if set to None. A tensor of shape (num_sessions, num_items) indicating the availability
                of each item in each session.
                Defaults to None.

        Other Kwargs (Observables):
            One can specify the following types of observables, where * in shape denotes any positive
                integer. Typically * represents the number of observables.
            1. user observables must start with 'user_' and have shape (num_users, *)
            2. item observables must start with 'item_' and have shape (num_items, *)
            3. session observables must start with 'session_' and have shape (num_sessions, *)
            4. taste observables (those vary by user and item) must start with `taste_` and have shape
                (num_users, num_items, *).
            5. price observables (those vary by session and item) must start with `price_` and have
                shape (num_sessions, num_items, *)
        """
        # ENHANCEMENT(Tianyu): add item_names for summary.
        super(ChoiceDataset, self).__init__()
        self.label = label
        self.user_index = user_index
        self.session_index = session_index

        if self.session_index is None:
            if any([x.startswith('session_') or x.startswith('price_') for x in kwargs.keys()]):
                # if any session sensitive observable is provided, but session index is not,
                # infer each row in the dataset to be a session.
                self.session_index = torch.arange(len(self.label)).long()

        self.item_availability = item_availability

        self.observable_prefix = [
            'user_', 'item_', 'session_', 'taste_', 'price_']
        for key, item in kwargs.items():
            if any(key.startswith(prefix) for prefix in self.observable_prefix):
                setattr(self, key, item)

        self._is_valid()

    # @staticmethod
    # def _dict_index(d, indices) -> dict:
    #     # subset values of dictionary using the provided index, this method only subsets tensors and
    #     # keeps other values unchanged.
    #     subset = dict()
    #     for key, val in d.items():
    #         if torch.is_tensor(val):
    #             subset[key] = val[indices, ...]
    #         else:
    #             subset[key] = val
    #     return subset

    def __getitem__(self, indices: Union[int, torch.LongTensor]):
        # TODO: Do we really need to initialize a new ChoiceDataset object?
        new_dict = dict()

        new_dict['label'] = self.label[indices]

        if self.user_index is None:
            new_dict['user_index'] = None
        else:
            new_dict['user_index'] = self.user_index[indices]

        if self.session_index is None:
            new_dict['session_index'] = None
        else:
            new_dict['session_index'] = self.session_index[indices]

        # item_availability has shape (num_sessions, num_items), no need
        # to index it.
        new_dict['item_availability'] = self.item_availability
        # copy other keys.
        for key, val in self.__dict__.items():
            if key in new_dict.keys():
                # ignore 'label', 'user_index', 'session_index' and 'item_availability' keys, already added.
                continue
            if torch.is_tensor(val):
                new_dict[key] = val.clone()
            else:
                new_dict[key] = val
        return self._from_dict(new_dict)

    def __len__(self) -> int:
        return len(self.label)

    def __contains__(self, key: str) -> bool:
        return key in self.keys

    @property
    def device(self) -> str:
        return self.label.device

    @property
    def num_users(self) -> int:
        for key, val in self.__dict__.items():
            if torch.is_tensor(val):
                if self._is_user_attribute(key) or self._is_taste_attribute(key):
                    return val.shape[0]
        return 1

    @property
    def num_items(self) -> int:
        for key, val in self.__dict__.items():
            if torch.is_tensor(val):
                if self._is_item_attribute(key):
                    return val.shape[0]
                elif self._is_taste_attribute(key) or self._is_price_attribute(key):
                    return val.shape[1]
        return 1

    @property
    def num_sessions(self) -> int:
        if self.session_index is None:
            return 1

        for key, val in self.__dict__.items():
            if torch.is_tensor(val):
                if self._is_session_attribute(key) or self._is_price_attribute(key):
                    return val.shape[0]
        return 1

    @property
    def x_dict(self) -> Dict[object, torch.Tensor]:
        """Get the x_dict object for used in model's forward function."""
        # reshape raw tensors into (num_sessions, num_items/num_category, num_params).
        out = dict()
        for key, val in self.__dict__.items():
            if self._is_attribute(key):
                out[key] = self._expand_tensor(key, val)
        # ENHANCEMENT(Tianyu): cache results, check performance.
        return out

    @classmethod
    def _from_dict(cls, dictionary: Dict[str, torch.tensor]):
        dataset = cls(**dictionary)
        for key, item in dictionary.items():
            setattr(dataset, key, item)
        return dataset

    def apply_tensor(self, func):
        for key, item in self.__dict__.items():
            if torch.is_tensor(item):
                setattr(self, key, func(item))
            elif isinstance(getattr(self, key), dict):
                for obj_key, obj_item in getattr(self, key).items():
                    if torch.is_tensor(obj_item):
                        setattr(getattr(self, key), obj_key, func(obj_item))
        return self

    def to(self, device):
        return self.apply_tensor(lambda x: x.to(device))

    def clone(self):
        dictionary = {}
        for k, v in self.__dict__.items():
            if torch.is_tensor(v):
                dictionary[k] = v.clone()
            else:
                dictionary[k] = copy.deepcopy(v)
        return self.__class__._from_dict(dictionary)

    def _check_device_consistency(self):
        # assert all tensors are on the same device.
        devices = list()
        for val in self.__dict__.values():
            if torch.is_tensor(val):
                devices.append(val.device)
        if len(set(devices)) > 1:
            raise Exception(f'Found tensors on different devices: {set(devices)}.',
                            'Use dataset.to() method to align devices.')

    def _size_repr(self, value) -> List[int]:
        if torch.is_tensor(value):
            return list(value.size())
        elif isinstance(value, int) or isinstance(value, float):
            return [1]
        elif isinstance(value, list) or isinstance(value, tuple):
            return [len(value)]
        else:
            return []

    def __repr__(self):
        info = [
            f'{key}={self._size_repr(item)}' for key, item in self.__dict__.items()]
        return f"{self.__class__.__name__}({', '.join(info)}, device={self.device})"

    @staticmethod
    def _is_item_attribute(key: str) -> bool:
        return key.startswith('item_') and key != 'item_availability'

    @staticmethod
    def _is_user_attribute(key: str) -> bool:
        return key.startswith('user_') and key != 'user_index'

    @staticmethod
    def _is_session_attribute(key: str) -> bool:
        return key.startswith('session_') and key != 'session_index'

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

    def _is_valid(self):
        batch_size = len(self.label)
        if self.user_index is not None:
            assert self.user_index.shape == (batch_size,)

        if self.session_index is not None:
            assert self.session_index.shape == (batch_size,)

        # infer some information.
        num_sessions = None
        if self.item_availability is not None:
            num_sessions = self.item_availability.shape[0]

        for key, value in self.__dict__.items():
            if self._is_session_attribute(key) or self._is_price_attribute(key):
                num_sessions = value.shape[0]

        if any(self._is_user_attribute(x) for x in self.__dict__.keys()):
            assert self.user_index is not None

        if any(self._is_session_attribute(x) or self._is_price_attribute(x) for x in self.__dict__.keys()):
            assert self.session_index is not None

    def _expand_tensor(self, key: str, val: torch.Tensor) -> torch.Tensor:
        # convert raw tensors into (len(self), num_items/num_category, num_params).
        if not self._is_attribute(key):
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
