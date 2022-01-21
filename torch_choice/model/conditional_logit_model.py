"""
Conditional Logit Model, the generalized version of the `cmclogit' command in Stata.
This is the most general implementation of the logit model class.

Author: Tianyu Du
Date: Aug. 8, 2021
"""
from copy import deepcopy
from typing import Dict, Optional, Tuple, Union
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepchoice.model.coefficient import Coefficient


class ConditionalLogitModel(nn.Module):
    """The more generalized version of conditional logit model, the model allows for research specific
    variable types(groups) and different levels of variations for coefficeints.

    The model allows for the following levels for variable variations:
    NOTE: unless the `-full` flag is specified (which means we want to explicitly model coefficients
        for all items), for all variation levels related to item (item specific and user-item specific),
        the model force coefficients for the first item to be zero. This design follows stadnard
        econometric practice.

    - constant: constant over all users and items,

    - user: user-specific parameters but constant across all items,

    - item: item-specific parameters but constant across all users, parameters for the first item are
        forced to be zero.
    - item-full: item-specific parameters but constant across all users, explicitly model for all items.

    - user-item: parameters that are specific to both user and item, parameterts for the first item
        for all users are forced to be zero.
    - user-item-full: parameters that are specific to both user and item, explicitly model for all items.
    """

    def __init__(self,
                 coef_variation_dict: Dict[str, str],
                 num_param_dict: Dict[str, int],
                 num_items: Optional[int]=None,
                 num_users: Optional[int]=None
                 ) -> None:
        """
        Args:
            num_items (int): number of items in the dataset.
            num_users (int): number of users in the dataset.
            coef_variation_dict (Dict[str, str]): variable type to variation level dictionary.
                Put None or 'zero' if there is no this kind of variable in the model.
            num_param_dict (Dict[str, int]): variable type to number of parameters dictionary,
                records number of features in each kind of variable.
                Put None if there is no this kind of variable in the model.
        """
        super(ConditionalLogitModel, self).__init__()

        assert coef_variation_dict.keys() == num_param_dict.keys()

        self.variable_types = list(deepcopy(num_param_dict).keys())

        self.coef_variation_dict = deepcopy(coef_variation_dict)
        self.num_param_dict = deepcopy(num_param_dict)

        self.num_items = num_items
        self.num_users = num_users

        # check number of parameters specified are all positive.
        for var_type, num_params in self.num_param_dict.items():
            assert num_params > 0, f'num_params needs to be positive, got: {num_params}.'

        # infer the number of parameters for intercept if the researcher forgets.
        if 'intercept' in self.coef_variation_dict.keys() and 'intercept' not in self.num_param_dict.keys():
            warnings.warn("'intercept' key found in coef_variation_dict but not in num_param_dict, num_param_dict['intercept'] has been set to 1.")
            self.num_param_dict['intercept'] = 1

        # construct trainable parameters.
        coef_dict = dict()
        for var_type, variation in self.coef_variation_dict.items():
            coef_dict[var_type] = Coefficient(variation=variation,
                                              num_items=self.num_items,
                                              num_users=self.num_users,
                                              num_params=self.num_param_dict[var_type])
        # A ModuleDict is required to properly register all trainiabel parameters.
        # self.parameter() will fail if a python dictionary is used instead.
        self.coef_dict = nn.ModuleDict(coef_dict)

    def __repr__(self) -> str:
        out_str_lst = ['Conditional logistic discrete choice model, expects input features:\n']
        for var_type, num_params in self.num_param_dict.items():
            out_str_lst.append(f'X[{var_type}] with {num_params} parameters, with {self.coef_variation_dict[var_type]} level variation.')
        return super().__repr__() + '\n' + '\n'.join(out_str_lst)

    @property
    def num_params(self) -> int:
        return sum(w.numel() for w in self.parameters())

    def summary(self):
        for var_type, coefficient in self.coef_dict.items():
            if coefficient is not None:
                print('Variable Type: ', var_type)
                print(coefficient.coef)

    def forward(self,
                batch,
                manual_coef_value_dict: Optional[Dict[str, torch.Tensor]] = None
                ) -> torch.Tensor:
        """
        The forward function with explicit arguments, this forward function is for internal usages
        only, reserachers should use the forward() function insetad.

        Args:
            batch:
            manual_coef_value_dict (Optional[Dict[str, torch.Tensor]], optional): a dictionary with
                keys in {'u', 'i'} etc and tensors as values. If provided, the model will force
                coefficient to be the provided values and compute utility conditioned on the provided
                coefficient values. This feature is useful when the research wishes to plug in particular
                values of coefficients and exmaine the utility values. If not provided, the model will
                use the learned coefficient values in self.coef_dict.
                Defaults to None.

        Returns:
            torch.Tensor: a tensor of shape (num_trips, num_items) whose (t, i) entry represents
                the utility from item i in trip t for the user involved in that trip.
        """
        x_dict = batch.x_dict

        if 'intercept' in self.coef_variation_dict.keys():
            # intercept term has no input tensor, which has only 1 feature.
            x_dict['intercept'] = torch.ones((len(batch), self.num_items, 1), device=batch.device)

        # compute the utility from each item in each choice session.
        total_utility = torch.zeros((len(batch), self.num_items), device=batch.device)
        # for each type of variables, apply the corresponding coefficient to input x.

        for var_type, coef in self.coef_dict.items():
            total_utility += coef(
                x_dict[var_type], batch.user_index,
                manual_coef_value=None if manual_coef_value_dict is None else manual_coef_value_dict[var_type])

        assert total_utility.shape == (len(batch), self.num_items)

        if batch.item_availability is not None:
            # mask out unavilable items.
            total_utility[~batch.item_availability[batch.session_index, :]] = -1.0e20
        return total_utility

    @staticmethod
    def flatten_coef_dict(coef_dict: Dict[str, Union[torch.Tensor, torch.nn.Parameter]]) -> Tuple[torch.Tensor, dict]:
        """Flattens the coef_dict into a 1-dimension tensor, used for hessian computation."""
        type2idx = dict()
        param_list = list()
        start = 0

        for var_type in coef_dict.keys():
            num_params = coef_dict[var_type].coef.numel()
            # track which portion of all_param tensor belongs to this variable type.
            type2idx[var_type] = (start, start + num_params)
            start += num_params
            # use reshape instead of view to make a copy.
            param_list.append(coef_dict[var_type].coef.clone().reshape(-1,))

        all_param = torch.cat(param_list)  # (self.num_params(), )
        return all_param, type2idx

    @staticmethod
    def unwrap_coef_dict(param: torch.Tensor, type2idx: Dict[str, Tuple[int, int]]) -> Dict[str, torch.Tensor]:
        """Rebuild coef_dict from output of self.flatten_coef_dict method."""
        coef_dict = dict()
        for var_type in type2idx.keys():
            start, end = type2idx[var_type]
            # no need to reshape here, Coefficient handles it.
            coef_dict[var_type] = param[start:end]
        return coef_dict

    def compute_hessian(self, x_dict, availability, user_index, y) -> torch.Tensor:
        """Computes the hessian of negaitve log-likelihood (total cross-entropy loss) with respect
        to all parameters in this model.

        Args:
            x_dict ,availability, user_index: see definitions in self._forward.
            y (torch.LongTensor): a tensor with shape (num_trips,) of IDs of items actually purchased.

        Returns:
            torch.Tensor: a (self.num_params, self.num_params) tensor of the Hessian matrix.
        """
        all_coefs, type2idx = self.flatten_coef_dict(self.coef_dict)

        def compute_nll(P: torch.Tensor) -> float:
            coef_dict = self.unwrap_coef_dict(P, type2idx)
            y_pred = self._forward(x_dict=x_dict,
                                   availability=availability,
                                   user_index=user_index,
                                   manual_coef_value_dict=coef_dict)
            # the reduction needs to be 'sum' to obtain NLL.
            loss = F.cross_entropy(y_pred, y, reduction='sum')
            return loss

        H = torch.autograd.functional.hessian(compute_nll, all_coefs)
        assert H.shape == (self.num_params, self.num_params)
        return H

    def compute_std(self, x_dict, availability, user_index, y) -> Dict[str, torch.Tensor]:
        """Computes

        Args:f
            See definitions in self.compute_hessian.

        Returns:
            Dict[str, torch.Tensor]: a dictoinary whose keys are the same as self.coef_dict.keys()
            the values are standard errors of coefficients in each coefficient group.
        """
        _, type2idx = self.flatten_coef_dict(self.coef_dict)
        H = self.compute_hessian(x_dict, availability, user_index, y)
        std_all = torch.sqrt(torch.diag(torch.inverse(H)))
        std_dict = dict()
        for var_type in type2idx.keys():
            # get std of variables belonging to each type.
            start, end = type2idx[var_type]
            std_dict[var_type] = std_all[start:end]
        return std_dict
