"""
Conditional Logit Model.

Author: Tianyu Du
Date: Aug. 8, 2021
Update: Apr. 28, 2022
"""
import warnings
from copy import deepcopy
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_choice.data.choice_dataset import ChoiceDataset
from torch_choice.model.coefficient import Coefficient
from torch_choice.model.formula_parser import parse_formula


class ConditionalLogitModel(nn.Module):
    """The more generalized version of conditional logit model, the model allows for research specific
    variable types(groups) and different levels of variations for coefficient.

    The model allows for the following levels for variable variations:
    NOTE: unless the `-full` flag is specified (which means we want to explicitly model coefficients
        for all items), for all variation levels related to item (item specific and user-item specific),
        the model force coefficients for the first item to be zero. This design follows standard
        econometric practice.

    - constant: constant over all users and items,

    - user: user-specific parameters but constant across all items,

    - item: item-specific parameters but constant across all users, parameters for the first item are
        forced to be zero.
    - item-full: item-specific parameters but constant across all users, explicitly model for all items.

    - user-item: parameters that are specific to both user and item, parameter for the first item
        for all users are forced to be zero.
    - user-item-full: parameters that are specific to both user and item, explicitly model for all items.
    """

    def __init__(self,
                 formula: Optional[str]=None,
                 dataset: Optional[ChoiceDataset]=None,
                 coef_variation_dict: Optional[Dict[str, str]]=None,
                 num_param_dict: Optional[Dict[str, int]]=None,
                 num_items: Optional[int]=None,
                 num_users: Optional[int]=None,
                 regularization: Optional[str]=None,
                 regularization_weight: Optional[float]=None
                 ) -> None:
        """
        Args:
            formula (str): a string representing the utility formula.
                The formula consists of '(variable_name|variation)'s separated by '+', for example:
                "(var1|item) + (var2|user) + (var3|constant)"
                where the first part of each term is the name of the variable
                and the second part is the variation of the coefficient.
                The variation can be one of the following:
                'constant', 'item', 'item-full', 'user', 'user-item', 'user-item-full'.
                All spaces in the formula will be ignored, hence please do not use spaces in variable/observable names.
            data (ChoiceDataset): a ChoiceDataset object for training the model, the parser will infer dimensions of variables
                and sizes of coefficients from the ChoiceDataset.
            coef_variation_dict (Dict[str, str]): variable type to variation level dictionary. Keys of this dictionary
                should be variable names in the dataset (i.e., these starting with `itemsession_`, `price_`, `user_`, etc), or `intercept`
                if the researcher requires an intercept term.
                For each variable name X_var (e.g., `user_income`) or `intercept`, the corresponding dictionary key should
                be one of the following values, this value specifies the "level of variation" of the coefficient.

                - `constant`: the coefficient constant over all users and items: $X \beta$.

                - `user`: user-specific parameters but constant across all items: $X \beta_{u}$.

                - `item`: item-specific parameters but constant across all users, $X \beta_{i}$.
                    Note that the coefficients for the first item are forced to be zero following the standard practice
                    in econometrics.

                - `item-full`: the same configuration as `item`, but does not force the coefficients of the first item to
                    be zeros.

                The following configurations are supported by the package, but we don't recommend using them due to the
                    large number of parameters.
                - `user-item`: parameters that are specific to both user and item, parameter for the first item
                    for all users are forced to be zero.

                - `user-item-full`: parameters that are specific to both user and item, explicitly model for all items.
            num_param_dict (Optional[Dict[str, int]]): variable type to number of parameters dictionary with keys exactly the same
                as the `coef_variation_dict`. Values of `num_param_dict` records numbers of features in each kind of variable.
                If None is supplied, num_param_dict will be a dictionary with the same keys as the `coef_variation_dict` dictionary
                and values of all ones. Default to be None.
            num_items (int): number of items in the dataset.
            num_users (int): number of users in the dataset.
            regularization (Optional[str]): this argument takes values from {'L1', 'L2', None}, which specifies the type of
                regularization added to the log-likelihood.
                - 'L1' will subtract regularization_weight * 1-norm of parameters from the log-likelihood.
                - 'L2' will subtract regularization_weight * 2-norm of parameters from the log-likelihood.
                - None does not modify the log-likelihood.
                Defaults to None.
            regularization_weight (Optional[float]): the weight of parameter norm subtracted from the log-likelihood.
                This term controls the strength of regularization. This argument is required if and only if regularization
                is not None.
                Defaults to None.
        """
        if coef_variation_dict is None and formula is None:
            raise ValueError("Either coef_variation_dict or formula should be provided to specify the model.")

        if (coef_variation_dict is not None) and (formula is not None):
            raise ValueError("Only one of coef_variation_dict or formula should be provided to specify the model.")

        if (formula is not None) and (dataset is None):
            raise ValueError("If formula is provided, data should be provided to specify the model.")

        # Use the formula to infer model, override dictionaries.
        if formula is not None:
            coef_variation_dict, num_param_dict = parse_formula(formula, dataset)

        super(ConditionalLogitModel, self).__init__()

        if num_param_dict is None:
            num_param_dict = {key:1 for key in coef_variation_dict.keys()}

        assert coef_variation_dict.keys() == num_param_dict.keys()

        self.variable_types = list(deepcopy(num_param_dict).keys())

        self.coef_variation_dict = deepcopy(coef_variation_dict)
        self.num_param_dict = deepcopy(num_param_dict)

        self.num_items = num_items
        self.num_users = num_users

        self.regularization = regularization
        assert self.regularization in ['L1', 'L2', None], f"Provided regularization={self.regularization} is not allowed, allowed values are ['L1', 'L2', None]."
        self.regularization_weight = regularization_weight
        if (self.regularization is not None) and (self.regularization_weight is None):
            raise ValueError(f'You specified regularization type {self.regularization} without providing regularization_weight.')
        if (self.regularization is None) and (self.regularization_weight is not None):
            raise ValueError(f'You specified no regularization but you provide regularization_weight={self.regularization_weight}, you should leave regularization_weight as None if you do not want to regularize the model.')

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
        # A ModuleDict is required to properly register all trainable parameters.
        # self.parameter() will fail if a python dictionary is used instead.
        self.coef_dict = nn.ModuleDict(coef_dict)

    def __repr__(self) -> str:
        """Return a string representation of the model.

        Returns:
            str: the string representation of the model.
        """
        out_str_lst = ['Conditional logistic discrete choice model, expects input features:\n']
        for var_type, num_params in self.num_param_dict.items():
            out_str_lst.append(f'X[{var_type}] with {num_params} parameters, with {self.coef_variation_dict[var_type]} level variation.')
        return super().__repr__() + '\n' + '\n'.join(out_str_lst) + '\n' + f'device={self.device}'

    @property
    def num_params(self) -> int:
        """Get the total number of parameters. For example, if there is only an user-specific coefficient to be multiplied
        with the K-dimensional observable, then the total number of parameters would be K x number of users, assuming no
        intercept is involved.

        Returns:
            int: the total number of learnable parameters.
        """
        return sum(w.numel() for w in self.parameters())

    def summary(self):
        """Print out the current model parameter."""
        for var_type, coefficient in self.coef_dict.items():
            if coefficient is not None:
                print('Variable Type: ', var_type)
                print(coefficient.coef)

    def forward(self,
                batch: ChoiceDataset,
                manual_coef_value_dict: Optional[Dict[str, torch.Tensor]] = None
                ) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            batch: a `ChoiceDataset` object.

            manual_coef_value_dict (Optional[Dict[str, torch.Tensor]], optional): a dictionary with
                keys in {'u', 'i'} etc and tensors as values. If provided, the model will force
                coefficient to be the provided values and compute utility conditioned on the provided
                coefficient values. This feature is useful when the research wishes to plug in particular
                values of coefficients and examine the utility values. If not provided, the model will
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
            total_utility[~batch.item_availability[batch.session_index, :]] = torch.finfo(total_utility.dtype).min / 2
        return total_utility


    def negative_log_likelihood(self, batch: ChoiceDataset, y: torch.Tensor, is_train: bool=True) -> torch.Tensor:
        """Computes the log-likelihood for the batch and label.
        TODO: consider remove y, change to label.
        TODO: consider move this method outside the model, the role of the model is to compute the utility.

        Args:
            batch (ChoiceDataset): a ChoiceDataset object containing the data.
            y (torch.Tensor): the label.
            is_train (bool, optional): whether to trace the gradient. Defaults to True.

        Returns:
            torch.Tensor: the negative log-likelihood.
        """
        if is_train:
            self.train()
        else:
            self.eval()
        # (num_trips, num_items)
        total_utility = self.forward(batch)
        logP = torch.log_softmax(total_utility, dim=1)
        nll = - logP[torch.arange(len(y)), y].sum()
        return nll

    def loss(self, *args, **kwargs):
        """The loss function to be optimized. This is a wrapper of `negative_log_likelihood` + regularization loss if required."""
        nll = self.negative_log_likelihood(*args, **kwargs)
        if self.regularization is not None:
            L = {'L1': 1, 'L2': 2}[self.regularization]
            for param in self.parameters():
                nll += self.regularization_weight * torch.norm(param, p=L)
        return nll

    @property
    def device(self) -> torch.device:
        """Returns the device of the coefficient.

        Returns:
            torch.device: the device of the model.
        """
        return next(iter(self.coef_dict.values())).device

    # NOTE: the method for computing Hessian and standard deviation has been moved to std.py.
    # @staticmethod
    # def flatten_coef_dict(coef_dict: Dict[str, Union[torch.Tensor, torch.nn.Parameter]]) -> Tuple[torch.Tensor, dict]:
    #     """Flattens the coef_dict into a 1-dimension tensor, used for hessian computation.

    #     Args:
    #         coef_dict (Dict[str, Union[torch.Tensor, torch.nn.Parameter]]): a dictionary holding learnable parameters.

    #     Returns:
    #         Tuple[torch.Tensor, dict]: 1. the flattened tensors with shape (num_params,), 2. an indexing dictionary
    #             used for reconstructing the original coef_dict from the flatten tensor.
    #     """
    #     type2idx = dict()
    #     param_list = list()
    #     start = 0

    #     for var_type in coef_dict.keys():
    #         num_params = coef_dict[var_type].coef.numel()
    #         # track which portion of all_param tensor belongs to this variable type.
    #         type2idx[var_type] = (start, start + num_params)
    #         start += num_params
    #         # use reshape instead of view to make a copy.
    #         param_list.append(coef_dict[var_type].coef.clone().reshape(-1,))

    #     all_param = torch.cat(param_list)  # (self.num_params(), )
    #     return all_param, type2idx

    # @staticmethod
    # def unwrap_coef_dict(param: torch.Tensor, type2idx: Dict[str, Tuple[int, int]]) -> Dict[str, torch.Tensor]:
    #     """Rebuilds coef_dict from output of self.flatten_coef_dict method.

    #     Args:
    #         param (torch.Tensor): the flattened coef_dict from self.flatten_coef_dict.
    #         type2idx (Dict[str, Tuple[int, int]]): the indexing dictionary from self.flatten_coef_dict.

    #     Returns:
    #         Dict[str, torch.Tensor]: the re-constructed coefficient dictionary.
    #     """
    #     coef_dict = dict()
    #     for var_type in type2idx.keys():
    #         start, end = type2idx[var_type]
    #         # no need to reshape here, Coefficient handles it.
    #         coef_dict[var_type] = param[start:end]
    #     return coef_dict

    # def compute_hessian(self, x_dict, availability, user_index, y) -> torch.Tensor:
    #     """Computes the Hessian of negative log-likelihood (total cross-entropy loss) with respect
    #     to all parameters in this model. The Hessian can be later used for constructing the standard deviation of
    #     parameters.

    #     Args:
    #         x_dict ,availability, user_index: see definitions in self.forward method.
    #         y (torch.LongTensor): a tensor with shape (num_trips,) of IDs of items actually purchased.

    #     Returns:
    #         torch.Tensor: a (self.num_params, self.num_params) tensor of the Hessian matrix.
    #     """
    #     all_coefs, type2idx = self.flatten_coef_dict(self.coef_dict)

    #     def compute_nll(P: torch.Tensor) -> float:
    #         coef_dict = self.unwrap_coef_dict(P, type2idx)
    #         y_pred = self._forward(x_dict=x_dict,
    #                                availability=availability,
    #                                user_index=user_index,
    #                                manual_coef_value_dict=coef_dict)
    #         # the reduction needs to be 'sum' to obtain NLL.
    #         loss = F.cross_entropy(y_pred, y, reduction='sum')
    #         return loss

    #     H = torch.autograd.functional.hessian(compute_nll, all_coefs)
    #     assert H.shape == (self.num_params, self.num_params)
    #     return H

    # def compute_std(self, x_dict, availability, user_index, y) -> Dict[str, torch.Tensor]:
    #     """Computes

    #     Args:f
    #         See definitions in self.compute_hessian.

    #     Returns:
    #         Dict[str, torch.Tensor]: a dictionary whose keys are the same as self.coef_dict.keys()
    #         the values are standard errors of coefficients in each coefficient group.
    #     """
    #     _, type2idx = self.flatten_coef_dict(self.coef_dict)
    #     H = self.compute_hessian(x_dict, availability, user_index, y)
    #     std_all = torch.sqrt(torch.diag(torch.inverse(H)))
    #     std_dict = dict()
    #     for var_type in type2idx.keys():
    #         # get std of variables belonging to each type.
    #         start, end = type2idx[var_type]
    #         std_dict[var_type] = std_all[start:end]
    #     return std_dict
