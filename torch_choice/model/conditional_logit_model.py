"""
Conditional Logit Model.

Author: Tianyu Du
Update: Apr. 10, 2023
"""
import warnings
from copy import deepcopy
from typing import Dict, Optional, Union

import torch
import torch.nn as nn

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
                 regularization_weight: Optional[float]=None,
                 weight_initialization: Optional[Union[str, Dict[str, str]]]=None,
                 model_outside_option: Optional[bool]=False
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
            weight_initialization (Optional[Union[str, Dict[str, str]]]): controls for how coefficients are initialized;
                users can pass a string from {'normal', 'uniform', 'zero'} to initialize all coefficients in the same way.
                Alternatively, users can pass a dictionary with keys exactly the same as the `coef_variation_dict` dictionary,
                and values from {'normal', 'uniform', 'zero'} to initialize coefficients of different types of variables differently.
                By default, all coefficients are initialized following a standard normal distribution.
            model_outside_option (Optional[bool]): whether to explicitly model the outside option (i.e., the consumer did not buy anything).
                To enable modeling outside option, the outside option is indicated by `item_index[n] == -1` in the item-index-tensor.
                In this case, the item-index-tensor can contain values in `{-1, 0, 1, ..., num_items-1}`.
                Otherwise, if the outside option is not modelled, the item-index-tensor should only contain values in `{0, 1, ..., num_items-1}`.
                The utility of the outside option is always set to 0 while computing the probability.
                By default, model_outside_option is set to False and the model does not model the outside option.
        """
        # ==============================================================================================================
        # Check that the model received a valid combination of inputs so that it can be initialized.
        # ==============================================================================================================
        if coef_variation_dict is None and formula is None:
            raise ValueError("Either coef_variation_dict or formula should be provided to specify the model.")

        if (coef_variation_dict is not None) and (formula is not None):
            raise ValueError("Only one of coef_variation_dict or formula should be provided to specify the model.")

        if (formula is not None) and (dataset is None):
            raise ValueError("If formula is provided, data should be provided to specify the model.")


        # ==============================================================================================================
        # Build necessary dictionaries for model initialization.
        # ==============================================================================================================
        if formula is None:
            # Use dictionaries to initialize the model.
            if num_param_dict is None:
                warnings.warn("`num_param_dict` is not provided, all variables will be treated as having one parameter.")
                num_param_dict = {key:1 for key in coef_variation_dict.keys()}

            assert coef_variation_dict.keys() == num_param_dict.keys()

            # variable `var` with variation `spec` to variable `var[spec]`.
            rename = dict()  # old variable name --> new variable name.
            for variable, specificity in coef_variation_dict.items():
                rename[variable] = f"{variable}[{specificity}]"

            for old_name, new_name in rename.items():
                coef_variation_dict[new_name] = coef_variation_dict.pop(old_name)
                num_param_dict[new_name] = num_param_dict.pop(old_name)
        else:
            # Use the formula to infer model.
            coef_variation_dict, num_param_dict = parse_formula(formula, dataset)

        # ==============================================================================================================
        # Model Initialization.
        # ==============================================================================================================
        super(ConditionalLogitModel, self).__init__()

        self.coef_variation_dict = deepcopy(coef_variation_dict)
        self.num_param_dict = deepcopy(num_param_dict)

        self.num_items = num_items
        self.num_users = num_users

        self.regularization = deepcopy(regularization)
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
        for variable in self.coef_variation_dict.keys():
            if self.is_intercept_term(variable) and variable not in self.num_param_dict.keys():
                warnings.warn(f"`{variable}` key found in coef_variation_dict but not in num_param_dict, num_param_dict['{variable}'] has been set to 1.")
                self.num_param_dict[variable] = 1

        # inform coefficients their ways of being initialized.
        self.weight_initialization = deepcopy(weight_initialization)

        # construct trainable parameters.
        coef_dict = dict()
        for var_type, variation in self.coef_variation_dict.items():
            if isinstance(self.weight_initialization, dict):
                if var_type.split('[')[0] in self.weight_initialization.keys():
                    # use the variable-specific initialization if provided.
                    init = self.weight_initialization[var_type.split('[')[0]]
                else:
                    # use default initialization.
                    init = None
            else:
                # initialize all coefficients in the same way.
                init = self.weight_initialization

            coef_dict[var_type] = Coefficient(variation=variation,
                                              num_items=self.num_items,
                                              num_users=self.num_users,
                                              num_params=self.num_param_dict[var_type],
                                              init=init)
        # A ModuleDict is required to properly register all trainable parameters.
        # self.parameter() will fail if a python dictionary is used instead.
        self.coef_dict = nn.ModuleDict(coef_dict)
        self.model_outside_option = model_outside_option

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

        for variable in self.coef_variation_dict.keys():
            if self.is_intercept_term(variable):
                # intercept term has no input tensor from the ChoiceDataset data structure.
                # the tensor for intercept has only 1 feature, every entry is 1.
                x_dict['intercept'] = torch.ones((len(batch), self.num_items, 1), device=batch.device)
                break

        # compute the utility from each item in each choice session.
        total_utility = torch.zeros((len(batch), self.num_items), device=batch.device)
        # for each type of variables, apply the corresponding coefficient to input x.

        for var_type, coef in self.coef_dict.items():
            # variable type is named as "observable_name[variation]", retrieve the corresponding observable name.
            corresponding_observable = var_type.split("[")[0]
            total_utility += coef(
                x_dict[corresponding_observable],
                batch.user_index,
                manual_coef_value=None if manual_coef_value_dict is None else manual_coef_value_dict[var_type])

        assert total_utility.shape == (len(batch), self.num_items)

        if batch.item_availability is not None:
            # mask out unavailable items.
            total_utility[~batch.item_availability[batch.session_index, :]] = torch.finfo(total_utility.dtype).min / 2

        # accommodate the outside option.
        if self.model_outside_option:
            # the outside option has zero utility.
            util_zero = torch.zeros(total_utility.size(0), 1, device=batch.device)  # (len(batch), 1)  zero tensor.
            # outside option is indicated by item_index == -1, we put it at the end.
            total_utility = torch.cat((total_utility, util_zero), dim=1)  # (len(batch), num_items+1)
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
        # check shapes.
        if self.model_outside_option:
            assert total_utility.shape == (len(batch), self.num_items+1)
            assert torch.all(total_utility[:, -1] == 0), "The last column of total_utility should be all zeros, which corresponds to the outside option."
        else:
            assert total_utility.shape == (len(batch), self.num_items)
        logP = torch.log_softmax(total_utility, dim=1)
        # since y == -1 indicates the outside option and the last column of total_utility is the outside option, the following
        # indexing should correctly retrieve the log-likelihood even for outside options.
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

    @staticmethod
    def is_intercept_term(variable: str):
        # check if the given variable is an intercept (fixed effect) term.
        # intercept (fixed effect) terms are defined as 'intercept[*]' and looks like 'intercept[user]', 'intercept[item]', etc.
        return (variable.startswith('intercept[') and variable.endswith(']'))

    def get_coefficient(self, variable: str) -> torch.Tensor:
        """Retrieve the coefficient tensor for the given variable.

        Args:
            variable (str): the variable name.

        Returns:
            torch.Tensor: the corresponding coefficient tensor of the requested variable.
        """
        return self.state_dict()[f"coef_dict.{variable}.coef"].detach().clone()
