"""
The general class of learnable coefficients in various models, this class serves as the building blocks for models in this package.
The weights (i.e., learnable parameters) in the Coefficient class are implemented using PyTorch and can be trained
directly using optimizers from PyTorch.

NOTE: torch-choice package users don't interact with classes in this file directly, please use conditional_logit_model.py
and nested_logit_model.py instead.

Author: Tianyu Du
Update: Apr. 28, 2022
"""
from typing import Optional

import torch
import torch.nn as nn


class Coefficient(nn.Module):
    def __init__(self,
                 variation: str,
                 num_params: int,
                 num_items: Optional[int]=None,
                 num_users: Optional[int]=None
                 ) -> None:
        """A generic coefficient object storing trainable parameters. This class corresponds to those variables typically
        in Greek letters in the model's utility representation.

        Args:
            variation (str): the degree of variation of this coefficient. For example, the coefficient can vary by users or items.
                Currently, we support variations 'constant', 'item', 'item-full', 'user', 'user-item', 'user-item-full'.
                For detailed explanation of these variations, please refer to the documentation of ConditionalLogitModel.
            num_params (int): number of parameters in this coefficient. Note that this number is the number of parameters
                per class, not the total number of parameters. For example, suppose we have U users and you want to initiate
                an user-specific coefficient called `theta_user`. The coefficient enters the utility form while being multiplied
                with some K-dimension observables. Then, for each user, there are K parameters to be multiplied with the K-dimensional
                observable. However, the total number of parameters is K * U (K for each of U users). In this case, `num_params` should
                be set to `K`, NOT `K*U`.
            num_items (int): the number of items in the prediction problem, this is required to reshape the parameter correctly.
            num_users (Optional[int], optional): number of users, this is only necessary if the coefficient varies by users.
                Defaults to None.
        """
        super(Coefficient, self).__init__()
        self.variation = variation
        self.num_items = num_items
        self.num_users = num_users
        self.num_params = num_params

        # construct the trainable.
        if self.variation == 'constant':
            # constant for all users and items.
            self.coef = nn.Parameter(torch.randn(num_params), requires_grad=True)
        elif self.variation == 'item':
            # coef depends on item j but not on user i.
            # force coefficients for the first item class to be zero.
            self.coef = nn.Parameter(torch.zeros(num_items - 1, num_params), requires_grad=True)
        elif self.variation == 'item-full':
            # coef depends on item j but not on user i.
            # model coefficient for every item.
            self.coef = nn.Parameter(torch.zeros(num_items, num_params), requires_grad=True)
        elif self.variation == 'user':
            # coef depends on the user.
            # we always model coefficient for all users.
            self.coef = nn.Parameter(torch.zeros(num_users, num_params), requires_grad=True)
        elif self.variation == 'user-item':
            # coefficients of the first item is forced to be zero, model coefficients for N - 1 items only.
            self.coef = nn.Parameter(torch.zeros(num_users, num_items - 1, num_params), requires_grad=True)
        elif self.variation == 'user-item-full':
            # construct coefficients for every items.
            self.coef = nn.Parameter(torch.zeros(num_users, num_items, num_params), requires_grad=True)
        else:
            raise ValueError(f'Unsupported type of variation: {self.variation}.')

    def __repr__(self) -> str:
        """Returns a string representation of the coefficient.

        Returns:
            str: the string representation of the coefficient.
        """
        return f'Coefficient(variation={self.variation}, num_items={self.num_items},' \
               + f' num_users={self.num_users}, num_params={self.num_params},' \
               + f' {self.coef.numel()} trainable parameters in total).'

    def forward(self,
                x: torch.Tensor,
                user_index: Optional[torch.Tensor]=None,
                manual_coef_value: Optional[torch.Tensor]=None
                ) -> torch.Tensor:
        """
        The forward function of the coefficient, which computes the utility from purchasing each item in each session.
        The output shape will be (num_sessions, num_items).

        Args:
            x (torch.Tensor): a tensor of shape (num_sessions, num_items, num_params). Please note that the Coefficient
                class will NOT reshape input tensors itself, this reshaping needs to be done in the model class.
            user_index (Optional[torch.Tensor], optional): a tensor of shape (num_sessions,)
                contain IDs of the user involved in that session. If set to None, assume the same
                user is making all decisions.
                Defaults to None.
            manual_coef_value (Optional[torch.Tensor], optional): a tensor with the same number of
                entries as self.coef. If provided, the forward function uses provided values
                as coefficient and return the predicted utility, this feature is useful when
                the researcher wishes to manually specify values for coefficients and examine prediction
                with specified coefficient values. If not provided, forward function is executed
                using values from self.coef.
                Defaults to None.

        Returns:
            torch.Tensor: a tensor of shape (num_sessions, num_items) whose (t, i) entry represents
                the utility of purchasing item i in session t.
        """
        if manual_coef_value is not None:
            assert manual_coef_value.numel() == self.coef.numel()
            # plugin the provided coefficient values, coef is a tensor.
            coef = manual_coef_value.reshape(*self.coef.shape)
        else:
            # use the learned coefficient values, coef is a nn.Parameter.
            coef = self.coef

        num_trips, num_items, num_feats = x.shape
        assert self.num_params == num_feats

        # cast coefficient tensor to (num_trips, num_items, self.num_params).
        if self.variation == 'constant':
            coef = coef.view(1, 1, self.num_params).expand(num_trips, num_items, -1)

        elif self.variation == 'item':
            # coef has shape (num_items-1, num_params)
            # force coefficient for the first item to be zero.
            zeros = torch.zeros(1, self.num_params).to(coef.device)
            coef = torch.cat((zeros, coef), dim=0)  # (num_items, num_params)
            coef = coef.view(1, self.num_items, self.num_params).expand(num_trips, -1, -1)

        elif self.variation == 'item-full':
            # coef has shape (num_items, num_params)
            coef = coef.view(1, self.num_items, self.num_params).expand(num_trips, -1, -1)

        elif self.variation == 'user':
            # coef has shape (num_users, num_params)
            coef = coef[user_index, :]  # (num_trips, num_params) user-specific coefficients.
            coef = coef.view(num_trips, 1, self.num_params).expand(-1, num_items, -1)

        elif self.variation == 'user-item':
            # (num_trips,) long tensor of user ID.
            # originally, coef has shape (num_users, num_items-1, num_params)
            # transform to (num_trips, num_items - 1, num_params), user-specific.
            coef = coef[user_index, :, :]
            # coefs for the first item for all users are enforced to 0.
            zeros = torch.zeros(num_trips, 1, self.num_params).to(coef.device)
            coef = torch.cat((zeros, coef), dim=1)  # (num_trips, num_items, num_params)

        elif self.variation == 'user-item-full':
            # originally, coef has shape (num_users, num_items, num_params)
            coef = coef[user_index, :, :]  # (num_trips, num_items, num_params)

        else:
            raise ValueError(f'Unsupported type of variation: {self.variation}.')

        assert coef.shape == (num_trips, num_items, num_feats) == x.shape

        # compute the utility of each item in each trip, take summation along the feature dimension, the same as taking
        # the inner product.
        return (x * coef).sum(dim=-1)
