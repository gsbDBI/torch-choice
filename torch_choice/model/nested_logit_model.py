"""
Implementation of the nested logit model, see page 86 of the book
"discrete choice methods with simulation" by Train. for more details.
"""
import warnings
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from deepchoice.model.coefficient import Coefficient


class NestedLogitModel(nn.Module):
    # The nested logit model.
    def __init__(self,
                 category_to_item: Dict[object, List[int]],
                 category_coef_variation_dict: Dict[str, str],
                 category_num_param_dict: Dict[str, int],
                 item_coef_variation_dict: Dict[str, str],
                 item_num_param_dict: Dict[str, int],
                 num_users: Optional[int]=None,
                 shared_lambda: bool=False
                 ) -> None:
        """Initialization method of the nested logit model.

        Args:
            category_to_item (Dict[object, List[int]]): a dictionary maps a category ID to a list
                of items IDs of the queried category.

            category_coef_variation_dict (Dict[str, str]): a dictionary maps a variable type
                (i.e., variable group) to the level of variation for the coefficient of this type
                of variables.
            category_num_param_dict (Dict[str, int]): a dictoinary maps a variable type name to
                the number of parameters in this variable group.

            item_coef_variation_dict (Dict[str, str]): the same as category_coef_variation_dict but
                for item features.
            item_num_param_dict (Dict[str, int]): the same as category_num_param_dict but for item
                features.

            num_users (Optional[int], optional): number of users to be modelled, this is only
                required if any of variable type requires user-specific variations.
                Defaults to None.

            shared_lambda (bool): a boolean indicating whether to enforce the elasticity lambda, which
                is the coefficient for inclusive values, to be constant for all categories.
                The lambda enters the category-level selection as the following
                Utility of choosing category k = lambda * inclusive value of category k
                                               + linear combination of some other category level features
                If set to True, a single lambda will be learned for all categories, otherwise, the
                model learns an individual lambda for each category.
                Defaults to False.
        """
        super(NestedLogitModel, self).__init__()
        self.category_to_item = category_to_item
        self.category_coef_variation_dict = category_coef_variation_dict
        self.category_num_param_dict = category_num_param_dict
        self.item_coef_variation_dict = item_coef_variation_dict
        self.item_num_param_dict = item_num_param_dict
        self.num_users = num_users

        self.categories = list(category_to_item.keys())
        self.num_categories = len(self.categories)
        self.num_items = sum(len(items) for items in category_to_item.values())

        # category coefficients.
        self.category_coef_dict = self._build_coef_dict(self.category_coef_variation_dict,
                                                        self.category_num_param_dict,
                                                        self.num_categories)

        # item coefficients.
        self.item_coef_dict = self._build_coef_dict(self.item_coef_variation_dict,
                                                    self.item_num_param_dict,
                                                    self.num_items)

        self.shared_lambda = shared_lambda
        if self.shared_lambda:
            self.lambda_weight = nn.Parameter(torch.ones(1), requires_grad=True)
        else:
            self.lambda_weight = nn.Parameter(torch.ones(self.num_categories) / 2, requires_grad=True)
        # breakpoint()
        # self.iv_weights = nn.Parameter(torch.ones(1), requires_grad=True)
        # used to warn users if forgot to call clamp.
        self._clamp_called_flag = True

    @property
    def num_params(self) -> int:
        return sum(w.numel() for w in self.parameters())

    def _build_coef_dict(self,
                         coef_variation_dict: Dict[str, str],
                         num_param_dict: Dict[str, int],
                         num_items: int):
        # build coefficient dictionary, mapping variable groups to the corresponding Coefficient
        # Module. num_items could be the actual number of items or the number of categories.
        coef_dict = dict()
        for var_type, variation in coef_variation_dict.items():
            num_params = num_param_dict[var_type]
            coef_dict[var_type] = Coefficient(variation=variation,
                                              num_items=num_items,
                                              num_users=self.num_users,
                                              num_params=num_params)
        return nn.ModuleDict(coef_dict)

    def _check_input_shapes(self, category_x_dict, item_x_dict, user_index, item_availability) -> None:
        T = list(category_x_dict.values())[0].shape[0]  # batch size.
        for var_type, x_category in category_x_dict.items():
            x_item = item_x_dict[var_type]
            assert len(x_item.shape) == len(x_item.shape) == 3
            assert x_category.shape[0] == x_item.shape[0]
            assert x_category.shape == (T, self.num_categories, self.category_num_param_dict[var_type])
            assert x_item.shape == (T, self.num_items, self.item_num_param_dict[var_type])

        if (user_index is not None) and (self.num_users is not None):
            assert user_index.shape == (T,)

        if item_availability is not None:
            assert item_availability.shape == (T, self.num_items)

    def forward(self, batch):
        return self._forward(batch['category'].x_dict,
                             batch['item'].x_dict,
                             batch['item'].user_index,
                             batch['item'].item_availability)

    def _forward(self,
                 category_x_dict: Dict[str, torch.Tensor],
                 item_x_dict: Dict[str, torch.Tensor],
                 user_index: Optional[torch.LongTensor] = None,
                 item_availability: Optional[torch.BoolTensor] = None
                 ) -> torch.Tensor:
        """"Computes log P[t, i] = the log probability for the user involved in trip t to choose item i.
        Let n denote the ID of the user involved in trip t, then P[t, i] = P_{ni} on page 86 of the
        book "discrete choice methods with simulation" by Train.

        Args:
            x_category (torch.Tensor): a tensor with shape (num_trips, num_categories, *) including
                features of all categories in each trip.
            x_item (torch.Tensor): a tensor with shape (num_trips, num_items, *) including features
                of all items in each trip.
            user_index (torch.LongTensor): a tensor of shape (num_trips,) indicating which user is
                making decision in each trip. Setting user_index = None assumes the same user is
                making decisions in all trips.
            item_availability (torch.BoolTensor): a boolean tensor with shape (num_trips, num_items)
                indicating the aviliability of items in each trip. If item_availability[t, i] = False,
                the utility of choosing item i in trip t, V[t, i], will be set to -inf.
                Given the decomposition V[t, i] = W[t, k(i)] + Y[t, i] + eps, V[t, i] is set to -inf
                by setting Y[t, i] = -inf for unavilable items.

        Returns:
            torch.Tensor: a tensor of shape (num_trips, num_items) including the log probabilty
            of choosing item i in trip t.
        """
        if self.shared_lambda:
            self.lambdas = self.lambda_weight.expand(self.num_categories)
        else:
            self.lambdas = self.lambda_weight

        # if not self._clamp_called_flag:
        #     warnings.warn('Did you forget to call clamp_lambdas() after optimizer.step()?')

        # The overall utility of item can be decomposed into V[item] = W[category] + Y[item] + eps.
        T = list(item_x_dict.values())[0].shape[0]
        device = list(item_x_dict.values())[0].device
        # compute category-specific utility with shape (T, num_categories).
        W = torch.zeros(T, self.num_categories).to(device)

        if 'intercept' in self.category_coef_variation_dict.keys():
            category_x_dict['intercept'] = torch.ones((T, self.num_categories, 1)).to(device)

        for var_type, coef in self.category_coef_dict.items():
            W += coef(category_x_dict[var_type], user_index)

        # compute item-specific utility (T, num_items).
        Y = torch.zeros(T, self.num_items).to(device)
        for var_type, coef in self.item_coef_dict.items():
            Y += coef(item_x_dict[var_type], user_index)

        if item_availability is not None:
            Y[~item_availability] = -1.0e20

        # =============================================================================
        # compute the inclusive value of each category.
        inclusive_value = dict()
        for k, Bk in self.category_to_item.items():
            # for nest k, divide the Y of all items in Bk by lambda_k.
            Y[:, Bk] /= self.lambdas[k]
            # compute inclusive value for category k.
            # mask out unavilable items.
            inclusive_value[k] = torch.logsumexp(Y[:, Bk], dim=1, keepdim=False)  # (T,)
        # boardcast inclusive value from (T, num_categories) to (T, num_items).
        # for trip t, I[t, i] is the inclusive value of the category item i belongs to.
        I = torch.zeros(T, self.num_items).to(device)
        for k, Bk in self.category_to_item.items():
            I[:, Bk] = inclusive_value[k].view(-1, 1)  # (T, |Bk|)

        # logP_item[t, i] = log P(ni|Bk), where Bk is the category item i is in, n is the user in trip t.
        logP_item = Y - I  # (T, num_items)

        # =============================================================================
        # logP_categroy[t, i] = log P(Bk), for item i in trip t, the probability of choosing the nest/bucket
        # item i belongs to. logP_category has shape (T, num_items)
        # logit[t, i] = W[n, k] + lambda[k] I[n, k], where n is the user involved in trip t, k is
        # the category item i belongs to.
        logit = torch.zeros(T, self.num_items).to(device)
        for k, Bk in self.category_to_item.items():
            logit[:, Bk] = (W[:, k] + self.lambdas[k] * inclusive_value[k]).view(-1, 1)  # (T, |Bk|)
        # only count each category once in the logsumexp within the category level model.
        cols = [x[0] for x in self.category_to_item.values()]
        logP_category = logit - torch.logsumexp(logit[:, cols], dim=1, keepdim=True)

        # =============================================================================
        # compute the joint log P_{ni} as in the textbook.
        logP = logP_item + logP_category
        self._clamp_called_flag = False
        return logP

    def log_likelihood(self, *args):
        return - self.negative_log_likelihood(*args)

    def negative_log_likelihood(self,
                                batch,
                                y: torch.LongTensor,
                                is_train: bool=True) -> torch.Tensor:
        # compute the negative log-likelihood loss directly.
        if is_train:
            self.train()
        else:
            self.eval()
        # (num_trips, num_items)
        logP = self.forward(batch)
        nll = - logP[torch.arange(len(y)), y].sum()
        return nll

    def clamp_lambdas(self):
        """
        Restrict values of lambdas to 0 < lambda <= 1 to guarantee the utility maximization property
        of the model.
        This method should be called everytime after optimizer.step().
        We add a self_clamp_called_flag to remind researchers if this method is not called.
        """
        for k in range(len(self.lambdas)):
            self.lambdas[k] = torch.clamp(self.lambdas[k], 1e-5, 1)
        self._clam_called_flag = True

    @staticmethod
    def add_constant(x: torch.Tensor, where: str='prepend') -> torch.Tensor:
        """A helper function used to add constant to feature tensor,
        x has shape (batch_size, num_classes, num_parameters),
        returns a tensor of shape (*, num_parameters+1).
        """
        batch_size, num_classes, num_parameters = x.shape
        ones = torch.ones((batch_size, num_classes, 1))
        if where == 'prepend':
            new = torch.cat((ones, x), dim=-1)
        elif where == 'append':
            new = torch.cat((x, ones), dim=-1)
        else:
            raise Exception
        return new
