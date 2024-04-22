"""
Implementation of the nested logit model, see page 86 of the book
"discrete choice methods with simulation" by Train. for more details.

Author: Tianyu Du
Update; Apr. 7, 2023
"""
from copy import deepcopy
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

from torch_choice.data.choice_dataset import ChoiceDataset
from torch_choice.data.joint_dataset import JointDataset
from torch_choice.model.coefficient import Coefficient
from torch_choice.model.formula_parser import parse_formula


class NestedLogitModel(nn.Module):
    def __init__(self,
                 nest_to_item: Dict[object, List[int]],
                 # method 1: specify variation and num param. dictionary.
                 nest_coef_variation_dict: Optional[Dict[str, str]]=None,
                 nest_num_param_dict: Optional[Dict[str, int]]=None,
                 item_coef_variation_dict: Optional[Dict[str, str]]=None,
                 item_num_param_dict: Optional[Dict[str, int]]=None,
                 # method 2: specify formula and dataset.
                 item_formula: Optional[str]=None,
                 nest_formula: Optional[str]=None,
                 dataset: Optional[JointDataset]=None,
                 num_users: Optional[int]=None,
                 shared_lambda: bool=False,
                 regularization: Optional[str]=None,
                 regularization_weight: Optional[float]=None,
                 nest_weight_initialization: Optional[Union[str, Dict[str, str]]]=None,
                 item_weight_initialization: Optional[Union[str, Dict[str, str]]]=None,
                 model_outside_option: Optional[bool]=False
                 ) -> None:
        """Initialization method of the nested logit model.

        Args:
            nest_to_item (Dict[object, List[int]]): a dictionary maps a nest ID to a list
                of items IDs of the queried nest.

            nest_coef_variation_dict (Dict[str, str]): a dictionary maps a variable type
                (i.e., variable group) to the level of variation for the coefficient of this type
                of variables.
            nest_num_param_dict (Dict[str, int]): a dictionary maps a variable type name to
                the number of parameters in this variable group.

            item_coef_variation_dict (Dict[str, str]): the same as nest_coef_variation_dict but
                for item features.
            item_num_param_dict (Dict[str, int]): the same as nest_num_param_dict but for item
                features.

            {nest, item}_formula (str): a string representing the utility formula for the {nest, item} level logit model.
                The formula consists of '(variable_name|variation)'s separated by '+', for example:
                "(var1|item) + (var2|user) + (var3|constant)"
                where the first part of each term is the name of the variable
                and the second part is the variation of the coefficient.
                The variation can be one of the following:
                'constant', 'item', 'item-full', 'user', 'user-item', 'user-item-full'.
                All spaces in the formula will be ignored, hence please do not use spaces in variable/observable names.
            dataset (JointDataset): a JointDataset object for training the model, the parser will infer dimensions of variables
                and sizes of coefficients for the nest level model from dataset.datasets['nest']. The parser will infer dimensions of variables and sizes of coefficients for the item level model from dataset.datasets['item'].

            num_users (Optional[int], optional): number of users to be modelled, this is only
                required if any of variable type requires user-specific variations.
                Defaults to None.

            shared_lambda (bool): a boolean indicating whether to enforce the elasticity lambda, which
                is the coefficient for inclusive values, to be constant for all nests.
                The lambda enters the nest-level selection as the following
                Utility of choosing nest k = lambda * inclusive value of nest k
                                               + linear combination of some other nest level features
                If set to True, a single lambda will be learned for all nests, otherwise, the
                model learns an individual lambda for each nest.
                Defaults to False.

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

            {nest, item}_weight_initialization (Optional[Union[str, Dict[str, str]]]): methods to initialize the weights of
                coefficients for {nest, item} level model. Please refer to the `weight_initialization` keyword in ConditionalLogitModel's documentation for more details.

            model_outside_option (Optional[bool]): whether to explicitly model the outside option (i.e., the consumer did not buy anything).
                To enable modeling outside option, the outside option is indicated by `item_index[n] == -1` in the item-index-tensor.
                In this case, the item-index-tensor can contain values in `{-1, 0, 1, ..., num_items-1}`.
                Otherwise, if the outside option is not modelled, the item-index-tensor should only contain values in `{0, 1, ..., num_items-1}`.
                The utility of the outside option is always set to 0 while computing the probability.
                By default, model_outside_option is set to False and the model does not model the outside option.
        """
        # handle nest level model.
        using_formula_to_initiate = (item_formula is not None) and (nest_formula is not None)
        if using_formula_to_initiate:
            # make sure that the research does not specify duplicated information, which might cause conflict.
            if (nest_coef_variation_dict is not None) or (item_coef_variation_dict is not None):
                raise ValueError('You specify the {item, nest}_formula to initiate the model, you should not specify the {item, nest}_coef_variation_dict at the same time.')
            if (nest_num_param_dict is not None) or (item_num_param_dict is not None):
                raise ValueError('You specify the {item, nest}_formula to initiate the model, you should not specify the {item, nest}_num_param_dict at the same time.')
            if dataset is None:
                raise ValueError('Dataset is required if {item, nest}_formula is specified to initiate the model.')

            nest_coef_variation_dict, nest_num_param_dict = parse_formula(nest_formula, dataset.datasets['nest'])
            item_coef_variation_dict, item_num_param_dict = parse_formula(item_formula, dataset.datasets['item'])

        else:
            # check for conflicting information.
            if (nest_formula is not None) or (item_formula is not None):
                raise ValueError('You should not specify {item, nest}_formula and {item, nest}_coef_variation_dict at the same time.')
            # make sure that the research specifies all the required information.
            if (nest_coef_variation_dict is None) or (item_coef_variation_dict is None):
                raise ValueError('You should specify the {item, nest}_coef_variation_dict to initiate the model.')
            if (nest_num_param_dict is None) or (item_num_param_dict is None):
                raise ValueError('You should specify the {item, nest}_num_param_dict to initiate the model.')

        super(NestedLogitModel, self).__init__()
        self.nest_to_item = nest_to_item
        self.nest_coef_variation_dict = nest_coef_variation_dict
        self.nest_num_param_dict = nest_num_param_dict
        self.item_coef_variation_dict = item_coef_variation_dict
        self.item_num_param_dict = item_num_param_dict
        self.num_users = num_users

        self.nests = list(nest_to_item.keys())
        self.num_nests = len(self.nests)
        self.num_items = sum(len(items) for items in nest_to_item.values())

        # nest coefficients.
        self.nest_coef_dict = self._build_coef_dict(self.nest_coef_variation_dict,
                                                    self.nest_num_param_dict,
                                                    self.num_nests,
                                                    weight_initialization=deepcopy(nest_weight_initialization))

        # item coefficients.
        self.item_coef_dict = self._build_coef_dict(self.item_coef_variation_dict,
                                                    self.item_num_param_dict,
                                                    self.num_items,
                                                    weight_initialization=deepcopy(item_weight_initialization))

        self.shared_lambda = shared_lambda
        if self.shared_lambda:
            self.lambda_weight = nn.Parameter(torch.ones(1), requires_grad=True)
        else:
            self.lambda_weight = nn.Parameter(torch.ones(self.num_nests) / 2, requires_grad=True)
        # breakpoint()
        # self.iv_weights = nn.Parameter(torch.ones(1), requires_grad=True)
        # used to warn users if forgot to call clamp.
        self._clamp_called_flag = True

        self.regularization = regularization
        assert self.regularization in ['L1', 'L2', None], f"Provided regularization={self.regularization} is not allowed, allowed values are ['L1', 'L2', None]."
        self.regularization_weight = regularization_weight
        if (self.regularization is not None) and (self.regularization_weight is None):
            raise ValueError(f'You specified regularization type {self.regularization} without providing regularization_weight.')
        if (self.regularization is None) and (self.regularization_weight is not None):
            raise ValueError(f'You specified no regularization but you provide regularization_weight={self.regularization_weight}, you should leave regularization_weight as None if you do not want to regularize the model.')

        self.model_outside_option = model_outside_option

    @property
    def num_params(self) -> int:
        """Get the total number of parameters. For example, if there is only an user-specific coefficient to be multiplied
        with the K-dimensional observable, then the total number of parameters would be K x number of users, assuming no
        intercept is involved.

        Returns:
            int: the total number of learnable parameters.
        """
        return sum(w.numel() for w in self.parameters())

    def _build_coef_dict(self,
                         coef_variation_dict: Dict[str, str],
                         num_param_dict: Dict[str, int],
                         num_items: int,
                         weight_initialization: Optional[Union[str, Dict[str, str]]]=None
                         ) -> nn.ModuleDict:
        """Builds a coefficient dictionary containing all trainable components of the model, mapping coefficient names
            to the corresponding Coefficient Module.
            num_items could be the actual number of items or the number of nests depends on the use case.
            NOTE: torch-choice users don't directly interact with this method.

        Args:
            coef_variation_dict (Dict[str, str]): a dictionary mapping coefficient names (e.g., theta_user) to the level
                of variation (e.g., 'user').
            num_param_dict (Dict[str, int]): a dictionary mapping coefficient names to the number of parameters in this
                coefficient. Be aware that, for example, if there is one K-dimensional coefficient for every user, then
                the `num_param` should be K instead of K x number of users.
            num_items (int): the total number of items in the prediction problem. `num_items` should be the number of nests if _build_coef_dict() is used for nest-level prediction.

        Returns:
            nn.ModuleDict: a PyTorch ModuleDict object mapping from coefficient names to training Coefficient.
        """
        coef_dict = dict()
        for var_type, variation in coef_variation_dict.items():
            num_params = num_param_dict[var_type]

            if isinstance(weight_initialization, dict):
                if var_type.split('[')[0] in weight_initialization.keys():
                    # use the variable-specific initialization if provided.
                    init = weight_initialization[var_type.split('[')[0]]
                else:
                    # use default initialization.
                    init = None
            else:
                # initialize all coefficients in the same way.
                init = weight_initialization

            coef_dict[var_type] = Coefficient(variation=variation,
                                              num_items=num_items,
                                              num_users=self.num_users,
                                              num_params=num_params,
                                              init=init)
        return nn.ModuleDict(coef_dict)


    def forward(self, batch: ChoiceDataset) -> torch.Tensor:
        """An standard forward method for the model, the user feeds a ChoiceDataset batch and the model returns the
            predicted log-likelihood tensor. The main forward passing happens in the _forward() method, but we provide
            this wrapper forward() method for a cleaner API, as forward() only requires a single batch argument.
            For more details about the forward passing, please refer to the _forward() method.

        # TODO: the ConditionalLogitModel returns predicted utility, the NestedLogitModel behaves the same?

        Args:
            batch (ChoiceDataset): a ChoiceDataset object containing the data batch.

        Returns:
            torch.Tensor: a tensor of shape (num_trips, num_items) including the log probability
            of choosing item i in trip t.
        """
        return self._forward(batch['nest'].x_dict,
                             batch['item'].x_dict,
                             batch['item'].user_index,
                             batch['item'].item_availability)

    def _forward(self,
                 nest_x_dict: Dict[str, torch.Tensor],
                 item_x_dict: Dict[str, torch.Tensor],
                 user_index: Optional[torch.LongTensor] = None,
                 item_availability: Optional[torch.BoolTensor] = None
                 ) -> torch.Tensor:
        """"Computes log P[t, i] = the log probability for the user involved in trip t to choose item i.
        Let n denote the ID of the user involved in trip t, then P[t, i] = P_{ni} on page 86 of the
        book "discrete choice methods with simulation" by Train.

        The `_forward` method is an internal API, users should refer to the `forward` method.

        Args:
            nest_x_dict (torch.Tensor): a dictionary mapping from nest-level feature names to the corresponding feature tensor.

            item_x_dict (torch.Tensor): a dictionary mapping from item-level feature names to the corresponding feature tensor.

                More details on the shape of the tensors can be found in the docstring of the `x_dict` method of `ChoiceDataset`.

            user_index (torch.LongTensor): a tensor of shape (num_trips,) indicating which user is
                making decision in each trip. Setting user_index = None assumes the same user is
                making decisions in all trips.
            item_availability (torch.BoolTensor): a boolean tensor with shape (num_trips, num_items)
                indicating the aviliability of items in each trip. If item_availability[t, i] = False,
                the utility of choosing item i in trip t, V[t, i], will be set to -inf.
                Given the decomposition V[t, i] = W[t, k(i)] + Y[t, i] + eps, V[t, i] is set to -inf
                by setting Y[t, i] = -inf for unavilable items.

        Returns:
            torch.Tensor: a tensor of shape (num_trips, num_items) including the log probability
            of choosing item i in trip t.
        """
        if self.shared_lambda:
            self.lambdas = self.lambda_weight.expand(self.num_nests)
        else:
            self.lambdas = self.lambda_weight

        # if not self._clamp_called_flag:
        #     warnings.warn('Did you forget to call clamp_lambdas() after optimizer.step()?')

        # The overall utility of item can be decomposed into V[item] = W[nest] + Y[item] + eps.
        T = list(item_x_dict.values())[0].shape[0]
        device = list(item_x_dict.values())[0].device
        # compute nest-specific utility with shape (T, num_nests).
        W = torch.zeros(T, self.num_nests).to(device)

        for variable in self.nest_coef_variation_dict.keys():
            if self.is_intercept_term(variable):
                nest_x_dict['intercept'] = torch.ones((T, self.num_nests, 1)).to(device)
                break

        for variable in self.item_coef_variation_dict.keys():
            if self.is_intercept_term(variable):
                item_x_dict['intercept'] = torch.ones((T, self.num_items, 1)).to(device)
                break

        for var_type, coef in self.nest_coef_dict.items():
            corresponding_observable = var_type.split("[")[0]
            W += coef(nest_x_dict[corresponding_observable], user_index)

        # compute item-specific utility (T, num_items).
        Y = torch.zeros(T, self.num_items).to(device)
        for var_type, coef in self.item_coef_dict.items():
            corresponding_observable = var_type.split("[")[0]
            Y += coef(item_x_dict[corresponding_observable], user_index)

        if item_availability is not None:
            Y[~item_availability] = torch.finfo(Y.dtype).min / 2

        # =============================================================================
        # compute the inclusive value of each nest.
        inclusive_value = dict()
        for k, Bk in self.nest_to_item.items():
            # for nest k, divide the Y of all items in Bk by lambda_k.
            Y[:, Bk] /= self.lambdas[k]
            # compute inclusive value for nest k.
            # mask out unavilable items.
            inclusive_value[k] = torch.logsumexp(Y[:, Bk], dim=1, keepdim=False)  # (T,)
        # boardcast inclusive value from (T, num_nests) to (T, num_items).
        # for trip t, I[t, i] is the inclusive value of the nest item i belongs to.
        I = torch.zeros(T, self.num_items).to(device)
        for k, Bk in self.nest_to_item.items():
            I[:, Bk] = inclusive_value[k].view(-1, 1)  # (T, |Bk|)

        # logP_item[t, i] = log P(ni|Bk), where Bk is the nest item i is in, n is the user in trip t.
        logP_item = Y - I  # (T, num_items)

        if self.model_outside_option:
            # if the model explicitly models the outside option, we need to add a column of zeros to logP_item.
            # log P(ni|Bk) = 0 for the outside option since Y = 0 and the outside option has its own nest.
            logP_item = torch.cat((logP_item, torch.zeros(T, 1).to(device)), dim=1)
            assert logP_item.shape == (T, self.num_items+1)
            assert torch.all(logP_item[:, -1] == 0)

        # =============================================================================
        # logP_nest[t, i] = log P(Bk), for item i in trip t, the probability of choosing the nest/bucket
        # item i belongs to. logP_nest has shape (T, num_items)
        # logit[t, i] = W[n, k] + lambda[k] I[n, k], where n is the user involved in trip t, k is
        # the nest item i belongs to.
        logit = torch.zeros(T, self.num_items).to(device)
        for k, Bk in self.nest_to_item.items():
            logit[:, Bk] = (W[:, k] + self.lambdas[k] * inclusive_value[k]).view(-1, 1)  # (T, |Bk|)
        # only count each nest once in the logsumexp within the nest level model.
        cols = [x[0] for x in self.nest_to_item.values()]
        if self.model_outside_option:
            # the last column corresponds to the outside option, which has W+lambda*I = 0 since W = I = Y = 0 for the outside option.
            logit = torch.cat((logit, torch.zeros(T, 1).to(device)), dim=1)
            assert logit.shape == (T, self.num_items+1)
            # we have already added W+lambda*I for each "actual" nest, now we add the "fake" nest for the outside option.
            cols.append(-1)
        logP_nest = logit - torch.logsumexp(logit[:, cols], dim=1, keepdim=True)

        # =============================================================================
        # compute the joint log P_{ni} as in the textbook.
        logP = logP_item + logP_nest
        self._clamp_called_flag = False
        return logP

    def log_likelihood(self, *args):
        """Computes the log likelihood of the model, please refer to the negative_log_likelihood() method.

        Returns:
            _type_: the log likelihood of the model.
        """
        return - self.negative_log_likelihood(*args)

    def negative_log_likelihood(self,
                                batch: ChoiceDataset,
                                y: torch.LongTensor,
                                is_train: bool=True) -> torch.scalar_tensor:
        """Computes the negative log likelihood of the model. Please note the log-likelihood is summed over all samples
            in batch instead of the average.

        Args:
            batch (ChoiceDataset): the ChoiceDataset object containing the data.
            y (torch.LongTensor): the label.
            is_train (bool, optional): which mode of the model to be used for the forward passing, if we need Hessian
                of the NLL through auto-grad, `is_train` should be set to True. If we merely need a performance metric,
                then `is_train` can be set to False for better performance.
                Defaults to True.

        Returns:
            torch.scalar_tensor: the negative log likelihood of the model.
        """
        # compute the negative log-likelihood loss directly.
        if is_train:
            self.train()
        else:
            self.eval()
        # (num_trips, num_items)
        logP = self.forward(batch)
        # check shapes
        if self.model_outside_option:
            assert logP.shape == (len(batch['item']), self.num_items+1)
        else:
            assert logP.shape == (len(batch['item']), self.num_items)
        # since y == -1 indicates the outside option and the last column of total_utility is the outside option, the following
        # indexing should correctly retrieve the log-likelihood even for outside options.
        nll = - logP[torch.arange(len(y)), y].sum()
        return nll

    def loss(self, *args, **kwargs):
        """The loss function to be optimized. This is a wrapper of `negative_log_likelihood` + regularization loss if required."""
        nll = self.negative_log_likelihood(*args, **kwargs)
        if self.regularization is not None:
            L = {'L1': 1, 'L2': 2}[self.regularization]
            for name, param in self.named_parameters():
                if name == 'lambda_weight':
                    # we don't regularize the lambda term, we only regularize coefficients.
                    continue
                nll += self.regularization_weight * torch.norm(param, p=L)
        return nll

    @property
    def device(self) -> torch.device:
        """Returns the device of the coefficient.

        Returns:
            torch.device: the device of the model.
        """
        return next(iter(self.item_coef_dict.values())).device

    @staticmethod
    def is_intercept_term(variable: str):
        # check if the given variable is an intercept (fixed effect) term.
        # intercept (fixed effect) terms are defined as 'intercept[*]' and looks like 'intercept[user]', 'intercept[item]', etc.
        return (variable.startswith('intercept[') and variable.endswith(']'))

    def get_coefficient(self, variable: str, level: Optional[str] = None) -> torch.Tensor:
        """Retrieve the coefficient tensor for the given variable.

        Args:
            variable (str): the variable name.
            level (str): from which level of model to extract the coefficient, can be 'item' or 'nest'. The `level` argument will be discarded if `variable` is `lambda`.

        Returns:
            torch.Tensor: the corresponding coefficient tensor of the requested variable.
        """
        if variable == 'lambda':
            return self.lambda_weight.detach().clone()

        if level not in ['item', 'nest']:
            raise ValueError(f"Level should be either 'item' or 'nest', got {level}.")

        return self.state_dict()[f'{level}_coef_dict.{variable}.coef'].detach().clone()

    # def clamp_lambdas(self):
    #     """
    #     Restrict values of lambdas to 0 < lambda <= 1 to guarantee the utility maximization property
    #     of the model.
    #     This method should be called everytime after optimizer.step().
    #     We add a self_clamp_called_flag to remind researchers if this method is not called.
    #     """
    #     for k in range(len(self.lambdas)):
    #         self.lambdas[k] = torch.clamp(self.lambdas[k], 1e-5, 1)
    #     self._clam_called_flag = True

    # @staticmethod
    # def add_constant(x: torch.Tensor, where: str='prepend') -> torch.Tensor:
    #     """A helper function used to add constant to feature tensor,
    #     x has shape (batch_size, num_classes, num_parameters),
    #     returns a tensor of shape (*, num_parameters+1).
    #     """
    #     batch_size, num_classes, num_parameters = x.shape
    #     ones = torch.ones((batch_size, num_classes, 1))
    #     if where == 'prepend':
    #         new = torch.cat((ones, x), dim=-1)
    #     elif where == 'append':
    #         new = torch.cat((x, ones), dim=-1)
    #     else:
    #         raise Exception
    #     return new
