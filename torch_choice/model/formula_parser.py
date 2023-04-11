import copy
from typing import Dict, Tuple

from torch_choice.data.choice_dataset import ChoiceDataset


def parse_formula(formula: str, data: ChoiceDataset) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Generates the coef_variation_dict and the num_samples_dict for Conditional/Nested logit models from a R-like
    formula and a ChoiceDataset.
    The parser reads variations of coefficients for different observables from the formula, then the parser retrieves
    the number of parameters for each coefficient (i.e., the dimension of each observable) from the ChoiceDataset.

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

    Returns:
        Tuple[Dict[str, int], Dict[str, int]]: coef_variation_dict and num_param_dict representing the model specified by the formula.
    """
    coef_variation_dict = dict()
    num_param_dict = dict()
    if formula == '':
        # empty formula, which is allowed for the category-level model in the nested logit model.
        return coef_variation_dict, num_param_dict


    # example: (var1|item) + (var2|item) + (var3|item)
    formula = formula.replace(' ', '')  # delete all spaces.
    term_list = formula.split('+')  # a list of elements like (var2|item).
    for term in term_list:
        # e.g., term: (var|item)
        term = term.strip('()')  # (var|item) --> var|item
        # get the variable/observable and its variation.
        corresponding_observable, specificity = term.split('|')[0], term.split('|')[1]
        variable = copy.deepcopy(corresponding_observable)

        if variable == '1':
            variable = 'intercept'  # the R-fashion for specifying intercept is 1, but we use `intercept` internally instead.

        # rename variable to incorporate the variation.
        variable = f"{variable}[{specificity}]"

        if variable in coef_variation_dict.keys():
            raise ValueError(f"variable[level of variation]={variable} is specified more than once in the formula, please remove the redundant one.")

        assert specificity in ['constant', 'item', 'item-full', 'user', 'user-item', 'user-item-full'], f'Component {term} must be one of constant, item, item-full, user, user-item, user-item-full.'

        # add to the coef_variation_dict dictionary.
        coef_variation_dict[variable] = specificity

        # retrieve the dimension of observable (i.e., number of parameters).
        if variable.startswith('intercept[') and variable.endswith(']'):
            # intercept only has one parameter.
            num_param_dict[variable] = 1
        else:
            # get the dimension of variable/observable from the data.
            num_param_dict[variable] = getattr(data, corresponding_observable).shape[-1]

    return coef_variation_dict, num_param_dict