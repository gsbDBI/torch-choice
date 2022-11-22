"""
Formula parser for easier model specification.
"""
from torch_choice.data.choice_dataset import ChoiceDataset
from typing import Tuple, Dict


def parse_formula(formula: str, data: ChoiceDataset) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Parse a formula string to the coef_variation_dict and num_param_dict.
    """
    coef_variation_dict = dict()
    num_param_dict = dict()
    if formula == '':
        # empty formula.
        return coef_variation_dict, num_param_dict


    # example: (var1|item) + (var2|item) + (var3|item)
    # infer num param from data.
    formula = formula.replace(' ', '')  # delete all spaces.
    term_list = formula.split('+')  # a list of elements like (var2|item).
    for term in term_list:
        # e.g., term: (var2|item)
        term = term.strip('()')
        # get the variable/observable and its variation.
        variable, specificity = term.split('|')[0], term.split('|')[1]
        assert specificity in ['constant', 'item', 'item-full', 'user', 'user-item', 'user-item-full'], f'Component {term} must be one of constant, item, item-full, user, user-item, user-item-full.'

        # add to the coef_variation_dict dictionary.
        coef_variation_dict[variable] = specificity

        # retrieve the dimension of observable (i.e., number of parameters).
        if variable == 'intercept':
            # intercept only has one parameter.
            num_param_dict[variable] = 1
        else:
            # get the dimension of variable/observable from the data.
            variable_dimension = getattr(data, variable).shape[-1]
            num_param_dict[variable] = variable_dimension

    return coef_variation_dict, num_param_dict