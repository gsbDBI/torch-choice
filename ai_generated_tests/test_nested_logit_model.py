import unittest
import torch

from torch_choice.model.nested_logit_model import NestedLogitModel
from torch_choice.data.choice_dataset import ChoiceDataset

class DummyDataset(ChoiceDataset):
    def __init__(self):
        import torch
        self.item_index = torch.tensor([0, 1], dtype=torch.long)
        self.itemsession_cost_freq_ovt = torch.rand(2, 2, 1)
        self.session_income = torch.rand(2, 1)
        self.itemsession_ivt = torch.rand(2, 2)
        # Add attribute 'x' required for formula 'y ~ x'
        self.x = torch.rand(2, 1)

class TestNestedLogitModel(unittest.TestCase):
    def test_model_creation_with_valid_formula(self):
        ds = DummyDataset()
        # Supply a dummy nest_to_item mapping for testing purposes - values should be lists
        nest_to_item = {0: [0], 1: [1]}
        # Supply dummy coefficient dictionaries for item and nest levels
        item_coef_variation_dict = {"dummy_item[constant]": "constant"}
        nest_coef_variation_dict = {"dummy_nest[constant]": "constant"}
        # Supply dummy num_param dictionaries for item and nest levels
        item_num_param_dict = {"dummy_item[constant]": 1}
        nest_num_param_dict = {"dummy_nest[constant]": 1}
        model = NestedLogitModel(dataset=ds, nest_to_item=nest_to_item,
                                  item_coef_variation_dict=item_coef_variation_dict,
                                  nest_coef_variation_dict=nest_coef_variation_dict,
                                  item_num_param_dict=item_num_param_dict,
                                  nest_num_param_dict=nest_num_param_dict)
        self.assertIsNotNone(model)
        self.assertIsInstance(model, NestedLogitModel)

if __name__ == '__main__':
    unittest.main()