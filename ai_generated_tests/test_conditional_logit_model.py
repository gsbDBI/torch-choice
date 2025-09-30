import unittest
import torch

from torch_choice.model.conditional_logit_model import ConditionalLogitModel
from torch_choice.data.choice_dataset import ChoiceDataset

class DummyDataset(ChoiceDataset):
    def __init__(self):
        import torch
        self.item_index = torch.tensor([0, 1], dtype=torch.long)
        self.itemsession_cost_freq_ovt = torch.rand(2, 2, 1)
        self.session_income = torch.rand(2, 1)
        self.itemsession_ivt = torch.rand(2, 2)
        self.x = torch.rand(2, 1)

class TestConditionalLogitModel(unittest.TestCase):
    def test_no_formula_no_coef(self):
        with self.assertRaises(ValueError):
            ConditionalLogitModel(formula=None, dataset=None, coef_variation_dict=None)

    def test_both_formula_and_coef(self):
        ds = DummyDataset()
        with self.assertRaises(ValueError):
            ConditionalLogitModel(formula='(x|constant)', dataset=ds, coef_variation_dict={'x': 'constant'})

    def test_formula_no_dataset(self):
        with self.assertRaises(ValueError):
            ConditionalLogitModel(formula='(x|constant)', dataset=None, coef_variation_dict=None)

    def test_model_creation_with_valid_formula(self):
        ds = DummyDataset()
        model = ConditionalLogitModel(formula='(x|constant)', dataset=ds)
        self.assertIsNotNone(model)

if __name__ == '__main__':
    unittest.main()