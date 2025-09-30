import unittest
import torch

from torch_choice.utils.run_helper_lightning import LightningModelWrapper

class DummyModel:
    def __init__(self, value):
        self.value = value
    def __call__(self, x):
        return x * self.value
    def loss(self, batch, item_index):
        return torch.tensor(0.)
    def negative_log_likelihood(self, batch, item_index):
        return torch.tensor(0.)
    def __str__(self):
        return f"DummyModel with value {self.value}"

class TestLightningModelWrapper(unittest.TestCase):
    def test_str_method(self):
        dummy = DummyModel(5)
        wrapper = LightningModelWrapper(dummy, learning_rate=0.01, model_optimizer='Adam')
        self.assertIn('DummyModel with value', str(wrapper))

    def test_forward_call(self):
        dummy = DummyModel(2)
        wrapper = LightningModelWrapper(dummy, learning_rate=0.01, model_optimizer='Adam')
        input_tensor = torch.tensor(3)
        output = wrapper(input_tensor)
        # Since DummyModel multiplies input by its value, output should be 6
        self.assertEqual(output.item(), 6)

if __name__ == '__main__':
    unittest.main()