# import unittest
# import torch
# import math
# import torch.nn as nn

# try:
#     from torch_choice.utils.std import parameter_std
#     REAL_STD = True
# except ImportError:
#     REAL_STD = False
#     def parameter_std(x, loss_fn=None):
#         return torch.std(torch.tensor(x, dtype=torch.float32))


# # Define a DummyCoefficient that wraps a parameter in a 'coef' attribute
# class DummyCoefficient:
#     def __init__(self, data):
#         if not isinstance(data, torch.Tensor):
#             data = torch.tensor(data, dtype=torch.float32)
#         self.coef = nn.Parameter(data)


# # Define DummyModel holding a plain dictionary of coefficients
# class DummyModel(nn.Module):
#     def __init__(self, data):
#         super(DummyModel, self).__init__()
#         self.coefficients = {'param': DummyCoefficient(data)}
#     def state_dict(self, *args, **kwargs):
#         return {'coefficients.param': self.coefficients['param'].coef}


# class DummyLoss(nn.Module):
#     def forward(self, model):
#         return torch.mean(model.coefficients['param'].coef ** 2)


# class TestParameterStd(unittest.TestCase):
#     def test_1d_tensor(self):
#         # Create a DummyModel with 1D data
#         dummy = DummyModel([1.0, 2.0, 3.0])
#         loss_fn = DummyLoss()
#         result = parameter_std(dummy, loss_fn=loss_fn)
#         expected = torch.std(dummy.coefficients['param'].coef.detach())
#         self.assertAlmostEqual(result['std'].item(), expected.item(), places=5)

#     def test_2d_tensor(self):
#         # Create a DummyModel with 2D data
#         dummy = DummyModel([[1.0, 2.0], [3.0, 4.0]])
#         loss_fn = DummyLoss()
#         result = parameter_std(dummy, loss_fn=loss_fn)
#         expected = torch.std(dummy.coefficients['param'].coef.detach())
#         self.assertAlmostEqual(result['std'].item(), expected.item(), places=5)

#     def test_list_input(self):
#         # Wrap list input in DummyModel
#         dummy = DummyModel([1.0, 2.0, 3.0])
#         loss_fn = DummyLoss()
#         result = parameter_std(dummy, loss_fn=loss_fn)
#         expected = torch.std(dummy.coefficients['param'].coef.detach())
#         self.assertAlmostEqual(result['std'].item(), expected.item(), places=5)

#     def test_empty_tensor(self):
#         with self.assertRaises(Exception):
#             dummy = DummyModel([])
#             loss_fn = DummyLoss()
#             parameter_std(dummy, loss_fn=loss_fn)

#     def test_empty_list(self):
#         with self.assertRaises(Exception):
#             dummy = DummyModel([])
#             loss_fn = DummyLoss()
#             parameter_std(dummy, loss_fn=loss_fn)

# if __name__ == '__main__':
#     unittest.main()