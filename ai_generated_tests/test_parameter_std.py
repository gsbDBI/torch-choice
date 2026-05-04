import unittest
import torch
import torch.nn as nn

from torch_choice.utils.std import parameter_std


class DummyCoefficient(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.coef = nn.Parameter(torch.zeros(shape, dtype=torch.float32))


class DummyModel(nn.Module):
    def __init__(self, param_shape):
        super().__init__()
        self.coef_dict = nn.ModuleDict({
            'param': DummyCoefficient(param_shape)
        })


def make_quadratic_loss_fn(diagonal: torch.Tensor):
    # Returns loss(model) = 0.5 * sum_i diagonal[i] * p_i^2
    def loss_fn(model: nn.Module) -> torch.Tensor:
        p = model.coef_dict['param'].coef.reshape(-1)
        d = diagonal.reshape(-1).to(p)
        return 0.5 * (d * p.pow(2)).sum()
    return loss_fn


class TestParameterStdHessian(unittest.TestCase):
    def test_1d_diagonal_hessian(self):
        diagonal = torch.tensor([2.0, 8.0, 18.0], dtype=torch.float32)
        model = DummyModel(param_shape=(3,))
        loss_fn = make_quadratic_loss_fn(diagonal)

        std_dict = parameter_std(model, loss_fn)
        std = std_dict['coef_dict.param.coef'].reshape(-1)

        expected = torch.sqrt(1.0 / diagonal).to(std.dtype)
        self.assertTrue(torch.allclose(std, expected, atol=1e-5, rtol=1e-5))

    def test_2d_diagonal_hessian(self):
        diagonal = torch.tensor([1.0, 4.0, 9.0, 16.0], dtype=torch.float32)
        model = DummyModel(param_shape=(2, 2))
        loss_fn = make_quadratic_loss_fn(diagonal)

        std_dict = parameter_std(model, loss_fn)
        std = std_dict['coef_dict.param.coef'].reshape(-1)

        expected = torch.sqrt(1.0 / diagonal).to(std.dtype)
        self.assertTrue(torch.allclose(std, expected, atol=1e-5, rtol=1e-5))


if __name__ == '__main__':
    unittest.main()