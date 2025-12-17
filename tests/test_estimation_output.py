import unittest

import pandas as pd
import torch

from torch_choice.data import ChoiceDataset
from torch_choice.model import ConditionalLogitModel
from torch_choice.utils.estimation_output import EstimationOutput

try:  # pragma: no cover - optional dependency
    import pytorch_lightning as pl  # noqa: F401

    _LIGHTNING_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _LIGHTNING_AVAILABLE = False


def _build_dataset(num_obs: int = 32, num_items: int = 3) -> ChoiceDataset:
    item_index = torch.randint(low=0, high=num_items, size=(num_obs,))
    user_index = torch.zeros(num_obs, dtype=torch.long)
    session_index = torch.arange(num_obs, dtype=torch.long) % 4
    return ChoiceDataset(
        item_index=item_index,
        num_items=num_items,
        user_index=user_index,
        num_users=1,
        session_index=session_index,
        num_sessions=4,
    )


def _build_model(num_items: int = 3) -> ConditionalLogitModel:
    return ConditionalLogitModel(
        coef_variation_dict={"intercept": "item"},
        num_param_dict={"intercept": 1},
        num_items=num_items,
    )


class TestEstimationOutputTorchBackend(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_fit_returns_estimation_output_torch_backend(self):
        dataset = _build_dataset()
        model = _build_model(num_items=dataset.num_items)

        output = model.fit(
            dataset,
            batch_size=-1,
            learning_rate=0.05,
            num_epochs=5,
            backend="torch",
            report_frequency=1,
        )

        self.assertIsInstance(output, EstimationOutput)
        self.assertIs(output.model, model)
        self.assertIsInstance(output.coef_summary, pd.DataFrame)
        self.assertGreater(len(output.coef_summary), 0)
        self.assertIsInstance(output.train_ll, float)
        self.assertEqual(output.backend, "torch")
        self.assertIn("intercept", "".join(output.coef_summary.index))


@unittest.skipUnless(_LIGHTNING_AVAILABLE, "PyTorch Lightning is required for this test.")
class TestEstimationOutputLightningBackend(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)

    def test_fit_returns_estimation_output_lightning_backend(self):
        dataset = _build_dataset()
        model = _build_model(num_items=dataset.num_items)

        output = model.fit(
            dataset,
            batch_size=-1,
            learning_rate=0.05,
            num_epochs=3,
            backend="lightning",
            num_workers=0,
        )

        self.assertIsInstance(output, EstimationOutput)
        self.assertEqual(output.backend, "lightning")
        self.assertIsInstance(output.coef_summary, pd.DataFrame)

