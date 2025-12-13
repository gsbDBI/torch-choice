"""
Legacy PyTorch Lightning helper maintained for backward compatibility.
"""
import warnings
from copy import deepcopy
from typing import Optional, Union

import pytorch_lightning as pl
import torch

from torch_choice.data import ChoiceDataset
from torch_choice.model.conditional_logit_model import ConditionalLogitModel
from torch_choice.model.nested_logit_model import NestedLogitModel


class LightningModelWrapper(pl.LightningModule):
    """Deprecated LightningModule wrapper retained for external users/tests."""

    def __init__(
        self,
        model: Union[ConditionalLogitModel, NestedLogitModel],
        learning_rate: float,
        model_optimizer: str,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.optimizer_class_string = model_optimizer

    def __str__(self) -> str:
        return str(self.model)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @torch.no_grad()
    def _get_performance_dict(self, batch):
        item_index = (
            batch["item"].item_index
            if isinstance(self.model, NestedLogitModel)
            else batch.item_index
        )
        ll = -self.model.negative_log_likelihood(batch, item_index).detach().item()
        return {"log_likelihood": ll}

    def training_step(self, batch, batch_idx):
        item_index = (
            batch["item"].item_index
            if isinstance(self.model, NestedLogitModel)
            else batch.item_index
        )
        loss = self.model.loss(batch, item_index)
        self.log("train_loss", loss, prog_bar=False, batch_size=len(batch))
        return loss

    def validation_step(self, batch, batch_idx):
        for key, val in self._get_performance_dict(batch).items():
            self.log("val_" + key, val, prog_bar=False, batch_size=len(batch))

    def test_step(self, batch, batch_idx):
        for key, val in self._get_performance_dict(batch).items():
            self.log("test_" + key, val, prog_bar=False, batch_size=len(batch))

    def configure_optimizers(self):
        return getattr(torch.optim, self.optimizer_class_string)(
            self.parameters(), lr=self.learning_rate
        )

def run(
    model: Union[ConditionalLogitModel, NestedLogitModel],
    dataset_train: ChoiceDataset,
    dataset_val: Optional[ChoiceDataset] = None,
    dataset_test: Optional[ChoiceDataset] = None,
    model_optimizer: str = "Adam",
    batch_size: int = -1,
    learning_rate: float = 0.01,
    num_epochs: int = 10,
    num_workers: int = 0,
    device: Optional[str] = None,
    report_std: bool = True,
    compute_std: Optional[bool] = None,  # Legacy arg, ignored (equivalent to report_std)
    **trainer_kwargs,
) -> Union[ConditionalLogitModel, NestedLogitModel]:
    """Backward compatible Lightning runner."""
    warnings.warn(
        "torch_choice.utils.run_helper_lightning.run is deprecated. "
        "Please call `model.fit(..., backend='lightning')` instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    assert isinstance(model, (ConditionalLogitModel, NestedLogitModel)), (
        f"A model of type {type(model)} is not supported by this runner."
    )

    # Respect compute_std if provided for backward compat (overrides report_std).
    if compute_std is not None:
        report_std = compute_std

    model_copy = deepcopy(model)
    estimation_output = model_copy.fit(
        dataset_train=dataset_train,
        dataset_val=dataset_val,
        dataset_test=dataset_test,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        model_optimizer=model_optimizer,
        device=device,
        num_workers=num_workers,
        backend="lightning",
        **trainer_kwargs,
    )

    if not report_std:
        warnings.warn(
            "`report_std=False` is ignored; EstimationOutput is always returned.",
            DeprecationWarning,
            stacklevel=2,
        )

    return estimation_output