from __future__ import annotations

import warnings
import time
import warnings
from copy import deepcopy
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import norm

from torch_choice.data.choice_dataset import ChoiceDataset
from torch_choice.data.joint_dataset import JointDataset
from torch_choice.data.utils import create_data_loader
from torch_choice.utils.estimation_output import EstimationOutput
from torch_choice.utils.std import parameter_std


BatchType = Union[ChoiceDataset, Dict[str, ChoiceDataset]]
DatasetType = Union[ChoiceDataset, JointDataset]

# Avoid reporting p-values as 0 due to floating point underflow.
# 2e-16 matches the common R convention of printing "< 2.2e-16".
MIN_REPORTED_PVALUE = 2e-16
SCIENTIFIC_NOTATION_THRESHOLD = 1e-3


class ChoiceModelFitMixin:
    """Mixin exposing sklearn-style fit/run helpers shared by choice models."""

    def fit(
        self,
        dataset_train: DatasetType,
        dataset_val: Optional[DatasetType] = None,
        dataset_test: Optional[DatasetType] = None,
        *,
        batch_size: int = -1,
        learning_rate: float = 0.01,
        num_epochs: int = 5000,
        model_optimizer: str = "Adam",
        device: Optional[Union[str, torch.device]] = None,
        backend: str = "torch",
        num_workers: int = 0,
        report_frequency: Optional[int] = None,
        print_summary: bool = True,
        **backend_kwargs: Any,
    ) -> EstimationOutput:
        """Train the current model instance.

        Args:
            dataset_train: dataset used for training.
            dataset_val: optional validation dataset.
            dataset_test: optional test dataset (evaluated after training).
            batch_size: mini-batch size (-1 means full batch).
            learning_rate: optimizer learning rate.
            num_epochs: number of epochs to train.
            model_optimizer: optimizer name from torch.optim (e.g., Adam, LBFGS).
            device: device string or torch.device (e.g., "cpu", "cuda", "cuda:0").
            backend: one of {"torch", "lightning"}.
            num_workers: DataLoader num_workers value.
            report_frequency: number of epochs between progress prints (torch backend only).
            print_summary: whether to print the regression table/log-likelihood after fitting.
            backend_kwargs: backend-specific keyword arguments. Torch backend accepts
                "optimizer_kwargs", "scheduler_kwargs", and "scheduler_class". Lightning backend
                forwards kwargs directly to pytorch_lightning.Trainer.

        Returns:
            EstimationOutput: structured object containing the trained model, regression table,
            log-likelihoods, and metadata, with dictionary-style access similar to HuggingFace's
            ``ModelOutput``.

        Example:
            >>> output = model.fit(dataset_train, num_epochs=100, learning_rate=0.01)
            >>> output.train_ll
            -1874.3
            >>> output.coef_summary.head()
            >>> print(output)
            # markdown-formatted regression summary.
        """
        if dataset_train is None:
            raise ValueError("dataset_train must be provided.")

        backend_key = backend.lower()
        # Normalize backend name once to keep the public API case-insensitive.
        if backend_key not in {"torch", "lightning"}:
            raise ValueError(f"Unsupported backend '{backend}'. Expected 'torch' or 'lightning'.")

        device_str = str(device) if device is not None else None
        # If the user requested a device, move all datasets first (so their tensors are
        # colocated with the model parameters during training/evaluation).
        dataset_train = self._maybe_move_dataset(dataset_train, device_str)
        dataset_val = self._maybe_move_dataset(dataset_val, device_str)
        dataset_test = self._maybe_move_dataset(dataset_test, device_str)
        if device_str is not None:
            # Mirror dataset placement to avoid accidental CPU<->GPU mismatches.
            self.to(device_str)

        start_time = time.perf_counter()
        # Resolve the actual device used downstream (training loop + LL evaluation).
        active_device = self._infer_active_device(device_str)
        if backend_key == "torch":
            backend_result = self._fit_with_torch_backend(
                dataset_train=dataset_train,
                dataset_val=dataset_val,
                dataset_test=dataset_test,
                batch_size=batch_size,
                learning_rate=learning_rate,
                num_epochs=num_epochs,
                model_optimizer=model_optimizer,
                num_workers=num_workers,
                report_frequency=report_frequency,
                device_hint=active_device,
                **backend_kwargs,
            )
        elif backend_key == "lightning":
            backend_result = self._fit_with_lightning_backend(
                dataset_train=dataset_train,
                dataset_val=dataset_val,
                dataset_test=dataset_test,
                batch_size=batch_size,
                learning_rate=learning_rate,
                num_epochs=num_epochs,
                model_optimizer=model_optimizer,
                num_workers=num_workers,
                device_hint=active_device,
                **backend_kwargs,
            )
        else:
            raise ValueError(f"Unsupported backend '{backend}'. Expected 'torch' or 'lightning'.")

        elapsed_time = time.perf_counter() - start_time

        # Build regression-style summary + package metrics/model into EstimationOutput.
        return self._summarize_estimation(
            dataset_train=dataset_train,
            dataset_val=dataset_val,
            dataset_test=dataset_test,
            batch_size=batch_size,
            learning_rate=learning_rate,
            backend_key=backend_key,
            optimizer_name=model_optimizer,
            elapsed_time=elapsed_time,
            backend_result=backend_result,
            device_hint=active_device,
            print_summary=print_summary,
        )

    def run(self, *args, **kwargs):
        """Backward-compatible alias for :meth:`fit`.

        This method exists to support older notebooks and scripts that call
        ``model.run(...)``. It forwards all arguments to :meth:`fit` and returns the
        same :class:`~torch_choice.utils.estimation_output.EstimationOutput` object.

        Notes:
            - Emits a :class:`DeprecationWarning` to encourage migration to
              :meth:`fit`.
            - Any printing behavior is controlled by the forwarded ``print_summary``
              argument (see :meth:`fit`).
        """
        warnings.warn(
            "model.run(...) is kept for backward compatibility. "
            "Please switch to model.fit(...) going forward.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.fit(*args, **kwargs)

    # --------------------------------------------------------------------------------------------------
    # Torch backend
    # --------------------------------------------------------------------------------------------------
    def _fit_with_torch_backend(
        self,
        *,
        dataset_train: DatasetType,
        dataset_val: Optional[DatasetType],
        dataset_test: Optional[DatasetType],
        batch_size: int,
        learning_rate: float,
        num_epochs: int,
        model_optimizer: str,
        num_workers: int,
        report_frequency: Optional[int],
        device_hint: Optional[Union[str, torch.device]],
        **backend_kwargs: Any,
    ) -> Dict[str, Any]:
        """Train using a native PyTorch optimization loop.

        This backend performs explicit mini-batch training with a
        :class:`torch.utils.data.DataLoader` and a ``torch.optim`` optimizer. It is
        intentionally lightweight (no Lightning dependency) and is suitable for
        simple scripts or environments where Lightning is not desired.

        Args:
            dataset_train: Training dataset.
            dataset_val: Optional validation dataset (evaluated after training).
            dataset_test: Optional test dataset (evaluated after training).
            batch_size: Mini-batch size (-1 means full-batch).
            learning_rate: Optimizer learning rate.
            num_epochs: Number of epochs to train.
            model_optimizer: Name of a class in ``torch.optim`` (e.g. ``"Adam"``,
                ``"LBFGS"``).
            num_workers: ``DataLoader`` worker count.
            report_frequency: How often to print average loss during training.
            device_hint: Preferred device for training/evaluation.
            **backend_kwargs: Torch-backend specific kwargs:
                - ``optimizer_kwargs``: dict forwarded into the optimizer constructor.
                - ``scheduler_class`` / ``scheduler_kwargs``: optional LR scheduler.

        Returns:
            A dictionary of backend results containing (at minimum):
            - ``train_ll`` / ``val_ll`` / ``test_ll``: log-likelihoods
            - ``epochs``: epochs run
            - ``backend``: backend identifier

        Raises:
            ValueError: If the requested optimizer is not available in ``torch.optim``.

        Notes:
            This method prints training progress and log-likelihood diagnostics for the
            torch backend. The final regression table printing is controlled by the
            higher-level ``print_summary`` flag in :meth:`fit`.
        """
        optimizer_kwargs = backend_kwargs.pop("optimizer_kwargs", {}) or {}
        scheduler_kwargs = backend_kwargs.pop("scheduler_kwargs", None)
        scheduler_class = backend_kwargs.pop("scheduler_class", torch.optim.lr_scheduler.StepLR)

        if backend_kwargs:
            # Remaining keys are most likely typos; warn rather than silently ignore.
            warnings.warn(
                f"Unused torch backend kwargs: {sorted(backend_kwargs.keys())}",
                stacklevel=2,
            )

        optimizer_class = getattr(torch.optim, model_optimizer, None)
        # Resolve optimizer by string name (keeps the public API simple and flexible).
        if optimizer_class is None:
            raise ValueError(
                f"Optimizer '{model_optimizer}' is not available in torch.optim."
            )

        optimizer_args = dict(lr=learning_rate, **optimizer_kwargs)
        if optimizer_class is torch.optim.LBFGS:
            # LBFGS is a second-order method and uses a closure evaluated multiple times.
            optimizer_args.setdefault("max_iter", 20)
            optimizer_args.setdefault("history_size", 10)
        optimizer = optimizer_class(self.parameters(), **optimizer_args)

        scheduler = None
        if scheduler_class is not None:
            # Default schedule: very infrequent step-down to keep training stable.
            scheduler_kwargs = scheduler_kwargs or {"step_size": 10000, "gamma": 0.7}
            scheduler = scheduler_class(optimizer, **scheduler_kwargs)

        self.train()
        train_loader = create_data_loader(
            dataset_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        if report_frequency is None:
            # Default: ~10 progress updates across the run (at minimum once).
            report_frequency = max(num_epochs // 10, 1)

        # Use a concrete device for moving batches/targets inside the training loop.
        training_device = self._infer_active_device(device_hint)

        for epoch in range(1, num_epochs + 1):
            epoch_loss = 0.0
            num_obs = 0
            for batch in train_loader:
                # Support both ChoiceDataset batches and dict-of-datasets batches.
                batch = self._move_batch_to_device(batch, training_device)
                targets = self._get_item_index_from_batch(batch)
                if torch.is_tensor(targets):
                    targets = targets.to(training_device)

                def closure():
                    # Closure is required by LBFGS so it can reevaluate the objective.
                    optimizer.zero_grad()
                    loss_val = self.loss(batch, targets)
                    loss_val.backward()
                    return loss_val

                if optimizer_class is torch.optim.LBFGS:
                    # For LBFGS, `step` drives repeated closure evaluations internally.
                    loss_value = closure()
                    optimizer.step(closure)
                else:
                    # Standard first-order optimizers: one forward/backward per batch.
                    loss_value = self.loss(batch, targets)
                    optimizer.zero_grad()
                    loss_value.backward()
                    optimizer.step()

                epoch_loss += float(loss_value.detach().item())
                num_obs += int(targets.numel())

            if scheduler is not None:
                # Step schedulers once per epoch (simple default).
                scheduler.step()

            if report_frequency and (epoch % report_frequency == 0 or epoch == num_epochs):
                avg_loss = epoch_loss / max(num_obs, 1)
                print(f"[fit-torch] Epoch {epoch}/{num_epochs} - avg loss per obs: {avg_loss:.6f}")

        # Evaluation summary (optional logging mirrors torch helper)
        self.eval()
        train_ll = self._compute_dataset_log_likelihood(
            dataset_train,
            batch_size=batch_size,
            num_workers=num_workers,
            device=training_device,
        )
        print(f"[fit-torch] Training log-likelihood: {train_ll:.6f}")

        val_ll = None
        if dataset_val is not None:
            val_ll = self._compute_dataset_log_likelihood(
                dataset_val,
                batch_size=batch_size,
                num_workers=num_workers,
                device=training_device,
            )
            print(f"[fit-torch] Validation log-likelihood: {val_ll:.6f}")

        test_ll = None
        if dataset_test is not None:
            test_ll = self._compute_dataset_log_likelihood(
                dataset_test,
                batch_size=batch_size,
                num_workers=num_workers,
                device=training_device,
            )
            print(f"[fit-torch] Test log-likelihood: {test_ll:.6f}")

        return {
            "train_ll": train_ll,
            "val_ll": val_ll,
            "test_ll": test_ll,
            "epochs": num_epochs,
            "backend": "torch",
        }

    # --------------------------------------------------------------------------------------------------
    # Lightning backend
    # --------------------------------------------------------------------------------------------------
    def _fit_with_lightning_backend(
        self,
        *,
        dataset_train: DatasetType,
        dataset_val: Optional[DatasetType],
        dataset_test: Optional[DatasetType],
        batch_size: int,
        learning_rate: float,
        num_epochs: int,
        model_optimizer: str,
        num_workers: int,
        device_hint: Optional[Union[str, torch.device]],
        **trainer_kwargs: Any,
    ) -> Dict[str, Any]:
        """Train using a minimal PyTorch Lightning wrapper.

        The Lightning backend integrates with :class:`pytorch_lightning.Trainer`
        for training loop orchestration, callback support, and richer diagnostics.
        This method creates an internal ``LightningModule`` wrapper around the model
        and trains with the provided datasets/dataloaders.

        Args:
            dataset_train: Training dataset.
            dataset_val: Optional validation dataset. If provided, an
                :class:`pytorch_lightning.callbacks.EarlyStopping` callback is added
                by default unless the user already supplied one.
            dataset_test: Optional test dataset (evaluated after training).
            batch_size: Mini-batch size (-1 means full-batch).
            learning_rate: Optimizer learning rate.
            num_epochs: Maximum epochs for Lightning ``Trainer``.
            model_optimizer: Name of a class in ``torch.optim`` used inside
                ``configure_optimizers``.
            num_workers: ``DataLoader`` worker count.
            device_hint: Preferred device for evaluation of log-likelihoods.
            **trainer_kwargs: Additional kwargs forwarded to
                :class:`pytorch_lightning.Trainer`.

        Returns:
            A dictionary of backend results containing (at minimum):
            - ``train_ll`` / ``val_ll`` / ``test_ll``: log-likelihoods
            - ``epochs``: epochs run (best-effort estimate)
            - ``backend``: backend identifier

        Raises:
            ImportError: If PyTorch Lightning is not installed.
            ValueError: If the requested optimizer is not available in ``torch.optim``.

        Notes:
            - Log-likelihoods returned here are computed explicitly after training using
              :meth:`_compute_dataset_log_likelihood` (not taken from Lightning logs),
              so they remain consistent across backends.
            - Final regression table printing is controlled by the higher-level
              ``print_summary`` flag in :meth:`fit`.
        """
        try:
            import pytorch_lightning as pl
            from pytorch_lightning.callbacks import EarlyStopping
            from pytorch_lightning.utilities.rank_zero import rank_zero_info
        except ImportError as exc:
            raise ImportError(
                "PyTorch Lightning is required for backend='lightning'. "
                "Install it via `pip install pytorch-lightning`."
            ) from exc

        # Lazily define the wrapper to avoid importing Lightning at module load time.
        class _LightningModelWrapper(pl.LightningModule):
            def __init__(self, wrapped_model, lr, optimizer_name):
                super().__init__()
                self.model = wrapped_model
                self.learning_rate = lr
                self.optimizer_name = optimizer_name

            def forward(self, *args, **kwargs):
                return self.model(*args, **kwargs)

            def _get_targets(self, batch):
                return self.model._get_item_index_from_batch(batch)

            def training_step(self, batch, batch_idx):
                targets = self._get_targets(batch)
                loss = self.model.loss(batch, targets)
                # Log batch loss (useful for progress bars / debugging).
                self.log("train_loss", loss, prog_bar=False, batch_size=int(targets.numel()))
                return loss

            def validation_step(self, batch, batch_idx):
                targets = self._get_targets(batch)
                # Log-likelihood is the negative of the (positive) NLL.
                ll = -self.model.negative_log_likelihood(batch, targets, is_train=False)
                self.log("val_ll", ll, prog_bar=True, batch_size=int(targets.numel()))

            def test_step(self, batch, batch_idx):
                targets = self._get_targets(batch)
                # Mirror validation: report LL for test data.
                ll = -self.model.negative_log_likelihood(batch, targets, is_train=False)
                self.log("test_ll", ll, prog_bar=True, batch_size=int(targets.numel()))

            def configure_optimizers(self):
                optimizer_cls = getattr(torch.optim, self.optimizer_name, None)
                if optimizer_cls is None:
                    raise ValueError(
                        f"Optimizer '{self.optimizer_name}' is not available in torch.optim."
                    )
                return optimizer_cls(self.model.parameters(), lr=self.learning_rate)

        lightning_module = _LightningModelWrapper(self, learning_rate, model_optimizer)

        train_loader = create_data_loader(
            dataset_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        # Validation/test loaders should not shuffle so LL is deterministic.
        val_loader = (
            create_data_loader(
                dataset_val,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
            if dataset_val is not None
            else None
        )
        test_loader = (
            create_data_loader(
                dataset_test,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
            if dataset_test is not None
            else None
        )

        trainer_defaults = {"max_epochs": num_epochs}
        trainer_defaults.setdefault(
            "check_val_every_n_epoch", max(num_epochs // 100, 1)
        )
        trainer_defaults.setdefault("log_every_n_steps", max(num_epochs // 100, 1))

        callbacks = trainer_kwargs.pop("callbacks", None)
        if callbacks is None:
            callbacks = []
        if dataset_val is not None and not any(
            isinstance(cb, EarlyStopping) for cb in callbacks
        ):
            # Add a sensible default early-stopping rule unless the user already did.
            callbacks = callbacks + [
                EarlyStopping(monitor="val_ll", mode="max", patience=10, min_delta=0.001)
            ]
        trainer_defaults["callbacks"] = callbacks

        accelerator = trainer_kwargs.pop("accelerator", None)
        if accelerator is None:
            # Best-effort inference of accelerator from the wrapped model's device.
            device = getattr(self, "device", None)
            if device is not None:
                device_str = str(device)
                accelerator = "cuda" if "cuda" in device_str else device_str
        if accelerator is not None:
            trainer_defaults["accelerator"] = accelerator

        trainer_defaults.setdefault("devices", trainer_kwargs.pop("devices", "auto"))

        trainer_defaults.update(trainer_kwargs or {})
        trainer = pl.Trainer(**trainer_defaults)

        rank_zero_info("Starting PyTorch Lightning training loop.")
        trainer.fit(lightning_module, train_loader, val_loader)
        if test_loader is not None:
            rank_zero_info("Running PyTorch Lightning test loop.")
            trainer.test(lightning_module, test_loader)

        # Lightning epochs are 0-indexed internally; add 1 for a human-readable count.
        epochs_ran = getattr(trainer, "current_epoch", num_epochs - 1) + 1

        self.eval()
        inference_device = self._infer_active_device(device_hint)
        # Compute LL explicitly (instead of using Lightning logs) so metrics match the torch backend.
        train_ll = self._compute_dataset_log_likelihood(
            dataset_train,
            batch_size=batch_size,
            num_workers=num_workers,
            device=inference_device,
        )
        val_ll = (
            self._compute_dataset_log_likelihood(
                dataset_val,
                batch_size=batch_size,
                num_workers=num_workers,
                device=inference_device,
            )
            if dataset_val is not None
            else None
        )
        test_ll = (
            self._compute_dataset_log_likelihood(
                dataset_test,
                batch_size=batch_size,
                num_workers=num_workers,
                device=inference_device,
            )
            if dataset_test is not None
            else None
        )

        return {
            "train_ll": train_ll,
            "val_ll": val_ll,
            "test_ll": test_ll,
            "epochs": epochs_ran,
            "backend": "lightning",
        }

    # --------------------------------------------------------------------------------------------------
    # Shared helpers
    # --------------------------------------------------------------------------------------------------
    def negative_log_likelihood(
        self,
        batch: BatchType,
        y: torch.Tensor,
        is_train: bool = True,
    ) -> torch.Tensor:  # pragma: no cover
        """Compute the (summed) negative log-likelihood for a batch.

        This is the core objective used throughout this mixin:
            - training/evaluation log-likelihood reporting
            - Lightning backend validation/test logging

        Subclasses are expected to implement this method.

        Args:
            batch: A batch returned by the DataLoader. In this codebase it is either a
                :class:`~torch_choice.data.choice_dataset.ChoiceDataset` or a ``dict`` of
                datasets (e.g. for :class:`~torch_choice.data.joint_dataset.JointDataset`).
            y: A 1D integer tensor of chosen alternative indices.
            is_train: Whether to run the model in training mode (and potentially keep
                autograd graphs) for this call.

        Returns:
            A scalar tensor: negative log-likelihood summed over all observations.
        """
        raise NotImplementedError(
            "negative_log_likelihood must be implemented by subclasses using ChoiceModelFitMixin."
        )

    def loss(
        self,
        batch: BatchType,
        y: torch.Tensor,
        is_train: bool = True,
    ) -> torch.Tensor:  # pragma: no cover
        """Training objective optimized by :meth:`fit`.

        Most models implement this as ``negative_log_likelihood`` plus optional
        regularization terms. By default, subclasses should override this (or at
        least implement :meth:`negative_log_likelihood` and override this method to
        call it).
        """
        raise NotImplementedError(
            "loss must be implemented by subclasses using ChoiceModelFitMixin."
        )

    def _get_item_index_from_batch(self, batch: BatchType) -> torch.Tensor:  # pragma: no cover
        """Extract target item indices from a training/eval batch.

        Subclasses must implement this to return the item indices used as targets in
        :meth:`loss` / :meth:`negative_log_likelihood`.

        Args:
            batch: Either a :class:`~torch_choice.data.choice_dataset.ChoiceDataset`
                batch or a mapping of feature-group to dataset (used by some models).

        Returns:
            A 1D integer tensor of shape ``(num_observations,)`` representing the chosen
            alternative index per observation.

        Raises:
            NotImplementedError: Always, unless implemented by a subclass.
        """
        raise NotImplementedError(
            "_get_item_index_from_batch must be implemented by subclasses."
        )

    def _maybe_move_dataset(
        self,
        dataset: Optional[DatasetType],
        device: Optional[Union[str, torch.device]],
    ) -> Optional[DatasetType]:
        """Move a dataset to a device if both are provided.

        Args:
            dataset: Dataset to move (or ``None``).
            device: Target device (string or :class:`torch.device`). If ``None``, no-op.

        Returns:
            The dataset moved to ``device``, or the original dataset if ``dataset`` or
            ``device`` is ``None``.
        """
        if dataset is None or device is None:
            return dataset
        return dataset.to(device)

    def _infer_active_device(
        self, device: Optional[Union[str, torch.device]]
    ) -> torch.device:
        """Infer which device should be used for computation.

        Resolution order:
            1. The explicitly provided ``device`` argument, if not ``None``.
            2. ``self.device`` attribute, if present.
            3. The device of the first parameter in ``self.parameters()``.
            4. CPU fallback.

        Args:
            device: Optional device override.

        Returns:
            A concrete :class:`torch.device` to use.
        """
        if device is not None:
            # 1. The explicitly provided ``device`` argument, if not ``None``.
            return torch.device(device)
        model_device = getattr(self, "device", None)
        if model_device is not None:
            # 2. ``self.device`` attribute, if present.
            return torch.device(model_device)
        try:
            # 3. The device of the first parameter in ``self.parameters()``.
            return next(self.parameters()).device
        except (StopIteration, AttributeError):
            # 4. CPU fallback.
            return torch.device("cpu")

    def _move_batch_to_device(
        self, batch: BatchType, device: Optional[torch.device]
    ) -> BatchType:
        """Move a batch to the target device.

        Supports two batch shapes used in this codebase:
            - a :class:`~torch_choice.data.choice_dataset.ChoiceDataset`
            - a ``dict`` mapping keys to objects with a ``.to(...)`` method

        Args:
            batch: Batch object returned by the DataLoader.
            device: Target device. If ``None``, returns the batch unchanged.

        Returns:
            The batch moved to ``device`` (or original batch if ``device`` is ``None``).

        Notes:
            When ``batch`` is a ``dict``, values are moved in-place.
        """
        # No device specified, return batch unchanged.
        if device is None:
            return batch
        # ChoiceDataset has a built-in .to() method for device transfer.
        if isinstance(batch, ChoiceDataset):
            return batch.to(device)
        # Handle dict-style batches by moving each tensor value individually.
        if isinstance(batch, dict):
            for key, value in batch.items():
                if hasattr(value, "to"):
                    # Mutate in-place so downstream code sees the moved tensors.
                    batch[key] = value.to(device)
            return batch
        # Fallback: return batch as-is if type is unrecognized.
        return batch

    def _compute_dataset_log_likelihood(
        self,
        dataset: DatasetType,
        *,
        batch_size: int,
        num_workers: int,
        device: Optional[Union[str, torch.device]] = None,
    ) -> float:
        """Compute total log-likelihood of the model on a dataset.

        This evaluates ``-negative_log_likelihood`` across the full dataset with
        ``torch.no_grad()`` and sums across batches.

        Args:
            dataset: Dataset to evaluate.
            batch_size: DataLoader batch size (-1 means full-batch evaluation).
            num_workers: DataLoader worker count.
            device: Optional device override for evaluation.

        Returns:
            The total log-likelihood (scalar float).
        """
        self.eval()
        eval_device = self._infer_active_device(device)
        dataloader = create_data_loader(
            dataset,
            # `batch_size=-1` is treated as full-batch evaluation for convenience.
            batch_size=batch_size if batch_size != -1 else len(dataset),
            shuffle=False,
            num_workers=num_workers,
        )
        total_ll = 0.0
        with torch.no_grad():
            for batch in dataloader:
                batch = self._move_batch_to_device(batch, eval_device)
                targets = self._get_item_index_from_batch(batch)
                if torch.is_tensor(targets):
                    targets = targets.to(eval_device)
                # negative_log_likelihood returns NLL; negate it to get log-likelihood.
                ll = -self.negative_log_likelihood(batch, targets, is_train=False)
                total_ll += float(ll.detach().item())
        return total_ll

    # --------------------------------------------------------------------------------------------------
    # Estimation summary helpers
    # --------------------------------------------------------------------------------------------------
    def _summarize_estimation(
        self,
        *,
        dataset_train: DatasetType,
        dataset_val: Optional[DatasetType],
        dataset_test: Optional[DatasetType],
        batch_size: int,
        learning_rate: float,
        backend_key: str,
        optimizer_name: str,
        elapsed_time: float,
        backend_result: Dict[str, Any],
        device_hint: Optional[Union[str, torch.device]],
        print_summary: bool,
    ) -> EstimationOutput:
        """Build the post-fit regression summary and return an :class:`EstimationOutput`.

        This method:
            - snapshots parameter means
            - attempts Hessian-based standard error computation
            - builds a coefficient report DataFrame (estimate/std/z/p/significance)
            - returns everything packaged into :class:`~torch_choice.utils.estimation_output.EstimationOutput`

        Args:
            dataset_train: The training dataset used for std-error calculation.
            dataset_val: Optional validation dataset (for reporting log-likelihood).
            dataset_test: Optional test dataset (for reporting log-likelihood).
            batch_size: Batch size used for training/evaluation (stored for provenance).
            learning_rate: Learning rate used (stored for provenance).
            backend_key: Backend identifier (e.g. ``"torch"``, ``"lightning"``).
            optimizer_name: Optimizer name used (stored for provenance).
            elapsed_time: Wall-clock time spent in the backend fit call.
            backend_result: Backend-returned metrics (must contain ``train_ll``; may
                contain ``val_ll``/``test_ll`` and ``epochs``).
            device_hint: Device used for std-error calculation/evaluation.
            print_summary: If True, prints the regression summary by printing the
                returned :class:`EstimationOutput`.

        Returns:
            EstimationOutput containing the model, coefficient summary table, log-likelihoods,
            and auxiliary metadata.

        Notes:
            Standard error computation can fail (e.g. singular Hessian). In that case we
            warn and populate std/z/p columns with NaNs.
        """
        # Work on a clone for Hessian-based std errors to avoid mutating/corrupting the training dataset.
        dataset_for_std = self._clone_dataset_for_std(dataset_train)
        dataset_for_std = self._maybe_move_dataset(dataset_for_std, device_hint)
        # Snapshot fitted parameters for reporting (detached so it won't keep autograd graphs).
        mean_dict = {name: param.detach().clone() for name, param in self.named_parameters()}

        # Compute Hessian/std errors on a deep-copied model to keep the fitted instance clean.
        model_clone = deepcopy(self)
        state_dict = deepcopy(self.state_dict())
        if "lambdas" in state_dict and "lambda_weight" in state_dict:
            lambdas = state_dict["lambdas"]
            lambda_weight = state_dict["lambda_weight"]
            if not torch.allclose(lambdas, lambda_weight):
                raise ValueError(
                    "NestedLogitModel state dict mismatch between lambdas and lambda_weight."
                )
        elif "lambda_weight" in state_dict:
            # NestedLogitModel in some configurations only stores lambda_weight.
            state_dict["lambdas"] = state_dict["lambda_weight"].detach().clone()
        model_clone.load_state_dict(state_dict, strict=True)

        try:
            # Hessian-based approximation; can fail if the Hessian is singular/ill-conditioned.
            std_dict = self._compute_parameter_std(model_clone, dataset_for_std)
        except (RuntimeError, torch.linalg.LinAlgError, IndexError) as err:
            if isinstance(err, NotImplementedError):
                warnings.warn(
                    f"Skipping standard error computation: {err}. "
                    "The regression table will omit Std. Err./z/p values for this run.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                warnings.warn(
                    f"Failed to compute parameter standard errors due to: {err}. "
                    "The regression table will omit Std. Err./z/p values for this run. "
                    "Try training longer or adjusting regularization to avoid a singular Hessian.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            std_dict = None

        # Convert raw parameter tensors into a tidy regression-style DataFrame.
        report = self._build_coefficient_report(mean_dict, std_dict)

        train_ll = backend_result.get("train_ll")
        val_ll = backend_result.get("val_ll")
        test_ll = backend_result.get("test_ll")

        output = EstimationOutput(
            model=self,
            coef_summary=report,
            train_ll=train_ll,
            val_ll=val_ll,
            test_ll=test_ll,
            mean_dict=mean_dict,
            std_dict=std_dict,
            epochs=backend_result.get("epochs"),
            learning_rate=learning_rate,
            batch_size=batch_size,
            elapsed_time=elapsed_time,
            backend=backend_key,
            optimizer=optimizer_name,
        )

        if print_summary:
            # EstimationOutput implements pretty-printing (markdown-ish) for interactive use.
            print(output)

        return output

    def _clone_dataset_for_std(self, dataset: DatasetType) -> DatasetType:
        """Clone a dataset for standard error computation.

        Some dataset implementations may cache tensors or internal state. For
        Hessian-based standard error calculation we prefer to operate on a clone to
        avoid unexpected side-effects on the training dataset.

        Args:
            dataset: Dataset to clone.

        Returns:
            A clone of ``dataset``. If the dataset has a ``clone()`` method, it is used;
            otherwise a ``deepcopy`` fallback is used.

        Raises:
            ValueError: If ``dataset`` is ``None``.
        """
        if dataset is None:
            raise ValueError("dataset_train is required for standard error calculation.")
        if hasattr(dataset, "clone"):
            try:
                # Prefer dataset-native cloning to preserve any internal invariants/caches.
                return dataset.clone()
            except Exception:
                pass
        return deepcopy(dataset)

    def _compute_parameter_std(self, model_clone, dataset_for_std):
        """Compute parameter standard errors using Hessian-based approximation.

        This delegates to :func:`torch_choice.utils.std.parameter_std`, which expects a
        callable returning a scalar negative log-likelihood loss. We implement that loss
        differently depending on the model type.

        Args:
            model_clone: A cloned model with the same parameters as the fitted model.
                The clone is used so that Hessian computations do not interfere with the
                original model's autograd graph/state.
            dataset_for_std: Dataset to evaluate the negative log-likelihood on.

        Returns:
            A dict mapping parameter names to tensors of standard errors (same shape as
            the corresponding parameter tensor).

        Raises:
            NotImplementedError: If the model type is not supported for std-error computation.
        """
        from torch_choice.model.conditional_logit_model import ConditionalLogitModel
        from torch_choice.model.nested_logit_model import NestedLogitModel

        if isinstance(self, ConditionalLogitModel):
            def nll_loss(model):
                # ConditionalLogitModel exposes logits; use cross-entropy as the summed NLL.
                y_pred = model(dataset_for_std)
                item_index = dataset_for_std.item_index.clone()
                if getattr(model, "model_outside_option", False):
                    # Convention: outside option encoded as -1; map it to the extra class at `num_items`.
                    outside_mask = item_index == -1
                    item_index[outside_mask] = model.num_items
                return F.cross_entropy(y_pred, item_index, reduction="sum")
        elif isinstance(self, NestedLogitModel):
            def nll_loss(model):
                # NestedLogitModel expects a dict-like batch; index the dataset into that format.
                indices = torch.arange(len(dataset_for_std))
                data = dataset_for_std[indices]
                return model.negative_log_likelihood(data, data["item"].item_index)
        else:
            # Keep training usable for new model types: subclasses can override this
            # method if they want Hessian-based standard errors.
            raise NotImplementedError(
                "Standard error computation is not implemented for this model type. "
                "Override `_compute_parameter_std(...)` to enable it."
            )

        return parameter_std(model_clone, nll_loss)

    def _build_coefficient_report(
        self,
        mean_dict: Dict[str, torch.Tensor],
        std_dict: Optional[Dict[str, torch.Tensor]],
    ) -> pd.DataFrame:
        """Build a coefficient report table similar to R regression summaries.

        Args:
            mean_dict: Mapping from parameter name to parameter tensor (detached).
            std_dict: Optional mapping from parameter name to std-error tensor. If
                missing (or a key is absent), Std. Err./z/p are filled with NaNs.

        Returns:
            A pandas DataFrame indexed by a human-readable coefficient name and with columns:
            ``Estimation``, ``Std. Err.``, ``z-value``, ``Pr(>|z|)``, and ``Significance``.

        Notes:
            - Coefficient names are derived from parameter names. For ModuleDict-based
              coefficient tensors we strip common prefixes like ``"coef_dict."`` and suffixes
              like ``".coef"``, then append an index ``_{k}`` for flattened entries.
            - P-values are two-sided normal approximations using ``scipy.stats.norm``.
        """
        rows = []
        for coef_name, mean_tensor in mean_dict.items():
            # Flatten every parameter tensor into one row per scalar entry.
            mean_np = mean_tensor.detach().cpu().numpy().reshape(-1)
            if std_dict is not None and coef_name in std_dict:
                std_np = std_dict[coef_name].detach().cpu().numpy().reshape(-1)
            else:
                std_np = np.full_like(mean_np, np.nan, dtype=float)

            # Make coefficient names more user-facing by stripping common ModuleDict prefixes/suffixes.
            clean_name = coef_name.replace("coef_dict.", "").replace(".coef", "")
            for idx, (mean_val, std_val) in enumerate(zip(mean_np, std_np)):
                if np.isnan(std_val) or std_val == 0:
                    # Missing/degenerate std errors -> keep z/p as NaN rather than divide-by-zero.
                    z_value = float("nan")
                    p_value = float("nan")
                else:
                    z_value = mean_val / std_val
                    # Two-sided normal approximation for p-value.
                    p_value = (1 - norm.cdf(abs(z_value))) * 2
                    if np.isfinite(p_value) and p_value < MIN_REPORTED_PVALUE:
                        # Avoid printing "0" due to floating-point underflow.
                        p_value = MIN_REPORTED_PVALUE
                rows.append(
                    {
                        "Coefficient": f"{clean_name}_{idx}",
                        "Estimation": float(mean_val),
                        "Std. Err.": float(std_val),
                        "z-value": float(z_value),
                        "Pr(>|z|)": float(p_value),
                    }
                )
        report = pd.DataFrame(rows).set_index("Coefficient")
        report["Significance"] = ""
        # Common "star" thresholds used by R-style regression summaries.
        report.loc[report["Pr(>|z|)"] < 0.001, "Significance"] = "***"
        report.loc[
            (report["Pr(>|z|)"] >= 0.001) & (report["Pr(>|z|)"] < 0.01), "Significance"
        ] = "**"
        report.loc[
            (report["Pr(>|z|)"] >= 0.01) & (report["Pr(>|z|)"] < 0.05), "Significance"
        ] = "*"
        report.loc[
            (report["Pr(>|z|)"] >= 0.05) & (report["Pr(>|z|)"] < 0.1), "Significance"
        ] = "."
        return report

