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
        """
        if dataset_train is None:
            raise ValueError("dataset_train must be provided.")

        backend_key = backend.lower()
        if backend_key not in {"torch", "lightning"}:
            raise ValueError(f"Unsupported backend '{backend}'. Expected 'torch' or 'lightning'.")

        device_str = str(device) if device is not None else None
        dataset_train = self._maybe_move_dataset(dataset_train, device_str)
        dataset_val = self._maybe_move_dataset(dataset_val, device_str)
        dataset_test = self._maybe_move_dataset(dataset_test, device_str)
        if device_str is not None:
            self.to(device_str)

        start_time = time.perf_counter()
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
        else:
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

        elapsed_time = time.perf_counter() - start_time

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
        )

    def run(self, *args, **kwargs):
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
        optimizer_kwargs = backend_kwargs.pop("optimizer_kwargs", {}) or {}
        scheduler_kwargs = backend_kwargs.pop("scheduler_kwargs", None)
        scheduler_class = backend_kwargs.pop("scheduler_class", torch.optim.lr_scheduler.StepLR)

        if backend_kwargs:
            warnings.warn(
                f"Unused torch backend kwargs: {sorted(backend_kwargs.keys())}",
                stacklevel=2,
            )

        optimizer_class = getattr(torch.optim, model_optimizer, None)
        if optimizer_class is None:
            raise ValueError(
                f"Optimizer '{model_optimizer}' is not available in torch.optim."
            )

        optimizer_args = dict(lr=learning_rate, **optimizer_kwargs)
        if optimizer_class is torch.optim.LBFGS:
            optimizer_args.setdefault("max_iter", 20)
            optimizer_args.setdefault("history_size", 10)
        optimizer = optimizer_class(self.parameters(), **optimizer_args)

        scheduler = None
        if scheduler_class is not None:
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
            report_frequency = max(num_epochs // 10, 1)

        training_device = self._infer_active_device(device_hint)

        for epoch in range(1, num_epochs + 1):
            epoch_loss = 0.0
            num_obs = 0
            for batch in train_loader:
                batch = self._move_batch_to_device(batch, training_device)
                targets = self._get_item_index_from_batch(batch)
                if torch.is_tensor(targets):
                    targets = targets.to(training_device)

                def closure():
                    optimizer.zero_grad()
                    loss_val = self.loss(batch, targets)
                    loss_val.backward()
                    return loss_val

                if optimizer_class is torch.optim.LBFGS:
                    loss_value = closure()
                    optimizer.step(closure)
                else:
                    loss_value = self.loss(batch, targets)
                    optimizer.zero_grad()
                    loss_value.backward()
                    optimizer.step()

                epoch_loss += float(loss_value.detach().item())
                num_obs += int(targets.numel())

            if scheduler is not None:
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
                self.log("train_loss", loss, prog_bar=False, batch_size=int(targets.numel()))
                return loss

            def validation_step(self, batch, batch_idx):
                targets = self._get_targets(batch)
                ll = -self.model.negative_log_likelihood(batch, targets, is_train=False)
                self.log("val_ll", ll, prog_bar=True, batch_size=int(targets.numel()))

            def test_step(self, batch, batch_idx):
                targets = self._get_targets(batch)
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
            callbacks = callbacks + [
                EarlyStopping(monitor="val_ll", mode="max", patience=10, min_delta=0.001)
            ]
        trainer_defaults["callbacks"] = callbacks

        accelerator = trainer_kwargs.pop("accelerator", None)
        if accelerator is None:
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

        epochs_ran = getattr(trainer, "current_epoch", num_epochs - 1) + 1

        self.eval()
        inference_device = self._infer_active_device(device_hint)
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
    def _get_item_index_from_batch(self, batch: BatchType) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError(
            "_get_item_index_from_batch must be implemented by subclasses."
        )

    def _maybe_move_dataset(
        self,
        dataset: Optional[DatasetType],
        device: Optional[Union[str, torch.device]],
    ) -> Optional[DatasetType]:
        if dataset is None or device is None:
            return dataset
        return dataset.to(device)

    def _infer_active_device(
        self, device: Optional[Union[str, torch.device]]
    ) -> torch.device:
        if device is not None:
            return torch.device(device)
        model_device = getattr(self, "device", None)
        if model_device is not None:
            return torch.device(model_device)
        try:
            return next(self.parameters()).device
        except (StopIteration, AttributeError):
            return torch.device("cpu")

    def _move_batch_to_device(
        self, batch: BatchType, device: Optional[torch.device]
    ) -> BatchType:
        if device is None:
            return batch
        if isinstance(batch, ChoiceDataset):
            return batch.to(device)
        if isinstance(batch, dict):
            for key, value in batch.items():
                if hasattr(value, "to"):
                    batch[key] = value.to(device)
            return batch
        return batch

    def _compute_dataset_log_likelihood(
        self,
        dataset: DatasetType,
        *,
        batch_size: int,
        num_workers: int,
        device: Optional[Union[str, torch.device]] = None,
    ) -> float:
        self.eval()
        eval_device = self._infer_active_device(device)
        dataloader = create_data_loader(
            dataset,
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
    ) -> EstimationOutput:
        dataset_for_std = self._clone_dataset_for_std(dataset_train)
        dataset_for_std = self._maybe_move_dataset(dataset_for_std, device_hint)
        mean_dict = {name: param.detach().clone() for name, param in self.named_parameters()}

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

        std_error_failed = False
        try:
            std_dict = self._compute_parameter_std(model_clone, dataset_for_std)
        except (RuntimeError, torch.linalg.LinAlgError, IndexError) as err:
            warnings.warn(
                f"Failed to compute parameter standard errors due to: {err}. "
                "The regression table will omit Std. Err./z/p values for this run. "
                "Try training longer or adjusting regularization to avoid a singular Hessian.",
                RuntimeWarning,
                stacklevel=2,
            )
            std_dict = None
            std_error_failed = True

        report = self._build_coefficient_report(mean_dict, std_dict)

        train_ll = backend_result.get("train_ll")
        val_ll = backend_result.get("val_ll")
        test_ll = backend_result.get("test_ll")

        def _fmt_3digits(value: Any) -> str:
            """Format numbers with 3 digits, using scientific notation for tiny magnitudes."""
            if value is None:
                return "None"
            try:
                x = float(value)
            except (TypeError, ValueError):
                return str(value)
            if not np.isfinite(x):
                return str(x)
            if x != 0.0 and abs(x) < SCIENTIFIC_NOTATION_THRESHOLD:
                return f"{x:.3e}"
            return f"{x:.3f}"

        print("=" * 20, "model results", "=" * 20)
        print(
            "Log-likelihood: "
            f"[Training] {_fmt_3digits(train_ll)}, "
            f"[Validation] {_fmt_3digits(val_ll)}, "
            f"[Test] {_fmt_3digits(test_ll)}\n"
        )
        # Display convention: print "< 2e-16" for extremely small p-values, while keeping the
        # underlying stored values numeric in the returned `coef_summary`.
        z_cutoff = float(norm.isf(MIN_REPORTED_PVALUE / 2.0))
        report_to_print = report.copy()
        if "Pr(>|z|)" in report_to_print.columns and "z-value" in report_to_print.columns:
            z_abs = report_to_print["z-value"].astype(float).abs()
            lt_mask = np.isfinite(z_abs.to_numpy()) & (z_abs.to_numpy() > z_cutoff)
            # Format z-values and p-values for display (3 digits; scientific for tiny magnitudes).
            report_to_print["z-value"] = report_to_print["z-value"].map(_fmt_3digits)

            pvals = report_to_print["Pr(>|z|)"].astype(float).to_numpy()
            report_to_print["Pr(>|z|)"] = [
                (f"< {MIN_REPORTED_PVALUE:g}" if bool(is_lt) else _fmt_3digits(p))
                for p, is_lt in zip(pvals, lt_mask)
            ]

        print(report_to_print.to_markdown())
        print("Significance codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")

        return EstimationOutput(
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

    def _clone_dataset_for_std(self, dataset: DatasetType) -> DatasetType:
        if dataset is None:
            raise ValueError("dataset_train is required for standard error calculation.")
        if hasattr(dataset, "clone"):
            try:
                return dataset.clone()
            except Exception:
                pass
        return deepcopy(dataset)

    def _compute_parameter_std(self, model_clone, dataset_for_std):
        from torch_choice.model.conditional_logit_model import ConditionalLogitModel
        from torch_choice.model.nested_logit_model import NestedLogitModel

        if isinstance(self, ConditionalLogitModel):
            def nll_loss(model):
                y_pred = model(dataset_for_std)
                item_index = dataset_for_std.item_index.clone()
                if getattr(model, "model_outside_option", False):
                    outside_mask = item_index == -1
                    item_index[outside_mask] = model.num_items
                return F.cross_entropy(y_pred, item_index, reduction="sum")
        elif isinstance(self, NestedLogitModel):
            def nll_loss(model):
                indices = torch.arange(len(dataset_for_std))
                data = dataset_for_std[indices]
                return model.negative_log_likelihood(data, data["item"].item_index)
        else:
            raise TypeError("Unsupported model type for standard error computation.")

        return parameter_std(model_clone, nll_loss)

    def _build_coefficient_report(
        self,
        mean_dict: Dict[str, torch.Tensor],
        std_dict: Optional[Dict[str, torch.Tensor]],
    ) -> pd.DataFrame:
        rows = []
        for coef_name, mean_tensor in mean_dict.items():
            mean_np = mean_tensor.detach().cpu().numpy().reshape(-1)
            if std_dict is not None and coef_name in std_dict:
                std_np = std_dict[coef_name].detach().cpu().numpy().reshape(-1)
            else:
                std_np = np.full_like(mean_np, np.nan, dtype=float)

            clean_name = coef_name.replace("coef_dict.", "").replace(".coef", "")
            for idx, (mean_val, std_val) in enumerate(zip(mean_np, std_np)):
                if np.isnan(std_val) or std_val == 0:
                    z_value = float("nan")
                    p_value = float("nan")
                else:
                    z_value = mean_val / std_val
                    p_value = (1 - norm.cdf(abs(z_value))) * 2
                    if np.isfinite(p_value) and p_value < MIN_REPORTED_PVALUE:
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

