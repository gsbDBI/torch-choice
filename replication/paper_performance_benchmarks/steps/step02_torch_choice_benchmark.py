"""Torch-Choice benchmark runner."""

from __future__ import annotations

import argparse
import os
import platform
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Union

import pandas as pd
import torch

from torch_choice.data import ChoiceDataset
from torch_choice.data import utils as data_utils
from torch_choice.model import ConditionalLogitModel
from torch_choice.model.nested_logit_model import NestedLogitModel


DEVICE = "cpu"


def _set_device(device_arg: str) -> str:
    global DEVICE
    if device_arg == "auto":
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        DEVICE = device_arg
    return DEVICE


def _auto_batch_size() -> int:
    """Return a batch size appropriate for available GPU memory.

    Empirically tested on RTX 3090 (24GB) with num_records_experiment_large (100K records × 500 items):
    - batch_size=131,072: ~8,087 MB peak (OOM at 262,144)
    - batch_size=65,536: ~4,080 MB peak
    - batch_size=32,768: ~2,180 MB peak

    Thresholds based on 70% safety margin of empirically determined limits.

    Environment Variables:
        GPU_MEM_LIMIT: If set, limits the GPU memory considered for batch size selection
                      (in GB). Useful when running other concurrent GPU workloads.
                      Example: GPU_MEM_LIMIT=10 will select batch size as if GPU had 10GB.
    """
    if not torch.cuda.is_available():
        return -1  # CPU: full batch is fine

    props = torch.cuda.get_device_properties(0)
    actual_mem_gb = props.total_memory / (1024**3)
    gpu_name = props.name

    # Check for user-defined memory limit
    mem_limit_env = os.environ.get("GPU_MEM_LIMIT")
    if mem_limit_env:
        try:
            total_mem_gb = float(mem_limit_env)
            print(f"[Auto batch size] GPU: {gpu_name} (actual: {actual_mem_gb:.1f} GB, "
                  f"limit: {total_mem_gb:.1f} GB via GPU_MEM_LIMIT)")
        except ValueError:
            print(f"[Auto batch size] Warning: Invalid GPU_MEM_LIMIT='{mem_limit_env}', "
                  f"using actual GPU memory: {actual_mem_gb:.1f} GB")
            total_mem_gb = actual_mem_gb
    else:
        total_mem_gb = actual_mem_gb

    # Empirically determined thresholds based on stress testing
    # OOM at batch_size=262,144 on 24GB GPU with 100K records × 500 items
    # Using 70% safety margin of max successful batch (131,072)
    if total_mem_gb < 8:
        batch_size = 8192  # ~600 MB, safe for 8GB GPUs
    elif total_mem_gb < 12:
        batch_size = 16384  # ~1,100 MB, safe for 10-12GB GPUs
    elif total_mem_gb < 16:
        batch_size = 32768  # ~2,200 MB, safe for 16GB GPUs
    elif total_mem_gb < 22:
        batch_size = 65536  # ~4,100 MB, safe for 20-22GB GPUs (e.g., RTX 4000 Ada)
    else:
        batch_size = 131072  # ~8,100 MB, for 24GB+ GPUs (e.g., RTX 3090/4090)

    effective = f"{batch_size:,}" if batch_size > 0 else "full batch"
    print(f"[Auto batch size] GPU: {gpu_name} ({total_mem_gb:.1f} GB) -> batch_size={effective}")
    return batch_size


def load_dataset(
    data_path: Union[str, Path],
    filename: str,
    session_limit: int | None = None,
    num_params: int | None = None,
) -> ChoiceDataset:
    """Load a dataset and optionally trim sessions/latent dimensions."""
    ds = torch.load(Path(data_path) / filename, map_location=DEVICE, weights_only=False)
    if num_params is not None:
        ds.user_latents = ds.user_latents[:, :num_params]
        ds.item_latents = ds.item_latents[:, :num_params]
    if session_limit is not None:
        ds = ds[ds.session_index < session_limit]
    return ds


def run_model(
    model: Union[ConditionalLogitModel, NestedLogitModel],
    dataset: ChoiceDataset,
    batch_size: int,
    learning_rate: float,
    num_epochs: int,
    optimizer_name: str,
) -> tuple[float, float, int]:
    """Train the model with early stopping; return best loss, time, epochs run."""
    from copy import deepcopy
    from time import time
    from tqdm import tqdm

    model = deepcopy(model).to(DEVICE)
    dataset = dataset.to(DEVICE)
    data_loader = data_utils.create_data_loader(dataset, batch_size=batch_size, shuffle=True)

    optim_cls = {
        "SGD": torch.optim.SGD,
        "Adagrad": torch.optim.Adagrad,
        "Adadelta": torch.optim.Adadelta,
        "Adam": torch.optim.Adam,
    }.get(optimizer_name)
    if optim_cls is None:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    optimizer = optim_cls(model.parameters(), lr=learning_rate)

    not_improved_tolerance = 50
    best_loss = float("inf")
    not_improved_count = 0
    epochs_run = 0

    start_time = time()
    model.train()
    for e in tqdm(range(1, num_epochs + 1), desc=f"Training on {DEVICE}", leave=False):
        total_loss = 0.0
        for batch in data_loader:
            item_index = batch["item"].item_index if isinstance(model, NestedLogitModel) else batch.item_index
            loss = model.loss(batch, item_index)
            total_loss += float(loss.detach().item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        current_loss = total_loss
        if current_loss < best_loss:
            best_loss = current_loss
            not_improved_count = 0
        else:
            not_improved_count += 1

        if not_improved_count >= not_improved_tolerance:
            epochs_run = e
            break
        epochs_run = e

    time_taken = float(time() - start_time)
    return best_loss, time_taken, epochs_run


def run_experiment(
    args: argparse.Namespace,
    task_config: Dict,
    run_configs: Dict,
) -> pd.DataFrame:
    record_list = []
    for seed in range(args.num_seeds):
        torch.manual_seed(seed)
        for formula in task_config["formulas"]:
            for value in task_config["values"]:
                dataset = task_config["loader"](args.data_path, value)
                model = ConditionalLogitModel(formula=formula, dataset=dataset, num_items=dataset.num_items)
                best_loss, time_taken, epochs_run = run_model(
                    model=model,
                    dataset=dataset,
                    batch_size=run_configs["batch_size"],
                    learning_rate=run_configs["learning_rate"],
                    num_epochs=run_configs["num_epochs"],
                    optimizer_name=run_configs["optimizer"],
                )
                record_list.append(
                    {
                        "num_records": len(dataset),
                        "time": time_taken,
                        "formula": formula,
                        "seed": seed,
                        "best_loss": best_loss,
                        task_config["key"]: value,
                        "epochs_run": epochs_run,
                        "dataset": str(dataset),
                    }
                )
                del model, dataset
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    __import__("gc").collect()

    record = pd.DataFrame(record_list)
    for key, val in build_sys_info(args).items():
        record[key] = val
    return record


def build_sys_info(args: argparse.Namespace) -> Dict:
    return {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "torch_choice_version": __import__("torch_choice").__version__,
        "device": DEVICE,
        **args.__dict__,
        "cpu_name": platform.processor(),
        "cpu_count": os.cpu_count(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
    }


def build_experiment_configs() -> Dict[str, Dict]:
    formula_list = [
        "(user_latents|item) + (item_latents|constant)",
        "(user_latents|item)",
        "(item_latents|constant)",
    ]
    return {
        "num_records_experiment_small": {
            "key": "sample_size",
            "values": [3_000, 5_000, 7_000, 10_000, 30_000, 50_000, 70_000, 100_000],
            "formulas": formula_list,
            "loader": lambda dp, val: load_dataset(
                data_path=dp,
                filename="simulated_choice_data_num_records_experiment_small_seed_42.pt",
                session_limit=val,
                num_params=None,
            ),
        },
        "num_records_experiment_large": {
            "key": "sample_size",
            "values": [3_000, 5_000, 7_000, 10_000, 30_000, 50_000, 70_000, 100_000],
            "formulas": formula_list,
            "loader": lambda dp, val: load_dataset(
                data_path=dp,
                filename="simulated_choice_data_full_dataset_seed_42.pt",
                session_limit=val,
                num_params=None,
            ),
        },
        "num_params_experiment_small": {
            "key": "num_params",
            "values": [3, 5, 10, 15, 20, 30],
            "formulas": formula_list,
            "loader": lambda dp, val: load_dataset(
                data_path=dp,
                filename="simulated_choice_data_num_params_experiment_small_seed_42.pt",
                session_limit=None,
                num_params=val,
            ),
        },
        "num_params_experiment_large": {
            "key": "num_params",
            "values": [30, 1, 5, 10, 15, 20],
            "formulas": formula_list,
            "loader": lambda dp, val: load_dataset(
                data_path=dp,
                filename="simulated_choice_data_full_dataset_seed_42.pt",
                session_limit=20_000,
                num_params=val,
            ),
        },
        "num_items_experiment_small": {
            "key": "num_items",
            "values": [10, 20, 30, 50, 100, 150, 200],
            "formulas": formula_list,
            "loader": lambda dp, val: load_dataset(
                data_path=dp,
                filename=f"simulated_choice_data_num_items_experiment_small_{val}_items_seed_42.pt",
                session_limit=None,
                num_params=None,
            ),
        },
        "num_items_experiment_large": {
            "key": "num_items",
            "values": [10, 20, 30, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
            "formulas": formula_list,
            "loader": lambda dp, val: load_dataset(
                data_path=dp,
                filename=f"simulated_choice_data_num_items_experiment_large_{val}_items_seed_42.pt",
                session_limit=None,
                num_params=None,
            ),
        },
    }


def run_all(args: argparse.Namespace) -> None:
    exp_configs = build_experiment_configs()
    if getattr(args, "smoke_test", False):
        # Keep only the small tasks and restrict each to a single representative value.
        keep = ["num_records_experiment_small", "num_params_experiment_small", "num_items_experiment_small"]
        exp_configs = {k: exp_configs[k] for k in keep if k in exp_configs}
        if "num_records_experiment_small" in exp_configs:
            exp_configs["num_records_experiment_small"]["values"] = [3000, 5000]
        if "num_params_experiment_small" in exp_configs:
            exp_configs["num_params_experiment_small"]["values"] = [3, 5]
        if "num_items_experiment_small" in exp_configs:
            exp_configs["num_items_experiment_small"]["values"] = [10, 20]
    if args.experiment_name != "all":
        if args.experiment_name not in exp_configs:
            raise ValueError(f"Experiment '{args.experiment_name}' not found.")
        exp_configs = {args.experiment_name: exp_configs[args.experiment_name]}

    os.makedirs(args.output_path, exist_ok=True)
    for task, task_config in exp_configs.items():
        run_configs = {
            "num_epochs": args.num_epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "optimizer": args.optimizer,
        }
        df_record = run_experiment(args, task_config, run_configs)
        df_record["task"] = task
        out_file = Path(args.output_path) / f"torch_choice_performance_{task}.csv"
        df_record.to_csv(out_file, index=False)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_arg_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Torch-Choice benchmarks.", add_help=add_help)
    parser.add_argument("--data-path", type=Path, required=True, help="Path to dataset folder (.pt files).")
    parser.add_argument("--output-path", type=Path, required=True, help="Where to write benchmark CSVs.")
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="all",
        help="Name of experiment to run (or 'all').",
    )
    parser.add_argument("--device", type=str, default="auto", help="Device to run on (auto|cpu|cuda).")
    parser.add_argument("--num-seeds", type=int, default=5, help="Number of seeds.")
    parser.add_argument("--num-epochs", type=int, default=50_000, help="Max epochs.")
    parser.add_argument("--learning-rate", type=float, default=0.03, help="Learning rate.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size; -1 for full batch, omit to auto-detect based on GPU memory.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a minimal configuration (single value per task; small tasks only).",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="Adam",
        choices=["SGD", "Adagrad", "Adadelta", "Adam"],
        help="Optimizer to use.",
    )
    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    _set_device(args.device)

    # Resolve batch size: None means auto-detect
    if args.batch_size is None:
        args.batch_size = _auto_batch_size()

    run_all(args)


if __name__ == "__main__":
    main()

