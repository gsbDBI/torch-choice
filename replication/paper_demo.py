#!/usr/bin/env python3
"""Console-friendly replication script for the Torch-Choice paper demo."""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
import time
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import BatchSampler, SequentialSampler

import torch_choice
from torch_choice import run
from torch_choice.data import ChoiceDataset, JointDataset, load_mode_canada_dataset
from torch_choice.model import ConditionalLogitModel
from torch_choice.utils.easy_data_wrapper import EasyDatasetWrapper

warnings.filterwarnings("ignore")
SECTION_LINE = "=" * 80
SUBSECTION_LINE = "-" * 80


def print_section(title: str) -> None:
    print(f"\n{SECTION_LINE}\n{title}\n{SECTION_LINE}")


def print_subsection(title: str) -> None:
    print(f"\n{SUBSECTION_LINE}\n{title}\n{SUBSECTION_LINE}")


def print_dataframe(title: str, frame: pd.DataFrame, max_rows: int = 5) -> None:
    print_subsection(title)
    with pd.option_context("display.max_rows", max_rows, "display.max_columns", None):
        print(frame.head(max_rows).to_string(index=False))


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"[Setup] Random seed set to {seed}.")


def patch_choice_dataset_helpers() -> None:
    if getattr(ChoiceDataset, "_replication_patch_applied", False):
        return

    @classmethod
    def _from_dict_with_counts(cls, dictionary):
        base = dict(dictionary)
        constructor_kwargs = {k: v for k, v in base.items() if not k.startswith("_")}
        if "_num_users" in base and "num_users" not in constructor_kwargs:
            constructor_kwargs["num_users"] = base["_num_users"]
        if "_num_items" in base and "num_items" not in constructor_kwargs:
            constructor_kwargs["num_items"] = base["_num_items"]
        if "_num_sessions" in base and "num_sessions" not in constructor_kwargs:
            constructor_kwargs["num_sessions"] = base["_num_sessions"]
        dataset = cls(**constructor_kwargs)
        for key, item in base.items():
            if key in {"num_users", "num_items", "num_sessions"}:
                continue
            setattr(dataset, key, item)
        return dataset

    ChoiceDataset._from_dict = _from_dict_with_counts
    ChoiceDataset._replication_patch_applied = True


def launch_tensorboard(logdir: Path, port: int) -> None:
    print_section("TensorBoard")
    cmd = ["tensorboard", "--logdir", str(logdir), "--port", str(port)]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        print(f"[TensorBoard] Started with PID {proc.pid}.")
        print(f"[TensorBoard] Open http://localhost:{port}/ in your browser.")
        print("[TensorBoard] To stop TensorBoard when you're done:")
        print("  - Same shell: press Ctrl+C")
        print(f"  - macOS/Linux: run `kill {proc.pid}`")
        print(f"  - Windows (PowerShell): `Stop-Process -Id {proc.pid}`")
        print(f"  - Windows (Command Prompt): `taskkill /PID {proc.pid} /F`")
        print("  - Windows: end the TensorBoard process via Task Manager")
    except FileNotFoundError:
        print("[TensorBoard] `tensorboard` command not found.")
        print("Install TensorBoard or run it manually with:")
        print(f"  {' '.join(cmd)}")
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replicate the Torch-Choice paper demo as a single Python script.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--skip-training", action="store_true", help="Skip conditional logit training.")
    parser.add_argument("--num-epochs", type=int, default=1000, help="Epochs for conditional logit training.")
    parser.add_argument(
        "--tensorboard-logdir",
        type=Path,
        default=Path("lightning_logs"),
        help="Directory for PyTorch Lightning logs.",
    )
    parser.add_argument("--tensorboard-port", type=int, default=6006, help="TensorBoard port.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    replication_dir = Path(__file__).resolve().parent
    patch_choice_dataset_helpers()

    # === Package versions (matches notebook cell order) ============================================
    print_section("Package Versions")
    print(f"np.__version__={np.__version__}")
    print(f"pd.__version__={pd.__version__}")
    print(f"torch.__version__={torch.__version__}")
    print(f"torch_choice.__version__={torch_choice.__version__}")

    # === Set random seed =========================================================================
    set_seed(42)

    # === Data Structure ==========================================================================
    csv_path = replication_dir / "car_choice.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Cannot find car_choice.csv at {csv_path}.")
    car_choice = pd.read_csv(csv_path)
    print_section("Data Structure")
    print_dataframe("car_choice.head()", car_choice)

    # Method 1: observables derived from columns.
    print_subsection("Adding Observables, Method 1: Columns")
    wrapper_from_columns = EasyDatasetWrapper(
        main_data=car_choice,
        purchase_record_column="record_id",
        choice_column="purchase",
        item_name_column="car",
        user_index_column="consumer_id",
        session_index_column="session_id",
        user_observable_columns=["gender", "income"],
        item_observable_columns=["speed"],
        session_observable_columns=["discount"],
        itemsession_observable_columns=["price"],
    )
    wrapper_from_columns.summary()
    dataset_from_columns = wrapper_from_columns.choice_dataset
    print(f"[EasyDatasetWrapper] dataset_from_columns={dataset_from_columns}")

    # Method 2: provide observables as separate data frames.
    print_subsection("Adding Observables, Method 2: Separate DataFrames")
    gender = car_choice.groupby("consumer_id")["gender"].first().reset_index()
    income = car_choice.groupby("consumer_id")["income"].first().reset_index()
    gender_and_income = car_choice.groupby("consumer_id")[["gender", "income"]].first().reset_index()
    speed = car_choice.groupby("car")["speed"].first().reset_index()
    discount = car_choice.groupby("session_id")["discount"].first().reset_index()
    price = car_choice[["car", "session_id", "price"]]
    price = price.pivot(index="car", columns="session_id", values="price").melt(ignore_index=False).reset_index()
    wrapper_from_frames = EasyDatasetWrapper(
        main_data=car_choice,
        purchase_record_column="record_id",
        choice_column="purchase",
        item_name_column="car",
        user_index_column="consumer_id",
        session_index_column="session_id",
        user_observable_data={"gender": gender, "income": income},
        item_observable_data={"speed": speed},
        session_observable_data={"discount": discount},
        itemsession_observable_data={"price": price},
    )
    assert wrapper_from_frames.choice_dataset == dataset_from_columns
    print("[EasyDatasetWrapper] Method 2 matches Method 1.")

    # Method 3: mix columns and data frames.
    print_subsection("Adding Observables, Method 3: Mixed Inputs")
    wrapper_mixed = EasyDatasetWrapper(
        main_data=car_choice,
        purchase_record_column="record_id",
        choice_column="purchase",
        item_name_column="car",
        user_index_column="consumer_id",
        session_index_column="session_id",
        user_observable_data={"gender": gender, "income": income},
        item_observable_data={"speed": speed},
        session_observable_data={"discount": discount},
        itemsession_observable_columns=["price"],
    )
    assert wrapper_mixed.choice_dataset == dataset_from_columns
    print("[EasyDatasetWrapper] Mixed input method also matches.")

    # === Constructing a Choice Dataset from tensors ===============================================
    print_section("Constructing a Choice Dataset from Tensors")
    N = 10_000
    num_users = 10
    num_items = 4
    num_sessions = 500
    user_obs = torch.randn(num_users, 128)
    item_obs = torch.randn(num_items, 64)
    useritem_obs = torch.randn(num_users, num_items, 32)
    session_obs = torch.randn(num_sessions, 10)
    itemsession_obs = torch.randn(num_sessions, num_items, 12)
    usersession_obs = torch.randn(num_users, num_sessions, 10)
    usersessionitem_obs = torch.randn(num_users, num_sessions, num_items, 8)
    item_index = torch.LongTensor(np.random.choice(num_items, size=N))
    user_index = torch.LongTensor(np.random.choice(num_users, size=N))
    session_index = torch.LongTensor(np.random.choice(num_sessions, size=N))
    item_availability = torch.ones(num_sessions, num_items).bool()
    dataset = ChoiceDataset(
        item_index=item_index,
        num_items=num_items,
        user_index=user_index,
        num_users=num_users,
        session_index=session_index,
        num_sessions=num_sessions,
        item_availability=item_availability,
        user_obs=user_obs,
        item_obs=item_obs,
        session_obs=session_obs,
        itemsession_obs=itemsession_obs,
        useritem_obs=useritem_obs,
        usersession_obs=usersession_obs,
        usersessionitem_obs=usersessionitem_obs,
    )
    print(f"[Synthetic Dataset] dataset={dataset}")

    # === Functionalities of the Choice Dataset ====================================================
    print_section("Choice Dataset Functionalities")
    print(f"dataset.num_users={dataset.num_users}")
    print(f"dataset.num_items={dataset.num_items}")
    print(f"dataset.num_sessions={dataset.num_sessions}")
    print(f"len(dataset)={len(dataset)}")

    print_subsection("Cloning Behavior")
    print(f"dataset.item_index[:10]={dataset.item_index[:10]}")
    try:
        dataset_cloned = dataset.clone()
    except AssertionError as err:
        print(f"[Clone Warning] dataset.clone() raised '{err}'. Falling back to deepcopy.")
        dataset_cloned = deepcopy(dataset)
    dataset_cloned.item_index = torch.full_like(dataset.item_index, 99)
    print(f"dataset_cloned.item_index[:10]={dataset_cloned.item_index[:10]}")
    print(f"dataset.item_index[:10]={dataset.item_index[:10]}")

    print_subsection("Device Movement")
    print(f"dataset.device={dataset.device}")
    print(f"dataset.user_index.device={dataset.user_index.device}")
    print(f"dataset.session_index.device={dataset.session_index.device}")
    if torch.cuda.is_available():
        dataset_cuda = dataset.to("cuda")
        print(f"dataset_cuda.device={dataset_cuda.device}")
        print(f"dataset_cuda.item_index.device={dataset_cuda.item_index.device}")
        print(f"dataset_cuda.user_index.device={dataset_cuda.user_index.device}")
        print(f"dataset_cuda.session_index.device={dataset_cuda.session_index.device}")
        dataset_cuda._check_device_consistency()
        dataset = dataset_cuda.to("cpu")

    print_subsection("Observables Dictionary Shapes")
    for key, value in dataset.x_dict.items():
        if torch.is_tensor(value):
            print(f"dict.{key}.shape={tuple(value.shape)}")

    print_subsection("Mini-batch Extraction")
    indices = torch.from_numpy(np.random.choice(len(dataset), size=5, replace=False)).long()
    subset = dataset[indices]
    print(f"indices={indices}")
    print(f"subset={subset}")
    print(f"dataset.item_index[indices]={dataset.item_index[indices]}")
    subset.item_index += 1
    print("[Mini-batch] After modifying subset.item_index, original dataset remains unchanged.")
    print(f"subset.item_index={subset.item_index}")
    print(f"dataset.item_index[indices]={dataset.item_index[indices]}")
    subset.item_obs += 1
    print("[Mini-batch] subset.item_obs[0, 0] vs dataset.item_obs[0, 0]:")
    print(f"subset.item_obs[0, 0]={subset.item_obs[0, 0]}")
    print(f"dataset.item_obs[0, 0]={dataset.item_obs[0, 0]}")
    print(f"id(subset.item_index)={id(subset.item_index)}")
    print(f"id(dataset.item_index[indices])={id(dataset.item_index[indices])}")

    print_subsection("JointDataset Demonstration")
    item_level_dataset = dataset.clone()
    nest_level_dataset = dataset.clone()
    joint_dataset = JointDataset(item=item_level_dataset, nest=nest_level_dataset)
    print(f"joint_dataset={joint_dataset}")

    print_subsection("DataLoader Consistency Checks")
    batch_size = 32
    sampler = BatchSampler(SequentialSampler(dataset), batch_size=batch_size, drop_last=False)
    dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler, collate_fn=lambda x: x[0])
    item_obs_all = item_obs.view(1, num_items, -1).expand(len(dataset), -1, -1)
    item_index_all = item_index
    print(f"item_obs.shape={item_obs.shape}")
    print(f"item_obs_all.shape={item_obs_all.shape}")
    for i, batch in enumerate(dataloader):
        first, last = i * batch_size, min(len(dataset), (i + 1) * batch_size)
        idx = torch.arange(first, last)
        assert torch.all(item_obs_all[idx, :, :] == batch.x_dict["item_obs"])
        assert torch.all(item_index_all[idx] == batch.item_index)
        print(f"batch.x_dict['item_obs'].shape={batch.x_dict['item_obs'].shape}")
        break
    print_subsection("dataset.x_dict Shapes (post DataLoader check)")
    for key, value in dataset.x_dict.items():
        if torch.is_tensor(value):
            print(f"dict.{key}.shape={tuple(value.shape)}")
    print(f"dataset.__len__() returns {len(dataset)}")

    # === Conditional Logit Model ==============================================================
    if args.skip_training:
        print_section("Conditional Logit Model Training Skipped")
        return

    print_section("Conditional Logit Model")
    dataset_mode_canada = load_mode_canada_dataset()
    print(f"[Mode Canada] dataset={dataset_mode_canada}")

    print_subsection("Formula-based Specification")
    model = ConditionalLogitModel(
        formula="(itemsession_cost_freq_ovt|constant) + (session_income|item) + (itemsession_ivt|item-full) + (intercept|item)",
        dataset=dataset_mode_canada,
        num_items=4,
    )

    print_subsection("Dictionary-based Specification")
    model = ConditionalLogitModel(
        coef_variation_dict={
            "itemsession_cost_freq_ovt": "constant",
            "session_income": "item",
            "itemsession_ivt": "item-full",
            "intercept": "item",
        },
        num_param_dict={
            "itemsession_cost_freq_ovt": 3,
            "session_income": 1,
            "itemsession_ivt": 1,
            "intercept": 1,
        },
        num_items=4,
    )

    print_subsection("Dictionary Specification with Regularization")
    model = ConditionalLogitModel(
        coef_variation_dict={
            "itemsession_cost_freq_ovt": "constant",
            "session_income": "item",
            "itemsession_ivt": "item-full",
            "intercept": "item",
        },
        num_param_dict={
            "itemsession_cost_freq_ovt": 3,
            "session_income": 1,
            "itemsession_ivt": 1,
            "intercept": 1,
        },
        num_items=4,
        regularization="L1",
        regularization_weight=0.5,
    )

    print_subsection("Training via torch_choice.run")
    args.tensorboard_logdir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    run(
        model,
        dataset_mode_canada,
        batch_size=-1,
        learning_rate=0.01,
        num_epochs=args.num_epochs,
        model_optimizer="LBFGS",
        default_root_dir=str(args.tensorboard_logdir),
    )
    duration = time.perf_counter() - start
    print(f"[Training] Completed in {duration:.2f} seconds.")
    launch_tensorboard(args.tensorboard_logdir, args.tensorboard_port)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Exit] Interrupted by user.")
        sys.exit(1)

