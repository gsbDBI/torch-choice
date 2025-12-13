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
from torch_choice.data import ChoiceDataset, JointDataset, load_mode_canada_dataset, utils
from torch_choice.model import ConditionalLogitModel, NestedLogitModel
from torch_choice.utils.easy_data_wrapper import EasyDatasetWrapper

warnings.filterwarnings("ignore")
SECTION_LINE = "=" * 80
SUBSECTION_LINE = "-" * 80
HOUSE_COOLING_URL = (
    "https://raw.githubusercontent.com/gsbDBI/torch-choice/main/tutorials/public_datasets/HC.csv"
)
HOUSE_COOLING_ITEM_NAMES = ["ec", "ecc", "er", "erc", "gc", "gcc", "hpc"]
HOUSE_COOLING_FEATURE_COLUMNS = [
    "ich",
    "och",
    "icca",
    "occa",
    "inc.room",
    "inc.cooling",
    "int.cooling",
]


def build_house_cooling_joint_dataset() -> tuple[JointDataset, dict[int, list[int]], dict[str, int]]:
    """Build the joint dataset + default two-nest mapping for the House Cooling example.

    Returns:
        - JointDataset with `nest` (choice indices) and `item` (choice indices + `price_obs`) datasets.
        - nest_to_item mapping (nest_id -> list[item_id]) aligned with HOUSE_COOLING_ITEM_NAMES encoding.
        - summary dict with basic dataset stats.
    """
    replication_dir = Path(__file__).resolve().parent
    local_csv_path = replication_dir.parent / "tutorials/public_datasets/HC.csv"

    if local_csv_path.exists():
        df = pd.read_csv(local_csv_path, index_col=0)
    else:
        df = pd.read_csv(HOUSE_COOLING_URL, index_col=0)

    df = df.reset_index(drop=True)

    item_names = HOUSE_COOLING_ITEM_NAMES
    encoder = dict(zip(item_names, range(len(item_names))))

    # The chosen alternative per house (session).
    chosen_items = (
        df[df["depvar"] == True]  # noqa: E712 - compare to True explicitly for clarity
        .sort_values(by="idx.id1")["idx.id2"]
        .reset_index(drop=True)
    )
    item_index = torch.LongTensor(chosen_items.map(lambda x: encoder[x]).to_numpy())

    num_sessions = int(len(item_index))

    # Session-item observables (installation/operating costs + interactions).
    price_obs = utils.pivot3d(
        df,
        dim0="idx.id1",
        dim1="idx.id2",
        values=HOUSE_COOLING_FEATURE_COLUMNS,
    )

    # Align with the manuscript snippet: a nest dataset with only item_index, and an item dataset
    # with item_index + price_obs.
    nest_dataset = ChoiceDataset(item_index=item_index.clone())
    item_dataset = ChoiceDataset(item_index=item_index, price_obs=price_obs)
    dataset = JointDataset(nest=nest_dataset, item=item_dataset)

    # Default two-nest split used in the docs/tests: cooling vs no-cooling.
    nest_to_item = {
        0: ["gcc", "ecc", "erc", "hpc"],
        1: ["gc", "ec", "er"],
    }
    nest_to_item = {k: sorted(encoder[name] for name in v) for k, v in nest_to_item.items()}

    summary = {
        "num_sessions": num_sessions,
        "num_items": len(item_names),
        "num_choices": num_sessions,
    }
    return dataset, nest_to_item, summary


def print_section(title: str) -> None:
    print(f"\n{SECTION_LINE}\n{title}\n{SECTION_LINE}")


def print_subsection(title: str) -> None:
    print(f"\n{SUBSECTION_LINE}\n{title}\n{SUBSECTION_LINE}")


def print_paper_reference(location: str, description: str) -> None:
    """Emit a short pointer back to the manuscript for easy cross-linking."""
    print(f"[Paper Ref | {location}] {description}")


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


def clone_choice_dataset(dataset: ChoiceDataset) -> ChoiceDataset:
    """Clone a ChoiceDataset while gracefully falling back to deepcopy if needed."""
    try:
        return dataset.clone()
    except AssertionError as err:
        print(f"[Clone Warning] dataset.clone() raised '{err}'. Falling back to deepcopy.")
        return deepcopy(dataset)


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

    print_section("Torch-Choice Manuscript Crosswalk")
    print_paper_reference(
        "Overview",
        "Script mirrors the 'Torch-Choice: A PyTorch Package for Large-Scale Choice Modeling with Python' manuscript in "
        "`torch-choice-paper/ms.tex`.",
    )
    print_paper_reference(
        "Reading Guide",
        "Keep the PDF open while running this script: Sections 3 (data), 4 (models), and 5 (benchmarks) line up with the "
        "blocks below.",
    )

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
    print_paper_reference(
        "Section 3 (Data Structures)",
        "Car-choice example corresponds to the EasyDataWrapper walk-through in Section 3.1 of the manuscript.",
    )
    print_dataframe("car_choice.head()", car_choice)

    # Method 1: observables derived from columns.
    print_subsection("Adding Observables, Method 1: Columns")
    print_paper_reference(
        "Section 3.1 / Listing (EasyDataWrapper)",
        "Matches the first code listing that pipes the long-form car-choice table into `EasyDatasetWrapper`.",
    )
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
    print_paper_reference(
        "Section 3.1 (Manual Observables Table)",
        "Replicates the second listing where gender/income/speed/discount are supplied via auxiliary DataFrames.",
    )
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
    print_paper_reference(
        "Section 3.1 (Hybrid Wrapper Inputs)",
        "Mirrors the discussion about mixing column names and pre-aggregated tables when building EasyDatasetWrapper inputs.",
    )
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
    print_paper_reference(
        "Section 3.2 / Equation \\eqref{eq:random-obs-data}",
        "Synthetic tensor shapes (U=10, I=4, S=500, N=10,000) follow the Gaussian sampling recipe in that equation.",
    )
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
    print_paper_reference(
        "Section 3.3 (ChoiceDataset API)",
        "Correlates with the manuscript section that reviews `dataset.num_*`, cloning, device transfers, and batching.",
    )
    print(f"dataset.num_users={dataset.num_users}")
    print(f"dataset.num_items={dataset.num_items}")
    print(f"dataset.num_sessions={dataset.num_sessions}")
    print(f"len(dataset)={len(dataset)}")

    print_subsection("Cloning Behavior")
    print_paper_reference(
        "Section 3.3 (Cloning Listing)",
        "Matches the code block demonstrating that modifying a clone leaves the original dataset unchanged.",
    )
    print(f"dataset.item_index[:10]={dataset.item_index[:10]}")
    dataset_cloned = clone_choice_dataset(dataset)
    dataset_cloned.item_index = torch.full_like(dataset.item_index, 99)
    print(f"dataset_cloned.item_index[:10]={dataset_cloned.item_index[:10]}")
    print(f"dataset.item_index[:10]={dataset.item_index[:10]}")

    print_subsection("Device Movement")
    print_paper_reference(
        "Section 3.3 (Device Transfers)",
        "This mirrors the CPU→GPU transfer example right before the `_check_device_consistency()` listing.",
    )
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
        print_paper_reference(
            "Section 5 (Benchmarking)",
            "Keeping tensors on the same device is what enables the GPU speed-ups reported in Section 5.",
        )

        print_subsection("Device Consistency Error Demonstration")
        print_paper_reference(
            "Section 3.3 (Intentional Device Error)",
            "Same try/except setup discussed around the `_check_device_consistency()` error message.",
        )
        dataset_misaligned = dataset_cuda.clone()
        dataset_misaligned.item_index = dataset_misaligned.item_index.to("cpu")
        try:
            dataset_misaligned._check_device_consistency()
        except Exception as err:  # noqa: BLE001 - want to show exact exception message
            print("[Device Consistency] Expected error with mixed devices:")
            print(f"  {err}")

        dataset = dataset_cuda.to("cpu")

    print_subsection("Observables Dictionary Shapes")
    print_paper_reference(
        "Table 1 / Section 3.2",
        "`dataset.x_dict` shapes correspond to the tensor naming and dimension rules summarized in Table 1.",
    )
    for key, value in dataset.x_dict.items():
        if torch.is_tensor(value):
            print(f"dict.{key}.shape={tuple(value.shape)}")

    print_subsection("Mini-batch Extraction")
    print_paper_reference(
        "Section 3.3 (Mini-batch Example)",
        "Recreates the five-record sampling example that highlights in-place safety for subsets.",
    )
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
    print_paper_reference(
        "Section 3.4 (Chaining datasets)",
        "Demonstrates the same `JointDataset(item=..., nest=...)` example that prefaces the nested logit discussion.",
    )
    item_level_dataset = clone_choice_dataset(dataset)
    nest_level_dataset = clone_choice_dataset(dataset)
    joint_dataset = JointDataset(item=item_level_dataset, nest=nest_level_dataset)
    print(f"joint_dataset={joint_dataset}")

    print_subsection("DataLoader Consistency Checks")
    print_paper_reference(
        "Section 3.5 (PyTorch DataLoader)",
        "Aligns with the sampler/DataLoader walkthrough for researchers customizing training loops.",
    )
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
    print_paper_reference(
        "Section 4.1 (Conditional Logit)",
        "Implements the specification around Equations \\eqref{eq:genearl-utility-clm}, \\eqref{eq:clm-softmax}, "
        "and \\eqref{eq:clm-iia} for the Mode Canada case study.",
    )
    dataset_mode_canada = load_mode_canada_dataset()
    print(f"[Mode Canada] dataset={dataset_mode_canada}")

    print_subsection("Formula-based Specification")
    print_paper_reference(
        "Section 4.1.1 (Formula interface)",
        "Matches the `formula='(itemsession_cost_freq_ovt|constant)+...'` snippet shown in the manuscript.",
    )
    model = ConditionalLogitModel(
        formula="(itemsession_cost_freq_ovt|constant) + (session_income|item) + (itemsession_ivt|item-full) + (intercept|item)",
        dataset=dataset_mode_canada,
        num_items=4,
    )

    print_subsection("Dictionary-based Specification")
    print_paper_reference(
        "Section 4.1.2 (Dictionary interface)",
        "Explicitly replays the `coef_variation_dict` / `num_param_dict` example in the paper.",
    )
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
    print_paper_reference(
        "Section 4.1.3 / Equation \\eqref{eq:regularized-loglikelihood}",
        "Highlights how the same model can include L1 regularization with weight λ=0.5.",
    )
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

    print_subsection("Training via model.fit")
    print_paper_reference(
        "Section 4.1.4 (Model Estimation)",
        "Corresponds to the `model.fit(..., model_optimizer=\"LBFGS\")` example and the timing note in the manuscript.",
    )
    args.tensorboard_logdir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    model.fit(
        dataset_mode_canada,
        batch_size=-1,
        learning_rate=0.01,
        num_epochs=args.num_epochs,
        model_optimizer="LBFGS",
        backend="lightning",
        default_root_dir=str(args.tensorboard_logdir),
    )
    duration = time.perf_counter() - start
    print(f"[Training] Completed in {duration:.2f} seconds.")
    print_paper_reference(
        "Figure 1 (TensorBoard Curve)",
        "Generated logs can be visualized exactly like Figure 1 once you launch TensorBoard.",
    )
    launch_tensorboard(args.tensorboard_logdir, args.tensorboard_port)

    print_subsection("Conditional Logit Post-Estimation")
    print_paper_reference(
        "Section 4.1.5 (Post-Estimation)",
        "Pulls the same coefficients retrieved via `model.get_coefficient(...)` following Equation "
        "\\eqref{eq:clm-post-estimation-example}.",
    )
    clm_intercept_item = model.get_coefficient("intercept[item]")
    print(f"[CLM] intercept[item] shape={tuple(clm_intercept_item.shape)} sample={clm_intercept_item[:3]}")
    clm_session_income = model.get_coefficient("session_income[item]")
    print(f"[CLM] session_income[item] shape={tuple(clm_session_income.shape)} sample={clm_session_income[:3]}")
    clm_itemsession_cost = model.get_coefficient("itemsession_cost_freq_ovt[constant]")
    print(f"[CLM] itemsession_cost_freq_ovt[constant]={clm_itemsession_cost}")

    # === Nested Logit Model =====================================================================
    print_section("Nested Logit Model")
    print_paper_reference(
        "Section 4.2 (Nested Logit)",
        "Demonstrates the two-level specification linked to Equations \\eqref{eq:nlm-utility-decomposition}, "
        "\\eqref{eq:nlm-likelihood-decomposition}, and \\eqref{eq:nested-likelihood}.",
    )
    print_subsection("House Cooling Dataset")
    print_paper_reference(
        "Section 4.2 (Empirical Example)",
        "Uses the House Cooling dataset from Train & Croissant / Appendix tutorial to ground the nested logit discussion.",
    )
    nested_joint_dataset, nest_to_item, house_summary = build_house_cooling_joint_dataset()
    print(
        "[House Cooling] Summary: "
        f"{house_summary['num_sessions']} sessions, "
        f"{house_summary['num_items']} items, "
        f"{house_summary['num_choices']} observed choices."
    )
    print(f"[House Cooling] joint_dataset={nested_joint_dataset}")

    print_subsection("Nested Logit Specification via Formulas")
    print_paper_reference(
        "Section 4.2.2 (Formulas)",
        "Uses the formula interface described in the manuscript's Nested Logit section. "
        "For the House Cooling run we keep a minimal spec: nest intercepts via `(1|item)` and a constant coefficient on "
        "the item observable `price_obs`.",
    )
    nested_model = NestedLogitModel(
        nest_to_item=nest_to_item,
        nest_formula="(1|item)",
        item_formula="(price_obs|constant)",
        dataset=nested_joint_dataset,
        shared_lambda=False,
    )
    nested_model_regularized = NestedLogitModel(
        nest_to_item=nest_to_item,
        nest_formula="(1|item)",
        item_formula="(price_obs|constant)",
        dataset=nested_joint_dataset,
        shared_lambda=True,
        regularization="L2",
        regularization_weight=1.5,
    )
    print("[Nested Logit] Created regularized variant with L2 penalty (not trained here).")

    print_subsection("Training Nested Logit Model")
    print_paper_reference(
        "Section 4.2.3 (Training)",
        "Mirrors the Adam-based training recipe and TensorBoard logging mentioned for the nested model.",
    )
    nested_logdir = args.tensorboard_logdir / "nested_logit"
    nested_logdir.mkdir(parents=True, exist_ok=True)
    nested_start = time.perf_counter()
    nested_model.fit(
        nested_joint_dataset,
        batch_size=256,
        learning_rate=0.05,
        num_epochs=min(200, args.num_epochs),
        model_optimizer="Adam",
        backend="lightning",
        default_root_dir=str(nested_logdir),
    )
    nested_duration = time.perf_counter() - nested_start
    print(f"[Nested Logit] Training completed in {nested_duration:.2f} seconds.")

    print_subsection("Nested Logit Post-Estimation")
    print_paper_reference(
        "Section 4.2.4 (Post-Estimation)",
        "Demonstrates `get_coefficient(..., level=...)` as described in the manuscript. "
        "Note: coefficients are only retrievable if they were included in the corresponding formula.",
    )
    lambda_coeff = nested_model.get_coefficient("lambda")
    print(f"[Nested Logit] lambda coefficients={lambda_coeff}")
    nest_intercepts = nested_model.get_coefficient("intercept[item]", level="nest")
    print(f"[Nested Logit] nest intercepts shape={tuple(nest_intercepts.shape)} values={nest_intercepts}")
    try:
        item_user_intercepts = nested_model.get_coefficient("intercept[user]", level="item")
    except KeyError:
        print(
            "[Nested Logit] item-level user intercepts not available "
            "(not included in item_formula; add `(1|user)` if you want to estimate them)."
        )
    else:
        print(
            "[Nested Logit] item-level user intercepts "
            f"shape={tuple(item_user_intercepts.shape)} sample={item_user_intercepts[:2]}"
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Exit] Interrupted by user.")
        sys.exit(1)

