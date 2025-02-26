# Standard library imports
import argparse
from typing import Union
import os
import sys
import platform
from copy import deepcopy
from time import time
from datetime import datetime  # Added to record the current date

# Third-party imports
import pandas as pd
import torch
import torch.optim
from tqdm import tqdm

# Local imports
from torch_choice.data import utils as data_utils
from torch_choice.model import ConditionalLogitModel
from torch_choice.model.nested_logit_model import NestedLogitModel
from torch_choice.data import ChoiceDataset


def run(model: Union[ConditionalLogitModel, NestedLogitModel],
        dataset: ChoiceDataset,
        batch_size: int = -1,
        learning_rate: float = 0.01,
        num_epochs: int = 5000,
        model_optimizer: str = 'Adam') -> tuple[float, float, int]:
    """Run the model training and return the best loss, time taken, and the number of epochs run.
       The number of epochs returned is where early stopping was triggered (or num_epochs if never triggered).
    """
    assert isinstance(model, ConditionalLogitModel) or isinstance(model, NestedLogitModel), \
        f'A model of type {type(model)} is not supported by this runner.'
    # Create a copy of the model so that the original is not modified.
    model = deepcopy(model)
    data_loader = data_utils.create_data_loader(dataset, batch_size=batch_size, shuffle=True)

    optimizer_cls = {'SGD': torch.optim.SGD,
                     'Adagrad': torch.optim.Adagrad,
                     'Adadelta': torch.optim.Adadelta,
                     'Adam': torch.optim.Adam}.get(model_optimizer)
    if optimizer_cls is None:
        raise ValueError(f"Optimizer '{model_optimizer}' is not supported.")
    optimizer = optimizer_cls(model.parameters(), lr=learning_rate)

    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.7)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)
    print('=' * 20, 'received model', '=' * 20)
    print(model)
    print('=' * 20, 'received dataset', '=' * 20)
    print(dataset)
    print('=' * 20, 'training the model', '=' * 20)

    not_improved_tolerance = 50  # stop if the loss fails to improve for a number of consecutive epochs.
    best_loss = float('inf')
    not_improved_count = 0
    epochs_run = 0  # record the epoch at which training stops (or num_epochs if never early stopped)

    start_time = time()
    model.train()
    # Use one of the model's parameter devices for display purposes.
    device_info = next(model.parameters()).device if list(model.parameters()) else "unknown"
    for e in tqdm(range(1, num_epochs + 1), desc=f'Training on {device_info}', leave=False):
        total_loss = 0.0
        for batch in data_loader:
            # Use the appropriate item_index based on the type of model.
            item_index = batch['item'].item_index if isinstance(model, NestedLogitModel) else batch.item_index
            # the model.loss returns negative log-likelihood + regularization term,
            # but we are not using the regularization in this benchmark.
            loss = model.loss(batch, item_index)
            total_loss += float(loss.clone().detach().item())  # accumulate the loss to get the loss on the entire dataset.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # scheduler.step()
        current_loss = total_loss
        if current_loss < best_loss:
            best_loss = current_loss
            not_improved_count = 0  # reset the counter if improvement occurs
        else:
            not_improved_count += 1  # increment the counter

        if not_improved_count >= not_improved_tolerance:
            print(f'Early stopped at {e} epochs.')
            epochs_run = e
            break
        epochs_run = e

    time_taken = float(time() - start_time)  # total time taken in seconds
    return best_loss, time_taken, epochs_run


def load_dataset(data_path, filename, session_limit=None, num_params=None) -> ChoiceDataset:
    """
    Loads a dataset via torch.load and optionally:
      - Filters the dataset so that only sessions with session_index < session_limit are kept.
      - Limits the dimension of latent variables to num_params.
    """
    # load the dataset from the pickle file.
    ds = torch.load(os.path.join(data_path, filename), map_location=DEVICE, weights_only=False)
    if num_params is not None:
        ds.user_latents = ds.user_latents[:, :num_params]
        ds.item_latents = ds.item_latents[:, :num_params]
    if session_limit is not None:
        ds = ds[ds.session_index < session_limit]
    return ds


def run_experiment(args, task_config, run_configs) -> pd.DataFrame:
    record_list = []
    for seed in range(args.num_seeds):
        for formula in task_config['formulas']:
            for value in task_config['values']:
                torch.manual_seed(seed)
                dataset = task_config['loader'](args.data_path, value)
                model = ConditionalLogitModel(formula=formula,
                                              dataset=dataset,
                                              num_items=dataset.num_items).to(DEVICE)
                best_loss, time_taken, epochs_run = run(model, dataset, **run_configs)
                record_list.append({
                    'sample_size': len(dataset),
                    'time': time_taken,
                    'formula': formula,
                    'seed': seed,
                    'best_loss': best_loss,
                    task_config['key']: value,
                    'epochs_run': epochs_run
                })
                del model, dataset

    record = pd.DataFrame(record_list)
    # Attach system information to the record.
    for key, val in sys_info.items():
        record[key] = val
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="The path to the data.")
    parser.add_argument("--output_path", type=str, required=True, help="The path to save the results.")
    # Optional configurations.
    parser.add_argument("--device", type=str, required=False, default="auto", help="The device to run the experiment on.")
    parser.add_argument("--num_seeds", type=int, required=False, default=5, help="The number of seeds to run the experiment on.")
    parser.add_argument("--num_epochs", type=int, required=False, default=50000, help="The maximum number of epochs to run the experiment on.")
    parser.add_argument("--learning_rate", type=float, required=False, default=0.03, help="The learning rate to run the experiment on.")
    parser.add_argument("--batch_size", type=int, required=False, default=-1, help="The batch size to run the experiment on, use -1 for full batch training.")
    args = parser.parse_args()

    # Detect device automatically.
    if args.device == "auto":
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        DEVICE = args.device

    # Create the output directory if it does not exist.
    os.makedirs(args.output_path, exist_ok=True)

    # List of regression formulas to be benchmarked.
    formula_list = ["(user_latents|item) + (item_latents|constant)",
                    "(user_latents|item)",
                    "(item_latents|constant)"]

    global sys_info
    sys_info = {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "torch_choice_version": __import__("torch_choice").__version__,
        "device": DEVICE,
        **args.__dict__,
        "cpu_name": platform.processor(),
        "cpu_count": os.cpu_count(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "run_date": datetime.now().strftime("%Y-%m-%d")  # Added current date when benchmark is run
    }
    print(sys_info)

    # One set of experiments varies in one dimension, indicated by the (key, value) pairs.
    EXPERIMENT_CONFIGS = {
        "num_records_experiment_small": {
            "key": "sample_size",
            "values": [1_000, 2_000, 3_000, 5_000, 7_000, 10_000, 30_000, 50_000, 70_000, 100_000],
            "formulas": formula_list,
            "loader": lambda dp, val: load_dataset(data_path=dp, filename="simulated_choice_data_num_records_experiment_small_seed_42.pt", session_limit=val, num_params=None)
        },
        "num_records_experiment_large": {
            "key": "sample_size",
            "values": [1_000, 2_000, 3_000, 5_000, 7_000, 10_000, 30_000, 50_000, 70_000, 100_000],
            "formulas": formula_list,
            "loader": lambda dp, val: load_dataset(data_path=dp, filename="simulated_choice_data_full_dataset_seed_42.pt", session_limit=val, num_params=None)
        },
        "num_params_experiment_small": {
            "key": "num_params",
            "values": [1, 5, 10, 15, 20, 30],
            "formulas": formula_list,
            "loader": lambda dp, val: load_dataset(data_path=dp, filename="simulated_choice_data_num_params_experiment_small_seed_42.pt", session_limit=None, num_params=val)
        },
        "num_params_experiment_large": {
            "key": "num_params",
            "values": [1, 5, 10, 15, 20, 30],
            "formulas": formula_list,
            "loader": lambda dp, val: load_dataset(data_path=dp, filename="simulated_choice_data_full_dataset_seed_42.pt", session_limit=None, num_params=val)
        },
        "num_items_experiment_small": {
            "key": "num_items",
            "values": [10, 20, 30, 50, 100, 150, 200],
            "formulas": formula_list,
            "loader": lambda dp, val: load_dataset(data_path=dp, filename=f"simulated_choice_data_num_items_experiment_{val}_items_seed_42.pt", session_limit=None, num_params=None)
        },
        "num_items_experiment_large": {
            "key": "num_items",
            "values": [10, 20, 30, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
            "formulas": formula_list,
            "loader": lambda dp, val: load_dataset(data_path=dp, filename=f"simulated_choice_data_num_items_experiment_{val}_items_seed_42.pt", session_limit=None, num_params=None)
        }
    }

    for task in EXPERIMENT_CONFIGS.keys():
        task_config = EXPERIMENT_CONFIGS[task]
        task_config = EXPERIMENT_CONFIGS[task]
        run_configs = {
            "num_epochs": args.num_epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
        }
        df_record = run_experiment(args, task_config, run_configs)
        df_record["task"] = task
        df_record.to_csv(os.path.join(args.output_path, f"torch_choice_{task}.csv"), index=False)
