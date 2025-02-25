# Standard library imports
import argparse
import os
import sys
from copy import deepcopy
from time import time

# Third-party imports
import pandas as pd
import torch
import torch.optim
from tqdm import tqdm

# Local imports
from torch_choice.data import utils as data_utils
from torch_choice.model import ConditionalLogitModel
from torch_choice.model.nested_logit_model import NestedLogitModel

def run(model,
        dataset,
        batch_size=-1,
        learning_rate=0.01,
        num_epochs=5000,
        model_optimizer='Adam') -> tuple[float, float]:
    """All in one script for the model training and result presentation."""

    assert isinstance(model, ConditionalLogitModel) or isinstance(model, NestedLogitModel), \
        f'A model of type {type(model)} is not supported by this runner.'
    model = deepcopy(model)  # do not modify the model outside.
    # trained_model = deepcopy(model)  # create another copy for returning.
    data_loader = data_utils.create_data_loader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = {'SGD': torch.optim.SGD,
                 'Adagrad': torch.optim.Adagrad,
                 'Adadelta': torch.optim.Adadelta,
                 'Adam': torch.optim.Adam}[model_optimizer](model.parameters(), lr=learning_rate)

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

    not_improved_tolerance = 50  # stop if the loss failed to improve tol proportion of average performance in the last k iterations.
    best_loss = float('inf')
    not_improved_count = 0
    # fit the model.
    start_time = time()
    model.train()
    for e in tqdm(range(1, num_epochs + 1), desc=f'Training on {model.device}', leave=False):
        # the total loss for the entire dataset, which is the sum of the loss of all batches.
        total_loss = 0.0
        for batch in data_loader:
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
            not_improved_count = 0  # reset the counter.
        else:
            not_improved_count += 1  # increment the counter.

        if not_improved_count >= not_improved_tolerance:
            print(f'Early stopped at {e} epochs.')
            break
    time_taken = float(time() - start_time)  # elapsed time in seconds
    return best_loss, time_taken


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True,
                        choices=["num_records_experiment_large", "num_records_experiment_small", "num_params_experiment_small", "num_params_experiment_large", "num_items_experiment_small", "num_items_experiment_large"])
    parser.add_argument("--data_path", type=str, required=True, help="The path to the data.")
    parser.add_argument("--output_path", type=str, required=True, help="The path to save the results.")
    # optional configurations.
    parser.add_argument("--device", type=str, required=False, default="auto", help="The device to run the experiment on.")
    parser.add_argument("--num_seeds", type=int, required=False, default=5, help="The number of seeds to run the experiment on.")
    parser.add_argument("--num_epochs", type=int, required=False, default=50000, help="The maximum number of epochs to run the experiment on.")
    parser.add_argument("--learning_rate", type=float, required=False, default=0.03, help="The learning rate to run the experiment on.")
    parser.add_argument("--batch_size", type=int, required=False, default=-1, help="The batch size to run the experiment on, use -1 for full batch training.")
    args = parser.parse_args()

    run_configs = {
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
    }

    # detect device automatically.
    if args.device == "auto":
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        DEVICE = args.device

    os.makedirs(args.output_path, exist_ok=True)

    record_list = []
    formula_list = ["(user_latents|item) + (item_latents|constant)",
                    "(user_latents|item)",
                    "(item_latents|constant)"]

    sys_info = {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "torch_choice_version": __import__("torch_choice").__version__,
        "device": DEVICE,
        **args.__dict__,
    }

    # ==================================================================================================================
    # experiment 1: benchmark the performance of different number of records.
    # ==================================================================================================================
    if args.task == "num_records_experiment_large":
        for seed in range(args.num_seeds):
            for formula in formula_list:
                for subsample_size in [1000, 2000, 3000, 5000, 7000, 10000, 30000, 50000, 70000, 100000, 200000]:
                    torch.manual_seed(seed)
                    # NOTE: on the entire dataset.
                    dataset = torch.load(os.path.join(args.data_path, "simulated_choice_data_full.pt"))
                    dataset_subset = dataset[dataset.session_index < subsample_size].to(DEVICE)
                    # fix the random seed.
                    model = ConditionalLogitModel(formula=formula, dataset=dataset_subset, num_items=dataset_subset.num_items).to(DEVICE)
                    best_loss, time_taken = run(model, dataset_subset, **run_configs)
                    record_list.append({'sample_size': subsample_size, 'time': time_taken, 'formula': formula, 'seed': seed, 'best_loss': best_loss})
                    del model, dataset_subset
        record = pd.DataFrame(record_list)
        for key, val in sys_info.items():
            record[key] = val
        record.to_csv(os.path.join(args.output_path, f'Python_{args.task}.csv'), index=False)
    elif args.task == "num_records_experiment_small":
        for seed in range(args.num_seeds):
            for formula in formula_list:
                for subsample_size in [1000, 2000, 3000, 5000, 7000, 10000, 30000, 50000, 70000, 100000, 200000, 500000, 1000000]:
                    # fix the random seed.
                    torch.manual_seed(seed)
                    dataset = torch.load(os.path.join(args.data_path, "simulated_choice_data_num_records_experiment.pt"))
                    dataset_subset = dataset[dataset.session_index < subsample_size].to(DEVICE)
                    model = ConditionalLogitModel(formula=formula, dataset=dataset_subset, num_items=dataset_subset.num_items).to(DEVICE)
                    best_loss, time_taken = run(model, dataset_subset, **run_configs)
                    record_list.append({'sample_size': subsample_size, 'time': time_taken, 'formula': formula, 'seed': seed, 'best_loss': best_loss})
                    del model, dataset_subset

        record = pd.DataFrame(record_list)
        for key, val in sys_info.items():
            record[key] = val
        record.to_csv(os.path.join(args.output_path, f'Python_{args.task}.csv'), index=False)
    elif args.task == "num_params_experiment_small":
        record_list = []
        for seed in range(args.num_seeds):
            for formula in formula_list:
                for num_params in [1, 5, 10, 15, 20, 30]:
                    # fix the random seed.
                    torch.manual_seed(seed)
                    dataset_subset = torch.load(os.path.join(args.data_path, "simulated_choice_data_num_params_experiment_small.pt")).to(DEVICE)
                    dataset_subset.user_latents = dataset_subset.user_latents[:, :num_params]
                    dataset_subset.item_latents = dataset_subset.item_latents[:, :num_params]

                    model = ConditionalLogitModel(formula=formula, dataset=dataset_subset, num_items=dataset_subset.num_items).to(DEVICE)
                    print(model)
                    print(dataset_subset)
                    best_loss, time_taken = run(model, dataset_subset, **run_configs)
                    record_list.append({'sample_size': len(dataset_subset), 'time': time_taken, 'formula': formula, 'num_params': num_params, 'seed': seed, 'best_loss': best_loss})
                    del model, dataset_subset
        record = pd.DataFrame(record_list)
        for key, val in sys_info.items():
            record[key] = val
        record.to_csv(os.path.join(args.output_path, f'Python_{args.task}.csv'), index=False)

    elif args.task == "num_params_experiment_large":
        record_list = []
        for seed in range(args.num_seeds):
            for formula in formula_list:
                for num_params in [1, 5, 10, 15, 20, 30]:
                    # fix the random seed.
                    torch.manual_seed(seed)
                    dataset_subset = torch.load(os.path.join(args.data_path, "simulated_choice_data_full.pt")).to(DEVICE)
                    dataset_subset.user_latents = dataset_subset.user_latents[:, :num_params]
                    dataset_subset.item_latents = dataset_subset.item_latents[:, :num_params]
                    dataset_subset = dataset_subset[dataset_subset.session_index < 200000]

                    model = ConditionalLogitModel(formula=formula, dataset=dataset_subset, num_items=dataset_subset.num_items).to(DEVICE)
                    print(model)
                    print(dataset_subset)
                    best_loss, time_taken = run(model, dataset_subset, **run_configs)
                    record_list.append({'sample_size': len(dataset_subset), 'time': time_taken, 'formula': formula, 'num_params': num_params, 'seed': seed, 'best_loss': best_loss})
                    del model, dataset_subset
        record = pd.DataFrame(record_list)
        for key, val in sys_info.items():
            record[key] = val
        record.to_csv(os.path.join(args.output_path, f'Python_{args.task}.csv'), index=False)

    elif args.task == "num_items_experiment_small":
        # to compare with R.
        record_list = []
        for formula in formula_list:
            for num_items in [10, 30, 50, 100, 150, 200]:
                for seed in range(args.num_seeds):
                    # fix the random seed.
                    torch.manual_seed(seed)
                    dataset_subset = torch.load(os.path.join(args.data_path, f"simulated_choice_data_num_items_experiment_{num_items}_seed_42.pt"), weights_only=False).to(DEVICE)
                    model = ConditionalLogitModel(formula=formula, dataset=dataset_subset, num_items=dataset_subset.num_items).to(DEVICE)
                    print(model)
                    print(dataset_subset)
                    best_loss, time_taken = run(model, dataset_subset, **run_configs)
                    record_list.append({'sample_size': len(dataset_subset), 'time': time_taken, 'formula': formula, 'num_items': num_items, 'seed': seed, 'best_loss': best_loss})
                    del model, dataset_subset

        record = pd.DataFrame(record_list)
        for key, val in sys_info.items():
            record[key] = val
        record.to_csv(os.path.join(args.output_path, f'Python_{args.task}.csv'), index=False)
    elif args.task == "num_items_experiment_large":
        record_list = []
        for formula in formula_list:
            for num_items in [10, 20, 30, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]:
                for seed in range(args.num_seeds):
                    # fix the random seed.
                    torch.manual_seed(seed)
                    dataset_subset = torch.load(os.path.join(args.data_path, f"simulated_choice_data_{num_items}_items.pt")).to(DEVICE)
                    model = ConditionalLogitModel(formula=formula, dataset=dataset_subset, num_items=dataset_subset.num_items).to(DEVICE)
                    print(model)
                    print(dataset_subset)
                    best_loss, time_taken = run(model, dataset_subset, **run_configs)
                    record_list.append({'sample_size': len(dataset_subset), 'time': time_taken, 'formula': formula, 'num_items': num_items, 'seed': seed})
                    del model, dataset_subset

        record = pd.DataFrame(record_list)
        for key, val in sys_info.items():
            record[key] = val
        record.to_csv(os.path.join(args.output_path, f'Python_{args.task}.csv'), index=False)
