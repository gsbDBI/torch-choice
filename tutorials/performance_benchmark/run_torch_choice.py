"""
Benchmark the performance of torch-choice on different simulated datasets.
"""
import argparse
import os
from copy import deepcopy
from time import time

import numpy as np
import pandas as pd
import torch
import torch.optim

from torch_choice.data import utils as data_utils
from torch_choice.model import ConditionalLogitModel
from torch_choice.model.conditional_logit_model import ConditionalLogitModel
from torch_choice.model.nested_logit_model import NestedLogitModel


def run(model, dataset, dataset_test=None, batch_size=-1, learning_rate=0.03, num_epochs=5000, report_frequency=None, compute_std=True, return_final_training_log_likelihood=False, model_optimizer='Adam'):
    """All in one script for the model training and result presentation."""
    if report_frequency is None:
        report_frequency = (num_epochs // 10)

    assert isinstance(model, ConditionalLogitModel) or isinstance(model, NestedLogitModel), \
        f'A model of type {type(model)} is not supported by this runner.'
    model = deepcopy(model)  # do not modify the model outside.
    data_loader = data_utils.create_data_loader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = {'SGD': torch.optim.SGD,
                 'Adagrad': torch.optim.Adagrad,
                 'Adadelta': torch.optim.Adadelta,
                 'Adam': torch.optim.Adam,
                 'LBFGS': torch.optim.LBFGS}[model_optimizer](model.parameters(), lr=learning_rate)

    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.7)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)

    total_loss_history = list()
    tol = 1E-5  # stop if the loss failed to improve tol proportion of average performance in the last k iterations.
    k = 50
    # fit the model.
    for e in range(1, num_epochs + 1):
        # track the log-likelihood to minimize.
        ll, count, total_loss = 0.0, 0.0, 0.0
        for batch in data_loader:
            optimizer.zero_grad()
            item_index = batch['item'].item_index if isinstance(model, NestedLogitModel) else batch.item_index
            # the model.loss returns negative log-likelihood + regularization term.
            loss = model.loss(batch, item_index)
            total_loss -= loss

            with torch.no_grad():
                if (e % report_frequency) == 0:
                    # record log-likelihood.
                    ll -= model.negative_log_likelihood(batch, item_index).detach().item() # * len(batch)
                    count += len(batch)

                    pred = model.forward(batch).argmax(dim=1)
                    acc = (pred == item_index).float().mean().item()
                    print('Accuracy: ', acc)

            loss.backward()

            if model_optimizer == "LBFGS":
                def closure():
                    optimizer.zero_grad()
                    loss = model.loss(batch, item_index)
                    loss.backward()
                    return loss
                optimizer.step(closure)
            else:
                optimizer.step()

        scheduler.step()

        current_loss = float(total_loss.detach().item())
        if np.isnan(current_loss):
            # NAN loss encountered (diverged), stop training.
            return current_loss

        if e > k:
            past_avg = np.mean(total_loss_history[-k:])
            improvement = (past_avg - current_loss) / past_avg
            if improvement < tol:
                print(f'Early stopped at {e} epochs.')
                break
        total_loss_history.append(current_loss)
        # ll /= count
        if (e % report_frequency) == 0:
            print(f'Epoch {e}: Log-likelihood={current_loss}')

    return current_loss

if __name__ == "__main__":
    formula_list = ["(user_latents|item) + (item_latents|constant)",
                    "(user_latents|item)",
                    "(item_latents|constant)"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, help="Task name.")
    parser.add_argument("--model_optimizer", type=str, help="Optimizer name.")
    parser.add_argument("--data_path", type=str, default="/home/tianyudu/Development/torch-choice/tutorials/performance_benchmark/benchmark_data", help="Path to the dataset.")
    parser.add_argument("--output_path", type=str, default="/home/tianyudu/Development/torch-choice/tutorials/performance_benchmark/performance_results", help="Path to the output.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the experiment.")
    parser.add_argument("--num_seeds", type=int, default=5, help="Number of seeds to run the experiment.")
    args = parser.parse_args()

    # ==================================================================================================================
    # experiment 1: benchmark the performance of different number of records.
    # ==================================================================================================================
    if args.task == "num_records_experiment_large":
        record_list = []
        for seed in range(args.num_seeds):
            for formula in formula_list:
                for subsample_size in [1000, 2000, 3000, 5000, 7000, 10000, 30000, 50000, 70000, 100000, 200000]:
                    # NOTE: on the entire dataset.
                    dataset = torch.load(os.path.join(args.data_path, "simulated_choice_data_full.pt"))
                    dataset_subset = dataset[dataset.session_index < subsample_size].to(args.device)
                    dataset_subset._num_items = dataset_subset.item_latents.shape[0]
                    dataset_subset._num_users = dataset_subset.user_latents.shape[0]
                    start_time = time()
                    model = ConditionalLogitModel(formula=formula, dataset=dataset_subset,
                                                  num_users=dataset_subset.user_latents.shape[0],
                                                  num_items=dataset_subset.item_latents.shape[0]).to(args.device)
                    loss = run(model, dataset_subset, num_epochs=30000, learning_rate=0.03, batch_size=-1, report_frequency=100, model_optimizer=args.model_optimizer)
                    time_taken = time() - start_time
                    print('Time taken:', time_taken)
                    record_list.append({'sample_size': subsample_size, 'time': time_taken, 'formula': formula, 'seed': seed, 'loss': loss})
                    print(record_list[-1])
                    del model, dataset_subset
        record = pd.DataFrame(record_list)
    elif args.task == "num_records_experiment_small":
        record_list = []
        for seed in range(args.num_seeds):
            for formula in formula_list:
                for subsample_size in [1000, 2000, 3000, 5000, 7000, 10000, 30000, 50000, 70000, 100000, 200000, 500000, 1000000]:
                    dataset = torch.load(os.path.join(args.data_path, "simulated_choice_data_num_records_experiment.pt"))
                    dataset_subset = dataset[dataset.session_index < subsample_size].to(args.device)
                    dataset_subset._num_items = dataset_subset.item_latents.shape[0]
                    dataset_subset._num_users = dataset_subset.user_latents.shape[0]
                    start_time = time()
                    model = ConditionalLogitModel(formula=formula, dataset=dataset_subset,
                                                  num_users=dataset_subset.user_latents.shape[0],
                                                  num_items=dataset_subset.item_latents.shape[0]).to(args.device)
                    loss = run(model, dataset_subset, num_epochs=30000, learning_rate=0.03, batch_size=-1, report_frequency=100, model_optimizer=args.model_optimizer)
                    time_taken = time() - start_time
                    print('Time taken:', time_taken)
                    record_list.append({'sample_size': subsample_size, 'time': time_taken, 'formula': formula, 'seed': seed, 'loss': loss})
                    print(record_list[-1])
                    del model, dataset_subset

        record = pd.DataFrame(record_list)
    elif args.task == "num_params_experiment_small":
        record_list = []
        for seed in range(args.num_seeds):
            for formula in formula_list:
                for num_params in [1, 5, 10, 15, 20, 30]:
                    start_time = time()
                    dataset_subset = torch.load(os.path.join(args.data_path, "simulated_choice_data_num_params_experiment_small.pt")).to(args.device)
                    dataset_subset.user_latents = dataset_subset.user_latents[:, :num_params]
                    dataset_subset.item_latents = dataset_subset.item_latents[:, :num_params]
                    dataset_subset._num_items = dataset_subset.item_latents.shape[0]
                    dataset_subset._num_users = dataset_subset.user_latents.shape[0]

                    model = ConditionalLogitModel(formula=formula, dataset=dataset_subset,
                                                  num_users=dataset_subset.user_latents.shape[0],
                                                  num_items=dataset_subset.item_latents.shape[0]).to(args.device)
                    print(model)
                    print(dataset_subset)
                    loss = run(model, dataset_subset, num_epochs=30000, learning_rate=0.03, batch_size=-1, report_frequency=100, model_optimizer=args.model_optimizer)
                    time_taken = time() - start_time
                    print('Time taken:', time_taken)
                    record_list.append({'sample_size': len(dataset_subset), 'time': time_taken, 'formula': formula, 'num_params': num_params, 'seed': seed, 'loss': loss})
                    print(record_list[-1])
                    del model, dataset_subset
        record = pd.DataFrame(record_list)
    elif args.task == "num_params_experiment_large":
        record_list = []
        for seed in range(args.num_seeds):
            for formula in formula_list:
                for num_params in [1, 5, 10, 15, 20, 30]:
                    start_time = time()
                    dataset_subset = torch.load(os.path.join(args.data_path, "simulated_choice_data_full.pt")).to(args.device)
                    dataset_subset.user_latents = dataset_subset.user_latents[:, :num_params]
                    dataset_subset.item_latents = dataset_subset.item_latents[:, :num_params]
                    dataset_subset = dataset_subset[dataset_subset.session_index < 200000]
                    dataset_subset._num_items = dataset_subset.item_latents.shape[0]
                    dataset_subset._num_users = dataset_subset.user_latents.shape[0]

                    model = ConditionalLogitModel(formula=formula, dataset=dataset_subset,
                                                  num_users=dataset_subset.user_latents.shape[0],
                                                  num_items=dataset_subset.item_latents.shape[0]).to(args.device)
                    print(model)
                    print(dataset_subset)
                    loss = run(model, dataset_subset, num_epochs=30000, learning_rate=0.03, batch_size=-1, report_frequency=100, model_optimizer=args.model_optimizer)
                    time_taken = time() - start_time
                    print('Time taken:', time_taken)
                    record_list.append({'sample_size': len(dataset_subset), 'time': time_taken, 'formula': formula, 'num_params': num_params, 'seed': seed, 'loss': loss})
                    print(record_list[-1])
                    del model, dataset_subset
        record = pd.DataFrame(record_list)
    elif args.task == "num_items_experiment_small":
        # to compare with R.
        record_list = []
        for formula in formula_list:
            for num_items in [10, 50, 100, 150, 200]:
                for seed in range(args.num_seeds):
                    start_time = time()
                    dataset_subset = torch.load(os.path.join(args.data_path, f"simulated_choice_data_{num_items}_items.pt")).to(args.device)
                    dataset_subset._num_items = num_items
                    dataset_subset._num_users = dataset_subset.user_latents.shape[0]
                    model = ConditionalLogitModel(formula=formula, dataset=dataset_subset,
                                                  num_users=dataset_subset.user_latents.shape[0],
                                                  num_items=num_items).to(args.device)
                    print(model)
                    print(dataset_subset)
                    loss = run(model, dataset_subset, num_epochs=30000, learning_rate=0.03, batch_size=-1, report_frequency=100, model_optimizer=args.model_optimizer)
                    time_taken = time() - start_time
                    print('Time taken:', time_taken)
                    record_list.append({'sample_size': len(dataset_subset), 'time': time_taken, 'formula': formula, 'num_items': num_items, 'seed': seed, 'loss': loss})
                    print(record_list[-1])
                    del model, dataset_subset

        record = pd.DataFrame(record_list)
    elif args.task == "num_items_experiment_large":
        record_list = []
        for formula in formula_list:
            for num_items in [10, 20, 50, 100, 150, 200, 250, 300, 350, 450, 500]:
                for seed in range(args.num_seeds):
                    start_time = time()
                    dataset_subset = torch.load(os.path.join(args.data_path, f"simulated_choice_data_{num_items}_items.pt")).to(args.device)
                    dataset_subset._num_items = num_items
                    dataset_subset._num_users = dataset_subset.user_latents.shape[0]
                    model = ConditionalLogitModel(formula=formula, dataset=dataset_subset,
                                                  num_users=dataset_subset.user_latents.shape[0],
                                                  num_items=num_items).to(args.device)
                    print(model)
                    print(dataset_subset)
                    loss = run(model, dataset_subset, num_epochs=30000, learning_rate=0.03, batch_size=-1, report_frequency=100, model_optimizer=args.model_optimizer)
                    time_taken = time() - start_time
                    print('Time taken:', time_taken)
                    record_list.append({'sample_size': len(dataset_subset), 'time': time_taken, 'formula': formula, 'num_items': num_items, 'seed': seed, 'loss': loss})
                    print(record_list[-1])
                    del model, dataset_subset

        record = pd.DataFrame(record_list)
    else:
        raise ValueError(f"Unknown task {args.task}")

    # finally, save the result.
    record.to_csv(os.path.join(args.output_path, f'Python_{args.task}_{args.model_optimizer}.csv'), index=False)
