import os
import sys
from copy import deepcopy
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim

from torch_choice.data import utils as data_utils
from torch_choice.model import ConditionalLogitModel
from torch_choice.model.conditional_logit_model import ConditionalLogitModel
from torch_choice.model.nested_logit_model import NestedLogitModel

# The path storing the benchmark datasets, please check the README.md for instruction of
# getting these datasets and a list of expected files.
DATA_PATH = "./torch_choice_paper_data/"
assert os.path.exists(DATA_PATH), f"Please check README.md download the benchmark datasets and put them in {DATA_PATH}."
OUTPUT_PATH = "./results/"
assert os.path.exists(OUTPUT_PATH), f"Please check README.md create the output folder {OUTPUT_PATH} for storing performance benchmark outputs."

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if not torch.cuda.is_available():
    print("*" * 40)
    print("WARNING: CUDA (GPU device) is not available, benchmarking on CPU; the performance of torch-choice might not be as fast as expected.")
    print("*" * 40)

NUM_SEEDS = 5
OPTIMIZER = "Adam"


def run(model, dataset, dataset_test=None, batch_size=-1, learning_rate=0.01, num_epochs=5000, report_frequency=None, model_optimizer='LBFGS'):
    """All in one script for the model training and result presentation."""
    if report_frequency is None:
        report_frequency = (num_epochs // 10)

    assert isinstance(model, ConditionalLogitModel) or isinstance(model, NestedLogitModel), \
        f'A model of type {type(model)} is not supported by this runner.'
    model = deepcopy(model)  # do not modify the model outside.
    trained_model = deepcopy(model)  # create another copy for returning.
    data_loader = data_utils.create_data_loader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = {'SGD': torch.optim.SGD,
                 'Adagrad': torch.optim.Adagrad,
                 'Adadelta': torch.optim.Adadelta,
                 'Adam': torch.optim.Adam}[model_optimizer](model.parameters(), lr=learning_rate)

    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.7)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)
    print('=' * 20, 'received model', '=' * 20)
    print(model)
    print('=' * 20, 'received dataset', '=' * 20)
    print(dataset)
    print('=' * 20, 'training the model', '=' * 20)

    total_loss_history = list()
    tol = 1E-5  # stop if the loss failed to improve tol proportion of average performance in the last k iterations.
    k = 50
    # fit the model.
    for e in range(1, num_epochs + 1):
        # track the log-likelihood to minimize.
        ll, count, total_loss = 0.0, 0.0, 0.0
        for batch in data_loader:
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

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        current_loss = float(total_loss.detach().item())
        if e > k:
            past_avg = np.mean(total_loss_history[-k:])
            improvement = (past_avg - current_loss) / past_avg
            if improvement < tol:
                print(f'Early stopped at {e} epochs.')
                break
        total_loss_history.append(current_loss)
        # ll /= count
        if (e % report_frequency) == 0:
            print(f'Epoch {e}: Log-likelihood={ll}')

if __name__ == "__main__":
    record_list = []
    formula_list = ["(user_latents|item) + (item_latents|constant)",
                    "(user_latents|item)",
                    "(item_latents|constant)"]

    TASK = sys.argv[1]
    # ==================================================================================================================
    # experiment 1: benchmark the performance of different number of records.
    # ==================================================================================================================
    if TASK == "num_records_experiment_large":
        for seed in range(NUM_SEEDS):
            for formula in formula_list:
                for subsample_size in [1000, 2000, 3000, 5000, 7000, 10000, 30000, 50000, 70000, 100000, 200000]:
                    # NOTE: on the entire dataset.
                    dataset = torch.load(os.path.join(DATA_PATH, "simulated_choice_data_full.pt"))
                    dataset_subset = dataset[dataset.session_index < subsample_size].to(DEVICE)
                    start_time = time()
                    model = ConditionalLogitModel(formula=formula, dataset=dataset_subset, num_items=dataset_subset.num_items).to(DEVICE)
                    run(model, dataset_subset, num_epochs=50000, learning_rate=0.03, batch_size=-1, model_optimizer=OPTIMIZER)
                    time_taken = time() - start_time
                    print('Time taken:', time_taken)
                    record_list.append({'sample_size': subsample_size, 'time': time_taken, 'formula': formula, 'seed': seed})
                    del model, dataset_subset

    elif TASK == "num_records_experiment_small":
        for seed in range(NUM_SEEDS):
            for formula in formula_list:
                for subsample_size in [1000, 2000, 3000, 5000, 7000, 10000, 30000, 50000, 70000, 100000, 200000, 500000, 1000000]:
                    dataset = torch.load(os.path.join(DATA_PATH, "simulated_choice_data_num_records_experiment.pt"))
                    dataset_subset = dataset[dataset.session_index < subsample_size].to(DEVICE)
                    start_time = time()
                    model = ConditionalLogitModel(formula=formula, dataset=dataset_subset, num_items=dataset_subset.num_items).to(DEVICE)
                    run(model, dataset_subset, num_epochs=50000, learning_rate=0.03, batch_size=-1, model_optimizer=OPTIMIZER)
                    time_taken = time() - start_time
                    print('Time taken:', time_taken)
                    record_list.append({'sample_size': subsample_size, 'time': time_taken, 'formula': formula, 'seed': seed})
                    del model, dataset_subset

    # ==================================================================================================================
    # experiment 2: benchmark the performance of different number of parameters.
    # ==================================================================================================================
    elif TASK == "num_params_experiment_small":
        record_list = []
        for seed in range(NUM_SEEDS):
            for formula in formula_list:
                for num_params in [1, 5, 10, 15, 20, 30]:
                    start_time = time()
                    dataset_subset = torch.load(os.path.join(DATA_PATH, "simulated_choice_data_num_params_experiment_small.pt")).to(DEVICE)
                    dataset_subset.user_latents = dataset_subset.user_latents[:, :num_params]
                    dataset_subset.item_latents = dataset_subset.item_latents[:, :num_params]

                    model = ConditionalLogitModel(formula=formula, dataset=dataset_subset, num_items=dataset_subset.num_items).to(DEVICE)
                    print(model)
                    print(dataset_subset)
                    run(model, dataset_subset, num_epochs=50000, learning_rate=0.03, batch_size=-1, report_frequency=100, model_optimizer=OPTIMIZER)
                    time_taken = time() - start_time
                    print('Time taken:', time_taken)
                    record_list.append({'sample_size': len(dataset_subset), 'time': time_taken, 'formula': formula, 'num_params': num_params, 'seed': seed})
                    del model, dataset_subset

    elif TASK == "num_params_experiment_large":
        record_list = []
        for seed in range(NUM_SEEDS):
            for formula in formula_list:
                for num_params in [1, 5, 10, 15, 20, 30]:
                    start_time = time()
                    dataset_subset = torch.load(os.path.join(DATA_PATH, "simulated_choice_data_full.pt")).to(DEVICE)
                    dataset_subset.user_latents = dataset_subset.user_latents[:, :num_params]
                    dataset_subset.item_latents = dataset_subset.item_latents[:, :num_params]
                    dataset_subset = dataset_subset[dataset_subset.session_index < 200000]

                    model = ConditionalLogitModel(formula=formula, dataset=dataset_subset, num_items=dataset_subset.num_items).to(DEVICE)
                    print(model)
                    print(dataset_subset)
                    run(model, dataset_subset, num_epochs=50000, learning_rate=0.03, batch_size=-1, report_frequency=100, model_optimizer=OPTIMIZER)
                    time_taken = time() - start_time
                    print('Time taken:', time_taken)
                    record_list.append({'sample_size': len(dataset_subset), 'time': time_taken, 'formula': formula, 'num_params': num_params, 'seed': seed})
                    del model, dataset_subset

    # ==================================================================================================================
    # experiment 3: benchmark the performance of different number of items.
    # ==================================================================================================================
    elif TASK == "num_items_experiment_small":
        # to compare with R.
        record_list = []
        for formula in formula_list:
            for num_items in [10, 20, 50, 100, 150, 200]:
                for seed in range(NUM_SEEDS):
                    start_time = time()
                    dataset_subset = torch.load(os.path.join(DATA_PATH, f"simulated_choice_data_{num_items}_items.pt")).to(DEVICE)
                    model = ConditionalLogitModel(formula=formula, dataset=dataset_subset, num_items=dataset_subset.num_items).to(DEVICE)
                    print(model)
                    print(dataset_subset)
                    run(model, dataset_subset, num_epochs=50000, learning_rate=0.03, batch_size=-1, report_frequency=100, model_optimizer=OPTIMIZER)
                    time_taken = time() - start_time
                    print('Time taken:', time_taken)
                    record_list.append({'sample_size': len(dataset_subset), 'time': time_taken, 'formula': formula, 'num_items': num_items, 'seed': seed})
                    del model, dataset_subset

    elif TASK == "num_items_experiment_large":
        record_list = []
        for formula in formula_list:
            for num_items in [10, 20, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]:
                for seed in range(NUM_SEEDS):
                    start_time = time()
                    dataset_subset = torch.load(os.path.join(DATA_PATH, f"simulated_choice_data_{num_items}_items.pt")).to(DEVICE)
                    model = ConditionalLogitModel(formula=formula, dataset=dataset_subset, num_items=dataset_subset.num_items).to(DEVICE)
                    print(model)
                    print(dataset_subset)
                    run(model, dataset_subset, num_epochs=50000, learning_rate=0.03, batch_size=-1, report_frequency=100, model_optimizer=OPTIMIZER)
                    time_taken = time() - start_time
                    print('Time taken:', time_taken)
                    record_list.append({'sample_size': len(dataset_subset), 'time': time_taken, 'formula': formula, 'num_items': num_items, 'seed': seed})
                    del model, dataset_subset

    else:
        print("Invalid task, we don't have this task. Please check the README.md for the list of available tasks.")

    # combine records into a single dataframe.
    record = pd.DataFrame(record_list)
    # save the performance benchmark result.
    record.to_csv(os.path.join(OUTPUT_PATH, f'Python_{TASK}_{OPTIMIZER}.csv'), index=False)
