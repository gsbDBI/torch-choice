import os
import sys
from copy import deepcopy
import pandas as pd
from typing import List
from time import time
from xlogit import MixedLogit
from xlogit import MultinomialLogit
assert MixedLogit.check_if_gpu_available(), "xlogit needs to run with GPU for a fair comparison."

DATA_PATH = "/home/tianyudu/Development/torch-choice/tutorials/performance_benchmark/benchmark_data"
DEVICE = "cuda"
NUM_SEEDS = 5


def run_xlogit_model_num_records(formula: str, df: pd.DataFrame, user_latents: List[str], item_latents: List[str]):
    # get column names.
    latents = user_latents + item_latents

    model = MultinomialLogit()

    if formula == "(item_latents|constant)":
        model.fit(X=df[item_latents], varnames=item_latents,
                  y=df['choice'], alts=df['item_id'], ids=df['session_id'])

    elif formula == "(user_latents|item)":
        model.fit(X=df[user_latents], varnames=user_latents, isvars=user_latents,
                   y=df['choice'], alts=df['item_id'], ids=df['session_id'])

    elif formula == "(user_latents|item) + (item_latents|constant)":
        model.fit(X=df[latents], varnames=latents, isvars=user_latents,
                  y=df['choice'], alts=df['item_id'], ids=df['session_id'])

    else:
        raise ValueError("Unknown formula.")

if __name__ == "__main__":
    formula_list = [
        "(item_latents|constant)",
        "(user_latents|item)",
        "(user_latents|item) + (item_latents|constant)"]


    TASK = sys.argv[1]
    # ==================================================================================================================
    # experiment 1: benchmark the performance of different number of records.
    # ==================================================================================================================
    if TASK == "num_records_experiment_small":
        record_list = []
        user_latents = [f"user_latent_{i}" for i in range(0, 10)]
        item_latents = [f"item_latent_{i}" for i in range(0, 10)]

        for seed in range(NUM_SEEDS):
            for formula in formula_list:
                for subsample_size in [1000, 2000, 3000, 5000, 7000, 10000, 30000, 50000, 70000, 100000, 200000, 500000, 1000000]:
                    df = pd.read_csv("./benchmark_data/simulated_choice_data_num_records_experiment.csv")
                    # take a subset of dataset to explore performance at specific
                    df_subset = df[df["session_id"] < subsample_size]
                    # record time taken.
                    start_time = time()
                    run_xlogit_model_num_records(formula, df_subset)
                    time_taken = time() - start_time

                    print('Time taken:', time_taken)
                    record_list.append({'sample_size': subsample_size, 'time': time_taken, 'formula': formula, 'seed': seed})

        record = pd.DataFrame(record_list)
        record.to_csv(f'xlogit_{TASK}.csv', index=False)

    elif TASK == "num_params_experiment_small":
        record_list = []
        for seed in range(NUM_SEEDS):
            for formula in formula_list:
                for num_params in [1, 5, 10, 15, 20, 30]:
                    df = pd.read_csv("./benchmark_data/simulated_choice_data_num_params_experiment_small.csv")
                    user_latents = [f"user_latent_{i}" for i in range(0, num_params)]
                    item_latents = [f"item_latent_{i}" for i in range(0, num_params)]

                    start_time = time()
                    run_xlogit_model_num_records(formula, df, user_latents, item_latents)
                    time_taken = time() - start_time
                    print('Time taken:', time_taken)
                    record_list.append({'sample_size': df["session_id"].nunique(), 'time': time_taken, 'formula': formula, 'num_params': num_params, 'seed': seed})
        record = pd.DataFrame(record_list)
        record.to_csv(f'xlogit_{TASK}.csv', index=False)


    elif TASK == "num_items_experiment_small":
        user_latents = [f"user_latent_{i}" for i in range(0, 5)]
        item_latents = [f"item_latent_{i}" for i in range(0, 5)]

        # to compare with R.
        record_list = []
        for formula in formula_list:
            for num_items in [10, 30, 50, 100, 150, 200]:
                for seed in range(NUM_SEEDS):
                    df = pd.read_csv(f"./benchmark_data/simulated_choice_data_num_items_experiment_{num_items}.csv")
                    start_time = time()
                    run_xlogit_model_num_records(formula, df, user_latents, item_latents)
                    time_taken = time() - start_time
                    print('Time taken:', time_taken)
                    record_list.append({'sample_size': df["session_id"].nunqiue(), 'time': time_taken, 'formula': formula, 'num_items': num_items, 'seed': seed})

        record = pd.DataFrame(record_list)
        record.to_csv(f'xlogit_{TASK}.csv', index=False)
