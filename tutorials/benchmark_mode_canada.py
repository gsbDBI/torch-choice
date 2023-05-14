from time import time
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
tqdm.pandas()

torch.manual_seed(1234)
import torch_choice
import torch_choice.utils
from torch_choice.data import ChoiceDataset, utils
from torch_choice.model import ConditionalLogitModel
from torch_choice.utils.run_helper import run


def duplicate_items_mode_canada_datasets(num_copies: int):
    df = pd.read_csv('https://raw.githubusercontent.com/gsbDBI/torch-choice/main/tutorials/public_datasets/ModeCanada.csv', index_col=0)
    df_list = [df.copy()]
    for i in range(1, num_copies):
        df_copy = df.copy()
        df_copy['alt'] = df_copy['alt'] + f':copy_{i}'
        # don't choose these fake items.
        df_copy['choice'] = 0
        # df_copy[['cost', 'freq', 'ovt', 'ivt', 'income']] *= i
        df_list.append(df_copy)

    df = pd.concat(df_list, ignore_index=True)
    df = df.query('noalt == 4').reset_index(drop=True)
    df.sort_values(by='case', inplace=True)

    def change_choice(subset):
        current_choice = subset.query('choice == 1')['alt'].values[0]
        subset['choice'] = 0
        # choose an alternative at random.
        possible_choices = [current_choice] + [f'{current_choice}:copy_{i}' for i in range(1, num_copies)]
        random_choice = np.random.choice(possible_choices)
        subset.loc[subset['alt'] == random_choice, 'choice'] = 1
        return subset

    df = df.groupby('case').progress_apply(change_choice)

    # add a copy of dataset so that all items are chosen at least once.
    lst = list()
    df_109 = df.query('case == 109').copy().reset_index(drop=True)
    max_cases = df['case'].max()
    for i in range(len(df_109)):
        d = df_109.copy()
        d['choice'] = int(0)
        d.loc[i, 'choice'] = int(1)
        d[['cost', 'freq', 'ovt', 'ivt', 'income']] *= (i **2 * 0.01)
        d['case'] = max_cases + 1 + i
        lst.append(d)
    df_109 = pd.concat(lst, ignore_index=True)
    df = pd.concat([df, df_109], ignore_index=True)

    item_index = df[df['choice'] == 1].sort_values(by='case')['alt'].reset_index(drop=True)
    item_names = df['alt'].unique()
    num_items = len(item_names)
    encoder = dict(zip(item_names, range(num_items)))
    item_index = item_index.map(lambda x: encoder[x])
    item_index = torch.LongTensor(item_index)
    price_cost_freq_ovt = utils.pivot3d(df, dim0='case', dim1='alt',
                                        values=['cost', 'freq', 'ovt'])
    price_ivt = utils.pivot3d(df, dim0='case', dim1='alt', values='ivt')
    session_income = df.groupby('case')['income'].first()
    session_income = torch.Tensor(session_income.values).view(-1, 1)

    # session_index = torch.arange(len(session_income))
    dataset = ChoiceDataset(
        item_index=item_index,
        num_items=num_items,
        session_index=torch.arange(len(session_income)),
        price_cost_freq_ovt=price_cost_freq_ovt,
        session_income=session_income,
        price_ivt=price_ivt)
    return df, dataset.clone()


def duplicate_obs_mode_canada_datasets(num_copies: int):
    df = pd.read_csv('https://raw.githubusercontent.com/gsbDBI/torch-choice/main/tutorials/public_datasets/ModeCanada.csv', index_col=0)
    df_list = list()
    num_cases = df['case'].max()
    for i in range(num_copies):
        df_copy = df.copy()
        df_copy['case'] += num_cases * i
        df_list.append(df_copy)
    df = pd.concat(df_list, ignore_index=True)
    df = df.query('noalt == 4').reset_index(drop=True)
    df.sort_values(by='case', inplace=True)
    item_index = df[df['choice'] == 1].sort_values(by='case')['alt'].reset_index(drop=True)
    item_names = ['air', 'bus', 'car', 'train']
    num_items = 4
    encoder = dict(zip(item_names, range(num_items)))
    item_index = item_index.map(lambda x: encoder[x])
    item_index = torch.LongTensor(item_index)
    price_cost_freq_ovt = utils.pivot3d(df, dim0='case', dim1='alt',
                                        values=['cost', 'freq', 'ovt'])
    price_ivt = utils.pivot3d(df, dim0='case', dim1='alt', values='ivt')
    session_income = df.groupby('case')['income'].first()
    session_income = torch.Tensor(session_income.values).view(-1, 1)

    # session_index = torch.arange(len(session_income))

    dataset = ChoiceDataset(
        # item_index=item_index.repeat(num_copies),
        item_index=item_index,
        num_items=num_items,
        session_index=torch.arange(len(session_income)),
        price_cost_freq_ovt=price_cost_freq_ovt,
        session_income=session_income,
        price_ivt=price_ivt)
    return df, dataset.clone()


# def double_inflating(num_item_multiplier: int, num_obs_multiplier: int):

if __name__ == '__main__':
    performance_records = list()
    k_range = [1, 5, 10, 50, 100, 500, 1000]
    # k_range = [1, 5, 10, 50]
    # k_range = [1, 5]
    # k_range = [100]
    num_seeds = 1
    dataset_at_k = dict()
    use_cache = True
    for k in tqdm(k_range):
        if use_cache:
            dataset = torch.load(f'./benchmark_datasets/mode_canada_duplicate_items_{k}.pt')
            dataset_at_k[k] = dataset.clone()
        else:
            df, dataset = duplicate_items_mode_canada_datasets(k)
            # # df, dataset = duplicate_obs_mode_canada_datasets(k)
            dataset_at_k[k] = dataset.clone()
            # df.to_csv(f'./benchmark_datasets/mode_canada_duplicate_items_{k}.csv', index=False)
            # torch.save(dataset, f'./benchmark_datasets/mode_canada_duplicate_items_{k}.pt')
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', DEVICE)

    for optimizer in ['SGD', 'Adagrad', 'Adadelta', 'Adam']:
        for k in k_range:
            for seed in range(num_seeds):
                dataset = dataset_at_k[k].to(DEVICE)
                model = model = ConditionalLogitModel(
                    formula='(price_cost_freq_ovt|constant) + (session_income|item) + (price_ivt|item-full) + (intercept|item)',
                    dataset=dataset,
                    num_items=dataset.num_items).to(DEVICE)
                # only time the model estimation.
                start_time = time()
                # model, ll = run(model, dataset, batch_size=512, learning_rate=0.003 , num_epochs=30000, compute_std=False, return_final_training_log_likelihood=True, report_frequency=500)
                model, ll = run(model, dataset, batch_size=512, learning_rate=0.003, num_epochs=1000, compute_std=False, return_final_training_log_likelihood=True, report_frequency=100, model_optimizer=optimizer)
                end_time = time()
                performance_records.append(dict(k=k, seed=seed, time=end_time - start_time, ll=ll, device=DEVICE, optimizer=optimizer))

    # collect performance records to a dataframe.
    df_record = pd.DataFrame(performance_records)
    df_record.to_csv(sys.argv[1], index=False)
