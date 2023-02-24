from time import time
import sys

import pandas as pd
import torch
from tqdm import tqdm

from torch_choice.data import ChoiceDataset, utils
from torch_choice.model import ConditionalLogitModel
from torch_choice.utils.run_helper import run


def duplicate_mode_canada_datasets(num_copies: int):
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
        session_index=torch.arange(len(session_income)),
        price_cost_freq_ovt=price_cost_freq_ovt,
        session_income=session_income,
        price_ivt=price_ivt)
    return df, dataset.clone()


if __name__ == '__main__':
    performance_records = list()
    # k_range = [1, 5, 10, 50, 100, 500, 1_000, 5_000, 10_000]
    k_range = [1, 5]
    dataset_at_k = dict()
    for k in tqdm(k_range):
        df, dataset = duplicate_mode_canada_datasets(k)
        dataset_at_k[k] = dataset.clone()

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', DEVICE)

    for k in k_range:
        # run for 3 times.
        for seed in range(3):
            dataset = dataset_at_k[k].to(DEVICE)
            model = model = ConditionalLogitModel(
                formula='(price_cost_freq_ovt|constant) + (session_income|item) + (price_ivt|item-full) + (intercept|item)',
                dataset=dataset,
                num_items=4).to(DEVICE)
            # only time the model estimation.
            start_time = time()
            model, ll = run(model, dataset, batch_size=-1, learning_rate=0.03 , num_epochs=1000, compute_std=True, return_final_training_log_likelihood=True)
            end_time = time()
            performance_records.append(dict(k=k, seed=seed, time=end_time - start_time, ll=ll, device=DEVICE))

    # collect performance records to a dataframe.
    df_record = pd.DataFrame(performance_records)
    df_record.to_csv(sys.argv[1], index=False)
