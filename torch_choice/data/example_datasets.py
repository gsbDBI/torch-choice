"""
This file contains the example datasets for the torch-choice package.
"""
import pandas as pd
import torch
from torch_choice.data import ChoiceDataset, JointDataset, utils


def load_mode_canada_dataset():
    df = pd.read_csv('https://raw.githubusercontent.com/gsbDBI/torch-choice/main/tutorials/public_datasets/ModeCanada.csv')
    df = df.query('noalt == 4').reset_index(drop=True)
    df.sort_values(by='case', inplace=True)
    df.head()
    item_index = df[df['choice'] == 1].sort_values(by='case')['alt'].reset_index(drop=True)
    item_names = ['air', 'bus', 'car', 'train']
    num_items = 4
    encoder = dict(zip(item_names, range(num_items)))
    item_index = item_index.map(lambda x: encoder[x])
    item_index = torch.LongTensor(item_index)
    itemsession_cost_freq_ovt = utils.pivot3d(df, dim0='case', dim1='alt',
                                        values=['cost', 'freq', 'ovt'])

    itemsession_ivt = utils.pivot3d(df, dim0='case', dim1='alt', values='ivt')
    session_income = df.groupby('case')['income'].first()
    session_income = torch.Tensor(session_income.values).view(-1, 1)

    dataset = ChoiceDataset(item_index=item_index,
                            itemsession_cost_freq_ovt=itemsession_cost_freq_ovt,
                            session_income=session_income,
                            itemsession_ivt=itemsession_ivt
                            )
    return dataset


def load_house_cooling_dataset_v1():
    df = pd.read_csv('https://raw.githubusercontent.com/gsbDBI/torch-choice/main/tutorials/public_datasets/HC.csv', index_col=0)
    df = df.reset_index(drop=True)
    df.head()
    # what was actually chosen.
    item_index = df[df['depvar'] == True].sort_values(by='idx.id1')['idx.id2'].reset_index(drop=True)
    item_names = ['ec', 'ecc', 'er', 'erc', 'gc', 'gcc', 'hpc']
    num_items = df['idx.id2'].nunique()
    # cardinal encoder.
    encoder = dict(zip(item_names, range(num_items)))
    item_index = item_index.map(lambda x: encoder[x])
    item_index = torch.LongTensor(item_index)

    # nest feature: no nest feature, all features are item-level.
    nest_dataset = ChoiceDataset(item_index=item_index.clone())

    # item feature.
    item_feat_cols = ['ich', 'och', 'icca', 'occa', 'inc.room', 'inc.cooling', 'int.cooling']
    price_obs = utils.pivot3d(df, dim0='idx.id1', dim1='idx.id2', values=item_feat_cols)
    price_obs.shape

    item_dataset = ChoiceDataset(item_index=item_index, price_obs=price_obs)
    dataset = JointDataset(nest=nest_dataset, item=item_dataset)
    return dataset
