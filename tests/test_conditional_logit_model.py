"""
This scripts contain unit tests validating functionalities of the conditional logit model.

Author: Tianyu Du
Date: Jul. 23, 2022
"""
import unittest

import pandas as pd
from copy import deepcopy
import torch
from torch_choice.data import ChoiceDataset, utils
from torch_choice.model import ConditionalLogitModel, NestedLogitModel
from torch_choice.utils.run_helper import run


class TestConditionalLogitModel(unittest.TestCase):
    """Unit tests for the conditional logit model."""
    def load_mode_canada_data(self):
        self.df = pd.read_csv('./tutorials/public_datasets/ModeCanada.csv')
        self.df = self.df.query('noalt == 4').reset_index(drop=True)
        self.df.sort_values(by='case', inplace=True)

        self.item_index = self.df[self.df['choice'] == 1].sort_values(by='case')['alt'].reset_index(drop=True)
        item_names = ['air', 'bus', 'car', 'train']
        num_items = 4
        encoder = dict(zip(item_names, range(num_items)))
        self.item_index = self.item_index.map(lambda x: encoder[x])
        self.item_index = torch.LongTensor(self.item_index)

        self.price_cost_freq_ovt = utils.pivot3d(self.df, dim0='case', dim1='alt',
                                                 values=['cost', 'freq', 'ovt'])

        self.price_ivt = utils.pivot3d(self.df, dim0='case', dim1='alt', values='ivt')

        self.session_income = self.df.groupby('case')['income'].first()
        self.session_income = torch.Tensor(self.session_income.values).view(-1, 1)

        dataset = ChoiceDataset(item_index=self.item_index,
                                price_cost_freq_ovt=self.price_cost_freq_ovt,
                                session_income=self.session_income,
                                price_ivt=self.price_ivt
                                )

        return dataset

    def test_initialization(self):
        model = ConditionalLogitModel(coef_variation_dict={'price_cost_freq_ovt': 'constant',
                                                   'session_income': 'item',
                                                   'price_ivt': 'item-full',
                                                   'intercept': 'item'},
                              num_param_dict={'price_cost_freq_ovt': 3,
                                              'session_income': 1,
                                              'price_ivt': 1,
                                              'intercept': 1},
                              num_items=4)
        return model

    def test_model_fitting_functionality(self):
        dataset = self.load_mode_canada_data()
        model = self.test_initialization()
        # run for only 100 epochs.
        run(model, dataset, num_epochs=100, learning_rate=0.01, batch_size=-1)

    def test_model_fitting_correctness(self):
        dataset = self.load_mode_canada_data()
        model = self.test_initialization()
        learning_rate=0.01
        # run until the full convergence.
        num_epochs=50000
        model = deepcopy(model)  # do not modify the model outside.
        data_loader = utils.create_data_loader(dataset, batch_size=len(dataset), shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # fit the model.
        for e in range(1, num_epochs + 1):
            # track the log-likelihood to minimize.
            ll, count = 0.0, 0.0
            for batch in data_loader:
                item_index = batch['item'].item_index if isinstance(model, NestedLogitModel) else batch.item_index
                loss = model.negative_log_likelihood(batch, item_index)

                ll -= loss.detach().item()# * len(batch)
                count += len(batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # ll /= count
            if e % (num_epochs // 10) == 0:
                print(f'Epoch {e}: Log-likelihood={ll}')

        final_ll = ll
        # based on R output.
        expected_ll = -1874.3

        self.assertAlmostEqual(final_ll, expected_ll, delta=20)


if __name__ == '__main__':
    unittest.main()
