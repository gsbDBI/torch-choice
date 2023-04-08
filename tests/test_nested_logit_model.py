"""
This scripts contain unit tests validating functionalities of the nested logit model.

Author: Tianyu Du
Date: Jul. 23, 2022
"""
import unittest

import pandas as pd
import torch
from torch_choice.data import ChoiceDataset, JointDataset, utils
from torch_choice.model import NestedLogitModel
from torch_choice.utils.run_helper import run


class TestConditionalLogitModel(unittest.TestCase):
    """Unit tests for the conditional logit model."""

    def load_house_cooling_datasets(self):
        # loads the dataset of house cooling.
        df = pd.read_csv('./tutorials/public_datasets/HC.csv', index_col=0)
        df = df.reset_index(drop=True)
        # what was actually chosen.
        item_index = df[df['depvar'] == True].sort_values(by='idx.id1')['idx.id2'].reset_index(drop=True)
        item_names = ['ec', 'ecc', 'er', 'erc', 'gc', 'gcc', 'hpc']
        num_items = df['idx.id2'].nunique()
        # cardinal encoder.
        encoder = dict(zip(item_names, range(num_items)))
        item_index = item_index.map(lambda x: encoder[x])
        item_index = torch.LongTensor(item_index)

        nest_dataset = ChoiceDataset(item_index=item_index.clone())

        item_feat_cols = ['ich', 'och', 'icca', 'occa', 'inc.room', 'inc.cooling', 'int.cooling']
        price_obs = utils.pivot3d(df, dim0='idx.id1', dim1='idx.id2', values=item_feat_cols)

        item_dataset = ChoiceDataset(item_index=item_index, price_obs=price_obs)
        dataset = JointDataset(nest=nest_dataset, item=item_dataset)
        return encoder, dataset

    # ==================================================================================================================
    # Test if models under different configurations can be successfully initialized.
    # ==================================================================================================================

    def test_initialization_example_1(self):
        nest_to_item = {0: ['gcc', 'ecc', 'erc', 'hpc'],
                            1: ['gc', 'ec', 'er']}

        encoder, _ = self.load_house_cooling_datasets()
        # encode items to integers.
        for k, v in nest_to_item.items():
            v = [encoder[item] for item in v]
            nest_to_item[k] = sorted(v)

        model = NestedLogitModel(nest_to_item=nest_to_item,
                                 nest_coef_variation_dict={},
                                 nest_num_param_dict={},
                                 item_coef_variation_dict={'price_obs': 'constant'},
                                 item_num_param_dict={'price_obs': 7},
                                 shared_lambda=True)

        return model

    def test_initialization_example_2(self):
        # different definition of categories compared to example 1.
        nest_to_item = {0: ['ec', 'ecc', 'gc', 'gcc', 'hpc'],
                            1: ['er', 'erc']}

        encoder, _ = self.load_house_cooling_datasets()
        for k, v in nest_to_item.items():
            v = [encoder[item] for item in v]
            nest_to_item[k] = sorted(v)

        model = NestedLogitModel(nest_to_item=nest_to_item,
                                    nest_coef_variation_dict={},
                                    nest_num_param_dict={},
                                    item_coef_variation_dict={'price_obs': 'constant'},
                                    item_num_param_dict={'price_obs': 7},
                                    shared_lambda=True
                                    )
        return model

    def test_initialization_example_3(self):
        nest_to_item = {0: ['gcc', 'ecc', 'erc'],
                            1: ['hpc'],
                            2: ['gc', 'ec', 'er']}

        encoder, _ = self.load_house_cooling_datasets()
        for k, v in nest_to_item.items():
            v = [encoder[item] for item in v]
            nest_to_item[k] = sorted(v)

        model = NestedLogitModel(nest_to_item=nest_to_item,
                                nest_coef_variation_dict={},
                                nest_num_param_dict={},
                                item_coef_variation_dict={'price_obs': 'constant'},
                                item_num_param_dict={'price_obs': 7},
                                shared_lambda=True
                                )

        return model

    # ==================================================================================================================
    # Test the if the running script runs at all for different examples.
    # these tests only run the fitting for 100 epochs.
    # ==================================================================================================================

    def test_model_fitting_functionality_example_1(self):
        _, dataset = self.load_house_cooling_datasets()
        model = self.test_initialization_example_1()
        run(model, dataset, num_epochs=100)

    def test_model_fitting_functionality_example_2(self):
        _, dataset = self.load_house_cooling_datasets()
        model = self.test_initialization_example_2()
        run(model, dataset, num_epochs=100)

    def test_model_fitting_functionality_example_3(self):
        _, dataset = self.load_house_cooling_datasets()
        model = self.test_initialization_example_3()
        run(model, dataset, num_epochs=100)

    # ==================================================================================================================
    # Test the if the running script results in expected values of log-likelihood.
    # these tests aim to run the model until full convergence.
    # ==================================================================================================================

    def test_model_fitting_correctness_example_1(self):
        _, dataset = self.load_house_cooling_datasets()
        model = self.test_initialization_example_1()
        self.__test_model_fitting_correctness(model, dataset, expected_ll=-178.12, num_epochs=10000)

    def test_model_fitting_correctness_example_2(self):
        _, dataset = self.load_house_cooling_datasets()
        model = self.test_initialization_example_2()
        self.__test_model_fitting_correctness(model, dataset, expected_ll=-180.02, num_epochs=5000, learning_rate=0.3)

    def test_model_fitting_correctness_example_3(self):
        _, dataset = self.load_house_cooling_datasets()
        model = self.test_initialization_example_3()
        self.__test_model_fitting_correctness(model, dataset, expected_ll=-180.26)

    def __test_model_fitting_correctness(self, model, dataset, expected_ll, learning_rate=0.01, num_epochs=5000):
        # run until the full convergence and check if the models' log-likelihood is close to the expected value.
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
            if e % (num_epochs // 10) == 0:
                print(f'Epoch {e}: Log-likelihood={ll}')

        final_ll = ll

        self.assertAlmostEqual(final_ll, expected_ll, delta=10)


if __name__ == '__main__':
    unittest.main()
