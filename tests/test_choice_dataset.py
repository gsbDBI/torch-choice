"""
This scripts contain unit tests validating functionalities of data containers.

Author: Tianyu Du
Date: Jul. 22, 2022
"""
import unittest

import numpy as np
import pandas as pd
import torch
from torch_choice.data import ChoiceDataset, JointDataset
from torch_choice.utils.easy_data_wrapper import EasyDatasetWrapper
from torch.utils.data.sampler import BatchSampler, SequentialSampler, RandomSampler


class TestChoiceDataset(unittest.TestCase):
    """
    This test class mainly tests functionalities demonstrated in the data management tutorial.
    """
    def create_random_data(self):
        self.num_users = 10
        self.num_items = 4
        self.num_sessions = 500

        self.length_of_dataset = 10000

        # create observables/features, the number of parameters are arbitrarily chosen.
        # generate 128 features for each user, e.g., race, gender.
        self.user_obs = torch.randn(self.num_users, 128)
        # generate 64 features for each user, e.g., quality.
        self.item_obs = torch.randn(self.num_items, 64)
        # generate 10 features for each session, e.g., weekday indicator.
        self.session_obs = torch.randn(self.num_sessions, 10)
        # generate 12 features for each session user pair, e.g., the budget of that user at the shopping day.
        self.price_obs = torch.randn(self.num_sessions, self.num_items, 12)

        self.item_index = torch.LongTensor(np.random.choice(self.num_items, size=self.length_of_dataset))
        self.user_index = torch.LongTensor(np.random.choice(self.num_users, size=self.length_of_dataset))
        self.session_index = torch.LongTensor(np.random.choice(self.num_sessions, size=self.length_of_dataset))

        # assume all items are available in all sessions.
        self.item_availability = torch.ones(self.num_sessions, self.num_items).bool()

    def create_random_choice_dataset(self):
        self.create_random_data()
        return ChoiceDataset(
            # pre-specified keywords of __init__
            item_index=self.item_index,  # required.
            # optional:
            user_index=self.user_index,
            session_index=self.session_index,
            item_availability=self.item_availability,
            # additional keywords of __init__
            user_obs=self.user_obs,
            item_obs=self.item_obs,
            session_obs=self.session_obs,
            price_obs=self.price_obs)

    def test_initialization(self):
        self.create_random_data()
        dataset = ChoiceDataset(
            # pre-specified keywords of __init__
            item_index=self.item_index,  # required.
            # optional:
            user_index=self.user_index,
            session_index=self.session_index,
            item_availability=self.item_availability,
            # additional keywords of __init__
            user_obs=self.user_obs,
            item_obs=self.item_obs,
            session_obs=self.session_obs,
            price_obs=self.price_obs)

        self.assertTrue(torch.all(dataset.item_index == self.item_index))
        self.assertTrue(torch.all(dataset.user_index == self.user_index))
        self.assertTrue(torch.all(dataset.session_index == self.session_index))
        self.assertTrue(torch.all(dataset.item_availability == self.item_availability))
        self.assertTrue(torch.all(dataset.user_obs == self.user_obs))
        self.assertTrue(torch.all(dataset.item_obs == self.item_obs))
        self.assertTrue(torch.all(dataset.session_obs == self.session_obs))
        self.assertTrue(torch.all(dataset.price_obs == self.price_obs))

    def test_property_methods(self):
        dataset = self.create_random_choice_dataset()
        self.assertEqual(dataset.num_users, self.num_users)
        self.assertEqual(dataset.num_items, self.num_items)
        self.assertEqual(dataset.num_sessions, self.num_sessions)
        self.assertEqual(len(dataset), self.length_of_dataset)

    def test_clone_method(self):
        dataset = self.create_random_choice_dataset()
        # save a copy of original value.
        x = dataset.item_index.clone()

        # clone the dataset.
        dataset_cloned = dataset.clone()
        # make modification to the cloned dataset.
        dataset_cloned.item_index = 99 * torch.ones(len(dataset)) + 1
        # get value after modification.
        y = dataset_cloned.item_index.clone()

        self.assertTrue(torch.all(dataset.item_index == x))  # does not change the original dataset.
        self.assertFalse(torch.all(dataset.item_index == y))  # should change the cloned dataset.

    def test_to_device_method(self):
        # This method can only be test on a machine with computational devices other than CPU.
        pass

    def test_getitem_method(self):
        dataset = self.create_random_choice_dataset()
        # __getitem__ to get batch.
        # pick 5 random sessions as the mini-batch.
        indices = torch.Tensor(np.random.choice(len(dataset), size=5, replace=False)).long()
        subset = dataset[indices]

        for obs_type in ['user_obs', 'item_obs', 'session_obs', 'price_obs', 'item_availability']:
            self.assertTrue(torch.all(getattr(dataset, obs_type) == getattr(subset, obs_type)))

        for index_type in ['item_index', 'user_index', 'session_index']:
            self.assertTrue(torch.all(getattr(dataset, index_type)[indices] == getattr(subset, index_type)))

    def test_x_dict_method(self):
        dataset = self.create_random_choice_dataset()
        expanded_ground_truth_of_x_dict = {
            # item_obs @ (num_items, *) -> (1, num_items, *) -> (len(dataset), num_items, *)
            'item_obs': self.item_obs.view(1, self.num_items, -1).expand(len(dataset), -1, -1).clone(),
            # user_obs @ (num_users, *) -> (len(dataset), 1, *) -> (len(dataset), num_items, *)
            'user_obs': self.user_obs[dataset.user_index].view(len(dataset), 1, -1).expand(-1, self.num_items, -1).clone(),
            # session_obs @ (num_sessions, *) -> (len(dataset), 1, *) -> (len(dataset), num_items, *)
            'session_obs': self.session_obs[dataset.session_index].view(len(dataset), 1, -1).expand(-1, self.num_items, -1).clone(),
            # price_obs @ (num_sessions, num_items, *) -> (len(dataset), num_items, *)
            'price_obs': self.price_obs[dataset.session_index, :, :].clone()
            }

        # test on the entire dataset.
        for name, ground_truth in expanded_ground_truth_of_x_dict.items():
            assert ground_truth.shape[:2] == (len(dataset), self.num_items)
            self.assertTrue(torch.all(ground_truth == dataset.x_dict[name]))

        # test on the subset slice of dataset __getitem__ to get batch.
        indices = torch.Tensor(np.random.choice(len(dataset), size=len(dataset) // 10, replace=False)).long()

        for name, ground_truth in expanded_ground_truth_of_x_dict.items():
            self.assertTrue(torch.all(ground_truth[indices, :, :] == dataset[indices].x_dict[name]))

    def test_dataloader_compatibility(self):
        dataset = self.create_random_choice_dataset()
        shuffle = False  # for demonstration purpose.
        batch_size = 32

        # Create sampler.
        sampler = BatchSampler(
            RandomSampler(dataset) if shuffle else SequentialSampler(dataset),
            batch_size=batch_size,
            drop_last=False)

        dataloader = torch.utils.data.DataLoader(dataset,
                                                sampler=sampler,
                                                collate_fn=lambda x: x[0],
                                                pin_memory=(dataset.device == 'cpu'))

        for i, batch in enumerate(dataloader):
            first, last = i * batch_size, min(len(dataset), (i + 1) * batch_size)
            idx = torch.arange(first, last)

            for obs_type in ['user_obs', 'item_obs', 'session_obs', 'price_obs', 'item_availability']:
                self.assertTrue(torch.all(getattr(dataset, obs_type) == getattr(batch, obs_type)))

            for index_type in ['item_index', 'user_index', 'session_index']:
                self.assertTrue(torch.all(getattr(dataset, index_type)[idx] == getattr(batch, index_type)))

    def create_random_joint_dataset(self):
        dataset = self.create_random_choice_dataset()
        dataset1 = dataset.clone()
        dataset2 = dataset.clone()
        self.assertEqual(dataset1, dataset2)
        joint_dataset = JointDataset(the_dataset=dataset1, another_dataset=dataset2)
        return dataset1, dataset2, joint_dataset

    def test_joint_dataset_initialization(self):
        dataset1, dataset2, joint_dataset = self.create_random_joint_dataset()
        self.assertTrue(len(joint_dataset) == len(dataset1) == len(dataset2) == self.length_of_dataset)

    def test_joint_dataset_getitem_method(self):
        dataset1, dataset2, joint_dataset = self.create_random_joint_dataset()
        indices = torch.Tensor(np.random.choice(len(dataset1), size=len(dataset1) // 10, replace=False)).long()

        self.assertIn('the_dataset', joint_dataset[indices].keys())
        self.assertIn('another_dataset', joint_dataset[indices].keys())

        self.assertEqual(joint_dataset[indices]['the_dataset'], dataset1[indices])
        self.assertEqual(joint_dataset[indices]['another_dataset'], dataset2[indices])


if __name__ == '__main__':
    unittest.main()