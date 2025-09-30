import unittest
import torch
import numpy as np

from torch_choice.data.choice_dataset import ChoiceDataset


class TestComplexSyntheticDataset(unittest.TestCase):
    def test_complex_dataset(self):
        N = 10_000
        num_users = 10
        num_items = 4
        num_sessions = 500

        # Create synthetic observations
        user_obs = torch.randn(num_users, 128)
        item_obs = torch.randn(num_items, 64)
        useritem_obs = torch.randn(num_users, num_items, 32)
        session_obs = torch.randn(num_sessions, 10)
        itemsession_obs = torch.randn(num_sessions, num_items, 12)
        usersession_obs = torch.randn(num_users, num_sessions, 16)  # Added missing usersession_obs
        usersessionitem_obs = torch.randn(num_users, num_sessions, num_items, 8)

        # Create indices using numpy
        item_index = torch.LongTensor(np.random.choice(num_items, size=N))
        user_index = torch.LongTensor(np.random.choice(num_users, size=N))
        session_index = torch.LongTensor(np.random.choice(num_sessions, size=N))
        item_availability = torch.ones(num_sessions, num_items, dtype=torch.bool)

        # Create the synthetic dataset
        dataset = ChoiceDataset(
            item_index=item_index,  # required
            num_items=num_items,
            user_index=user_index,
            num_users=num_users,
            session_index=session_index,
            item_availability=item_availability,
            user_obs=user_obs,
            item_obs=item_obs,
            session_obs=session_obs,
            itemsession_obs=itemsession_obs,
            useritem_obs=useritem_obs,
            usersession_obs=usersession_obs,
            usersessionitem_obs=usersessionitem_obs
        )

        # Verify that dataset attributes have the expected shapes
        self.assertTrue(hasattr(dataset, 'user_obs'))
        self.assertEqual(dataset.user_obs.shape, (num_users, 128))
        self.assertTrue(hasattr(dataset, 'item_obs'))
        self.assertEqual(dataset.item_obs.shape, (num_items, 64))
        self.assertTrue(hasattr(dataset, 'useritem_obs'))
        self.assertEqual(dataset.useritem_obs.shape, (num_users, num_items, 32))
        self.assertTrue(hasattr(dataset, 'session_obs'))
        self.assertEqual(dataset.session_obs.shape, (num_sessions, 10))
        self.assertTrue(hasattr(dataset, 'itemsession_obs'))
        self.assertEqual(dataset.itemsession_obs.shape, (num_sessions, num_items, 12))
        self.assertTrue(hasattr(dataset, 'usersession_obs'))
        self.assertEqual(dataset.usersession_obs.shape, (num_users, num_sessions, 16))
        self.assertTrue(hasattr(dataset, 'usersessionitem_obs'))
        self.assertEqual(dataset.usersessionitem_obs.shape, (num_users, num_sessions, num_items, 8))

        # Also verify basic indices
        self.assertTrue(hasattr(dataset, 'item_index'))
        self.assertTrue(hasattr(dataset, 'user_index'))
        self.assertTrue(hasattr(dataset, 'session_index'))
        self.assertEqual(dataset.num_items, num_items)

    def test_minimal_dataset(self):
        # Test that a minimal dataset (only required fields) is created properly
        num_items = 4
        item_index = torch.LongTensor(np.random.choice(num_items, size=50))
        dataset = ChoiceDataset(item_index=item_index, num_items=num_items)
        self.assertEqual(dataset.num_items, num_items)
        self.assertTrue(hasattr(dataset, 'item_index'))
        # Optional fields like user_obs should not be present
        self.assertFalse(hasattr(dataset, 'user_obs'))

    def test_invalid_user_obs_shape(self):
        # Test that providing user_obs with a shape that doesn't match num_users raises an error
        num_users = 5
        # Incorrect shape: should be (num_users, feature), here we provide (10, 128)
        user_obs = torch.randn(10, 128)
        with self.assertRaises(AssertionError):
            _ = ChoiceDataset(
                item_index=torch.LongTensor(np.random.choice(4, size=10)),
                num_items=4,
                user_index=torch.LongTensor(np.random.choice(num_users, size=10)),
                num_users=num_users,
                user_obs=user_obs
            )

    def test_dataset_data_types(self):
        # Test that the indices in the dataset have the correct data type
        num_users, num_items, num_sessions = 5, 3, 50
        item_index = torch.LongTensor(np.random.choice(num_items, size=20))
        user_index = torch.LongTensor(np.random.choice(num_users, size=20))
        session_index = torch.LongTensor(np.random.choice(num_sessions, size=20))
        dataset = ChoiceDataset(
            item_index=item_index, num_items=num_items,
            user_index=user_index, num_users=num_users,
            session_index=session_index
        )
        self.assertEqual(dataset.item_index.dtype, torch.int64)
        self.assertEqual(dataset.user_index.dtype, torch.int64)
        self.assertEqual(dataset.session_index.dtype, torch.int64)


if __name__ == '__main__':
    unittest.main()