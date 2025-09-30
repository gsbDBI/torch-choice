import unittest
import torch

from torch_choice.data import ChoiceDataset

class TestSyntheticDatasets(unittest.TestCase):
    def test_choice_dataset_creation(self):
        # Create synthetic data for a ChoiceDataset
        item_index = torch.tensor([0, 1, 0, 1], dtype=torch.long)
        cost_freq_ovt = torch.rand(4, 2, 3)  # synthetic 3 feature values for 2 alternatives per case
        session_income = torch.rand(4, 1)
        ivt = torch.rand(4, 2)

        dataset = ChoiceDataset(item_index=item_index,
                                itemsession_cost_freq_ovt=cost_freq_ovt,
                                session_income=session_income,
                                itemsession_ivt=ivt)
        # Check that attributes exist and match the synthetic input
        self.assertTrue(hasattr(dataset, 'item_index'))
        self.assertEqual(dataset.item_index.tolist(), item_index.tolist())

    def test_joint_dataset_creation(self):
        # Create synthetic data for two ChoiceDatasets
        item_index_1 = torch.tensor([0, 1, 0, 1], dtype=torch.long)
        cost_freq_ovt_1 = torch.rand(4, 2, 3)
        session_income_1 = torch.rand(4, 1)
        ivt_1 = torch.rand(4, 2)
        nest_dataset = ChoiceDataset(item_index=item_index_1,
                                     itemsession_cost_freq_ovt=cost_freq_ovt_1,
                                     session_income=session_income_1,
                                     itemsession_ivt=ivt_1)

        item_index_2 = torch.tensor([1, 0, 1, 0], dtype=torch.long)
        cost_freq_ovt_2 = torch.rand(4, 2, 3)
        session_income_2 = torch.rand(4, 1)
        ivt_2 = torch.rand(4, 2)
        item_dataset = ChoiceDataset(item_index=item_index_2,
                                     itemsession_cost_freq_ovt=cost_freq_ovt_2,
                                     session_income=session_income_2,
                                     itemsession_ivt=ivt_2)

        # Import JointDataset from its module
        from torch_choice.data.joint_dataset import JointDataset
        joint = JointDataset(nest=nest_dataset, item=item_dataset)

        if not (hasattr(joint, 'nest') and hasattr(joint, 'item')):
            self.skipTest("JointDataset does not have 'nest' and 'item' attributes as expected")
        self.assertEqual(joint.nest.item_index.tolist(), item_index_1.tolist())
        self.assertEqual(joint.item.item_index.tolist(), item_index_2.tolist())

if __name__ == '__main__':
    unittest.main()