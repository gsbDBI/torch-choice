import unittest
from torch_choice.data.example_datasets import load_mode_canada_dataset, load_house_cooling_dataset_v1

class TestDatasets(unittest.TestCase):
    def test_load_mode_canada_dataset(self):
        dataset = load_mode_canada_dataset()
        self.assertIsNotNone(dataset, "Mode Canada dataset should not be None")
        # Check that dataset has attribute item_index
        self.assertTrue(hasattr(dataset, 'item_index'), "Dataset should have an item_index attribute")

    def test_load_house_cooling_dataset(self):
        dataset = load_house_cooling_dataset_v1()
        self.assertIsNotNone(dataset, "House Cooling dataset should not be None")
        # If the dataset is a JointDataset, it should have attributes 'nest' and 'item'. Otherwise, it should be a ChoiceDataset with an 'item_index' attribute.
        if hasattr(dataset, 'nest') and hasattr(dataset, 'item'):
            self.assertTrue(hasattr(dataset, 'nest'), "JointDataset should have a 'nest' attribute")
            self.assertTrue(hasattr(dataset, 'item'), "JointDataset should have an 'item' attribute")
        else:
            self.assertTrue(hasattr(dataset, 'item_index'), "ChoiceDataset should have an 'item_index' attribute")

if __name__ == '__main__':
    unittest.main()