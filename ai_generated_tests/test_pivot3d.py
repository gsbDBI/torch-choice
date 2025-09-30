import unittest
import pandas as pd
import torch

from torch_choice.data.utils import pivot3d

class TestPivot3d(unittest.TestCase):
    def test_single_value(self):
        # Create a DataFrame with one numeric column 'cost'
        df = pd.DataFrame({
            'case': [1, 1, 2, 2],
            'alt': [0, 1, 0, 1],
            'cost': [10, 20, 30, 40]
        })
        result = pivot3d(df, 'case', 'alt', 'cost')
        # Expected shape: (number of cases, alternatives) = (2, 2)
        expected_shape = (2, 2, 1)
        self.assertEqual(result.shape, expected_shape)
        # Check that the first row, first column equals 10 (case 1, alt 0)
        self.assertAlmostEqual(result[0, 0].item(), 10)

    def test_multiple_values(self):
        # Create a DataFrame with two numeric columns 'cost' and 'freq'
        df = pd.DataFrame({
            'case': [1, 1, 2, 2],
            'alt': [0, 1, 0, 1],
            'cost': [10, 20, 30, 40],
            'freq': [1.1, 2.2, 3.3, 4.4]
        })
        result = pivot3d(df, 'case', 'alt', ['cost', 'freq'])
        # Expected shape: (number of cases, alternatives, 2) = (2, 2, 2)
        expected_shape = (2, 2, 2)
        self.assertEqual(result.shape, expected_shape)
        # Check that for case 1, alt 1, cost is 20 and freq is 2.2
        self.assertAlmostEqual(result[0, 1, 0].item(), 20)
        self.assertAlmostEqual(result[0, 1, 1].item(), 2.2)

    def test_missing_column(self):
        # Test with a DataFrame missing the 'cost' column
        df = pd.DataFrame({
            'case': [1, 1, 2, 2],
            'alt': [0, 1, 0, 1]
        })
        with self.assertRaises(KeyError):
            pivot3d(df, 'case', 'alt', 'cost')

    def test_empty_dataframe(self):
        # Test with an empty DataFrame that has the required columns
        df = pd.DataFrame({'case': [], 'alt': [], 'cost': []})
        result = pivot3d(df, 'case', 'alt', 'cost')
        # Expect an empty tensor
        self.assertEqual(result.numel(), 0)

if __name__ == '__main__':
    unittest.main()