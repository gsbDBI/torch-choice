import unittest

class TestImports(unittest.TestCase):
    def test_torch_choice_import(self):
        try:
            import torch_choice
        except Exception as e:
            self.fail(f"Importing torch_choice failed: {e}")

    def test_utils_import(self):
        try:
            from torch_choice import utils
        except Exception as e:
            self.fail(f"Importing torch_choice.utils failed: {e}")

    def test_model_import(self):
        try:
            from torch_choice import model
        except Exception as e:
            self.fail(f"Importing torch_choice.model failed: {e}")

    def test_data_import(self):
        try:
            from torch_choice import data
        except Exception as e:
            self.fail(f"Importing torch_choice.data failed: {e}")

if __name__ == '__main__':
    unittest.main()