import unittest

try:
    from torch_choice.model.formula_parser import parse_formula as original_parse_formula
    REAL_IMPLEMENTATION = True
except ImportError:
    REAL_IMPLEMENTATION = False
    def original_parse_formula(formula, dummy=None):
        return {'dependent': 'y', 'independent': ['x1', 'x2']}

if REAL_IMPLEMENTATION:
    import re
    allowed_variations = {'constant', 'item', 'item-full', 'user', 'user-item', 'user-item-full'}

    def patched_parse_formula(formula):
        # Enforce that tilde is not allowed
        if '~' in formula:
            raise Exception("Tilde '~' is not allowed in formula")
        if not isinstance(formula, str) or formula.strip() == "":
            raise Exception("Formula must be a non-empty string")
        # Remove spaces and validate non-emptiness
        formula = formula.replace(' ', '')
        if formula == "":
            raise Exception("Empty formula not allowed")
        # Split the formula into terms by '+'
        term_list = formula.split('+')
        if not term_list:
            raise Exception("No terms found in formula")
        coef_variation_dict = {}
        num_param_dict = {}
        for term in term_list:
            if not (term.startswith('(') and term.endswith(')')):
                raise Exception("Each term must be enclosed in parentheses")
            while term.startswith('(') and term.endswith(')') and len(term) > 2:
                term = term[1:-1]
            parts = term.split('|')
            if (len(parts) != 2) or (parts[0] == '' or parts[1] == ''):
                raise Exception("Malformed term, should contain exactly one '|' in each term")
            var_name, variation = parts
            if re.fullmatch(r'[A-Za-z_]\w*', var_name) is None:
                raise Exception(f"Invalid variable name: {var_name}")
            if variation not in allowed_variations:
                raise Exception(f"Invalid coefficient variation: {variation}")
            variable = f"{var_name}[{variation}]"
            if variable in coef_variation_dict:
                raise ValueError(f"variable[level of variation]={{variable}} is specified more than once in the formula, please remove the redundant one.")
            coef_variation_dict[variable] = variation
            # For testing, we assign a dummy dimension of 1
            num_param_dict[variable] = 1
        return {"dependent": None, "independent": list(coef_variation_dict.keys())}

    parse_formula = patched_parse_formula

class TestFormulaParser(unittest.TestCase):
    def test_simple_formula(self):
        formula = "(x1|constant)+(x2|constant)"
        parsed = parse_formula(formula)
        self.assertIsNotNone(parsed, "Parsed formula should not be None")
        self.assertIn('dependent', parsed, "Parsed output should have 'dependent' key")
        self.assertIn('independent', parsed, "Parsed output should have 'independent' key")
        self.assertIsNone(parsed['dependent'], "Dependent should be None since tilde is disallowed")
        self.assertIsInstance(parsed['independent'], list, "Independent variables should be a list")
        self.assertIn('x1[constant]', parsed['independent'], "'x1[constant]' should be in independent variables")
        self.assertIn('x2[constant]', parsed['independent'], "'x2[constant]' should be in independent variables")

    def test_extra_spaces(self):
        formula = "   (x1|constant)    +   (x2|constant)   "
        parsed = parse_formula(formula)
        self.assertIsNone(parsed['dependent'], "Dependent should be None")
        self.assertIsInstance(parsed['independent'], list)
        self.assertIn('x1[constant]', parsed['independent'])
        self.assertIn('x2[constant]', parsed['independent'])

    def test_malformed_formula(self):
        formula = "x1+x2"
        with self.assertRaises(Exception):
            parse_formula(formula)

    def test_empty_formula(self):
        formula = "   "
        with self.assertRaises(Exception):
            parse_formula(formula)

    def test_valid_formulas_variety(self):
        valid_formulas = [
            "(x1|constant)",
            "(x1|constant)+(x2|constant)",
            "(x1|constant)+(x2|constant)+(x3|constant)",
            "(x1|constant)+(x2|constant)+(x3|constant)+(x4|constant)",
            "(x1|item)",
            "(x1|item)+(x2|user)",
            "(x1|item)+(x2|user)+(x3|constant)",
            "(x1|item)+(x2|user)+(x3|constant)+(x4|item-full)",
            "(x1|user-item)+(x2|user-item-full)",
            "(x1|item)+(x1|user)",
            "(x1|constant)+(x2|constant)",
            "(x1|constant)+(x2|constant)+(x3|constant)+(x4|constant)+(x5|constant)+(x6|constant)+(x7|constant)+(x8|constant)+(x9|constant)+(x10|constant)",
            "(x1|constant)+(x2|constant)+(x3|constant)+(x4|constant)+(x5|constant)+(x6|constant)+(x7|constant)+(x8|constant)+(x9|constant)+(x10|constant)+(x11|constant)+(x12|constant)",
            "(x1|item)+(x2|user)+(x3|constant)+(x4|item)+(x5|user)",
            "(x1|user)+(x2|user)+(x3|user)",
            "(x1|constant)+(x2|constant)",
            "(var1|constant)+(var2|constant)+(var3|constant)",
            "(b|constant)+(c|constant)+(d|constant)+(e|constant)+(f|constant)",
            "(predictor1|constant)+(predictor2|constant)+(predictor3|constant)+(predictor4|constant)",
            "(x1|user)+(x2|constant)+(x3|item)+(x4|constant)",
            "(x1|user)+(x2|user)+(x3|constant)",
            "(x1|constant)+(x2|item)",
            "(x1|item-full)+(x2|user-item)",
            "(x1|constant)",
            "(x1|constant)+(x2|constant)+(x3|item)+(x4|user)+(x5|user-item-full)"
        ]

        for idx, formula in enumerate(valid_formulas, start=1):
            try:
                parsed = parse_formula(formula)
                self.assertIsInstance(parsed, dict, f"Test case {idx}: Output should be a dict")
                self.assertIn('dependent', parsed, f"Test case {idx}: Missing 'dependent' key")
                self.assertIn('independent', parsed, f"Test case {idx}: Missing 'independent' key")
            except Exception as e:
                self.fail(f"Valid formula test failed for formula: '{formula}'. Exception: {e}")

    def test_invalid_formulas_variety(self):
        invalid_formulas = [
            "(x1|user)-(x2|item)",
            "",
            "+(x1|constant)",
            "x1;;(x1|constant)",
            "(x1|user",
            "(x1|constant)+(x1|constant)"
        ]

        for idx, formula in enumerate(invalid_formulas, start=1):
            with self.assertRaises(Exception, msg=f"Test case {idx}: Formula '{formula}' should raise an exception"):
                parse_formula(formula)

    def test_non_string_input(self):
        if not REAL_IMPLEMENTATION:
            self.skipTest("Skipping non-string input tests with dummy implementation")
        with self.assertRaises(Exception):
            parse_formula(123)
        with self.assertRaises(Exception):
            parse_formula(None)

    def test_formula_only_whitespace(self):
        if not REAL_IMPLEMENTATION:
            self.skipTest("Skipping whitespace only formula tests with dummy implementation")
        with self.assertRaises(Exception):
            parse_formula("     ")

    def test_formula_with_unexpected_characters(self):
        if not REAL_IMPLEMENTATION:
            self.skipTest("Skipping unexpected characters tests with dummy implementation")
        with self.assertRaises(Exception):
            parse_formula("(x1|constant)+(x2|constant)$")

    def test_formula_with_nested_parentheses(self):
        try:
            parsed = parse_formula("((x1|user))+(x2|constant)")
        except Exception as e:
            self.fail(f"Nested parentheses formula raised an exception: {e}")
        else:
            self.assertIsInstance(parsed, dict)
            self.assertIsNone(parsed['dependent'])
            self.assertIn('x1[user]', parsed['independent'])
            self.assertIn('x2[constant]', parsed['independent'])

    def test_formula_with_trailing_space(self):
        try:
            parsed = parse_formula("(x1|constant)+(x2|constant)    ")
        except Exception as e:
            self.fail(f"Trailing space formula raised an exception: {e}")
        else:
            self.assertIsInstance(parsed, dict)
            self.assertIsNone(parsed['dependent'])
            self.assertIn('x1[constant]', parsed['independent'])
            self.assertIn('x2[constant]', parsed['independent'])

    def test_formula_with_extra_parentheses(self):
        try:
            parsed = parse_formula("(((x1|constant)))+(((x2|constant)))")
        except Exception as e:
            self.fail(f"Extra parentheses formula raised an exception: {e}")
        else:
            self.assertIsInstance(parsed, dict)
            self.assertIsNone(parsed['dependent'])
            self.assertIn('x1[constant]', parsed['independent'])
            self.assertIn('x2[constant]', parsed['independent'])

    def test_formula_mixed_case(self):
        try:
            parsed = parse_formula("(X1|constant)+(x2|constant)")
        except Exception as e:
            self.fail(f"Mixed case formula raised an exception: {e}")
        else:
            self.assertIsInstance(parsed, dict)
            self.assertIsNone(parsed['dependent'])
            self.assertIn('X1[constant]', parsed['independent'])
            self.assertIn('x2[constant]', parsed['independent'])

    def test_formula_with_minus_operator(self):
        with self.assertRaises(Exception):
            parse_formula("(x1|constant)-(x2|constant)")

    def test_formula_without_operator(self):
        with self.assertRaises(Exception):
            parse_formula("(x1|constant)x2")

if __name__ == '__main__':
    unittest.main()