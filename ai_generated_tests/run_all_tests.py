import unittest
import sys
import warnings
import argparse
import os

def main():
    """Main entry point for the torch-choice test runner."""
    parser = argparse.ArgumentParser(description='Run all tests with optional warning suppression and fail fast option.')
    parser.add_argument('--ignore-warnings', action='store_true', help='Ignore all warnings during test execution.')
    parser.add_argument('--failfast', action='store_true', help='Stop on the first failure during test execution.')
    args, remaining_args = parser.parse_known_args()

    if args.ignore_warnings:
        warnings.filterwarnings("ignore")

    # Get the directory where this script is located
    test_dir = os.path.dirname(os.path.abspath(__file__))

    loader = unittest.TestLoader()
    suite = loader.discover(test_dir, pattern='test_*.py')
    runner = unittest.TextTestRunner(verbosity=2, failfast=args.failfast)
    result = runner.run(suite)

    # Flush output to ensure all test output is printed before summary
    sys.stdout.flush()
    sys.stderr.flush()

    # Print detailed summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    total_tests = result.testsRun
    num_failures = len(result.failures)
    num_errors = len(result.errors)
    num_skipped = len(result.skipped)
    num_passed = total_tests - num_failures - num_errors - num_skipped

    print(f"Total tests run:  {total_tests}")
    print(f"✅ Passed:        {num_passed}")
    if num_failures > 0:
        print(f"❌ Failed:        {num_failures}")
    if num_errors > 0:
        print(f"⚠️  Errors:        {num_errors}")
    if num_skipped > 0:
        print(f"⏭️  Skipped:       {num_skipped}")
    print("=" * 70)

    if result.wasSuccessful():
        print("🎉 All tests passed successfully!")
    else:
        print("❌ Some tests failed. Please review the output above.")
    print("=" * 70 + "\n")

    sys.exit(not result.wasSuccessful())


if __name__ == '__main__':
    main()