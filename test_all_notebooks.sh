#!/bin/bash
# Test all notebooks in the repository

cd /home/tianyudu/Development/torch-choice

echo "================================================================================"
echo "TESTING ALL NOTEBOOKS"
echo "================================================================================"
echo ""

NOTEBOOKS=(
    "replication/paper_demo.ipynb"
    "replication/paper_performance_benchmarks/simulate_datasets_synthetic.ipynb"
    "replication/paper_performance_benchmarks/visualize_performance_benchmarks_v2.ipynb"
    "tutorials/all_model_specification.ipynb"
    "tutorials/classroom_torch_choice_tutorial.ipynb"
    "tutorials/coefficient_initialization.ipynb"
    "tutorials/conditional_logit_model_mode_canada.ipynb"
    "tutorials/data_management.ipynb"
    "tutorials/easy_data_management.ipynb"
    "tutorials/landing_page/index.ipynb"
    "tutorials/landing_page_short_tutorial.ipynb"
    "tutorials/mnist/mnist.ipynb"
    "tutorials/nested_logit_model_house_cooling.ipynb"
    "tutorials/optimizer.ipynb"
    "tutorials/outside_option.ipynb"
    "tutorials/post_estimation_demos.ipynb"
    "tutorials/regularization.ipynb"
)

SUCCESS=0
FAILED=0

for nb in "${NOTEBOOKS[@]}"; do
    echo "--------------------------------------------------------------------------------"
    echo "Testing: $nb"
    echo "--------------------------------------------------------------------------------"

    # Run with timeout (1800s = 30 minutes per notebook)
    if timeout 1800 uv run jupyter execute --kernel_name python3 "$nb" > /tmp/notebook_test.log 2>&1; then
        echo "✓ PASS: $nb"
        ((SUCCESS++))
    else
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 124 ]; then
            echo "✗ TIMEOUT: $nb (exceeded 30 minutes)"
        else
            echo "✗ FAIL: $nb"
            echo "Last 20 lines of output:"
            tail -20 /tmp/notebook_test.log
        fi
        ((FAILED++))
    fi
    echo ""
done

echo "================================================================================"
echo "RESULTS"
echo "================================================================================"
echo "Passed: $SUCCESS"
echo "Failed: $FAILED"
echo "================================================================================"

if [ $FAILED -eq 0 ]; then
    echo "✓ ALL NOTEBOOKS EXECUTED SUCCESSFULLY"
    exit 0
else
    echo "✗ SOME NOTEBOOKS FAILED OR TIMED OUT"
    exit 1
fi


