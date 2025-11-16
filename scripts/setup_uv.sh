#!/bin/bash
# Shebang: tells the system to execute this script using bash shell

# setup_uv.sh - Quick setup script for torch-choice with uv
# Usage: ./scripts/setup_uv.sh [environment_type]
# environment_type can be: basic, dev, complete, notebooks, benchmarks
#
# This script installs:
# - All dependencies via uv from PyPI (fast, reliable resolution)
# - torch-choice from local source in editable mode (for development)

# Exit immediately if any command fails (error handling)
set -e

# Enable uv preview features to suppress experimental feature warnings
# This is needed for extra-build-dependencies feature in pyproject.toml
export UV_PREVIEW=1

# ANSI color codes for terminal output formatting
RED='\033[0;31m'      # Red text for errors
GREEN='\033[0;32m'    # Green text for success messages
YELLOW='\033[1;33m'   # Yellow text for warnings
BLUE='\033[0;34m'     # Blue text for informational messages
NC='\033[0m'          # No Color - resets text color to default

# Environment type is now fixed to 'complete' (full setup with all extras)
ENV_TYPE="complete"

# Print script header banner for visual clarity
echo "========================================="
echo "  torch-choice uv Setup Script"
echo "========================================="
echo

# Verify that uv package manager is installed on the system
echo -e "${BLUE}[INFO]${NC} Checking if uv is installed..."
# command -v checks if a command exists in PATH
# &> /dev/null redirects both stdout and stderr to null (silent check)
# ! negates the result (true if command NOT found)
if ! command -v uv &> /dev/null; then
    # uv is not installed - print error message and installation instructions
    echo -e "${RED}[ERROR]${NC} uv is not installed. Please install it first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "  Or visit: https://github.com/astral-sh/uv#installation"
    # Exit with error code 1
    exit 1
fi
# uv is installed - confirm and show version using command substitution $()
echo -e "${GREEN}[SUCCESS]${NC} uv is installed ($(uv --version))"

# Create virtual environment (remove existing one if present for clean install)
echo -e "${BLUE}[INFO]${NC} Setting up virtual environment..."
# Check if .venv directory already exists
# -d tests for directory existence
if [ -d ".venv" ]; then
    # .venv already exists - remove it for a clean installation
    echo -e "${YELLOW}[WARNING]${NC} Virtual environment already exists at .venv"
    echo -e "${BLUE}[INFO]${NC} Removing existing virtual environment for clean installation..."
    # Remove the existing .venv directory and all its contents
    rm -rf .venv
    echo -e "${GREEN}[SUCCESS]${NC} Existing virtual environment removed"
fi

# Create a fresh virtual environment with Python 3.12
echo -e "${BLUE}[INFO]${NC} Creating new virtual environment with Python 3.12..."
# Create a new virtual environment using uv (creates .venv directory)
# --python 3.12 specifies the Python version to use
uv venv --python 3.12
echo -e "${GREEN}[SUCCESS]${NC} Virtual environment created with Python 3.12"

# Install dependencies for the complete environment
echo -e "${BLUE}[INFO]${NC} Installing dependencies for 'complete' environment..."

# Install all core + optional dependencies via torch-choice "complete" extras
# Extras are defined in pyproject.toml [project.optional-dependencies]
# -e flag installs in "editable" mode (changes to source code take effect immediately)
echo -e "${BLUE}[INFO]${NC} Installing torch-choice with complete extras..."
uv pip install -e ".[complete]"

echo -e "${GREEN}[SUCCESS]${NC} torch-choice (complete) installed from source"

# Verify that the installation was successful
echo -e "${BLUE}[INFO]${NC} Verifying installation..."

# Test if torch-choice can be imported and print its version
# uv run executes python in the virtual environment without needing to activate it
# -c flag runs a single Python command
# if...then checks the exit code (0 = success, non-zero = failure)
if uv run python -c "import torch_choice; print(f'torch-choice version: {torch_choice.__version__}')"; then
    # Import succeeded
    echo -e "${GREEN}[SUCCESS]${NC} torch-choice imported successfully"
else
    # Import failed - critical error
    echo -e "${RED}[ERROR]${NC} Failed to import torch-choice"
    exit 1
fi

# Test if PyTorch can be imported and is working correctly
if uv run python -c "import torch; print(f'PyTorch version: {torch.__version__}')"; then
    # PyTorch import succeeded
    echo -e "${GREEN}[SUCCESS]${NC} PyTorch is working"
else
    # PyTorch import failed - critical error
    echo -e "${RED}[ERROR]${NC} PyTorch import failed"
    exit 1
fi

# Run a small torch-choice model as a quick installation check
echo
echo -e "${BLUE}[INFO]${NC} Running a quick torch-choice model to verify installation..."
if uv run python - << 'PY'
import torch
import torch_choice

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

dataset = torch_choice.data.load_mode_canada_dataset().to(device)

model = torch_choice.model.ConditionalLogitModel(
    formula='(itemsession_cost_freq_ovt|constant) + (session_income|item) + (itemsession_ivt|item-full) + (intercept|item)',
    dataset=dataset,
    num_items=4,
).to(device)

torch_choice.run(
    model,
    dataset,
    num_epochs=5,
    learning_rate=0.003,
    batch_size=-1,
    model_optimizer="LBFGS",
    device=device,
)
PY
then
    echo -e "${GREEN}[SUCCESS]${NC} Quick torch-choice model run completed successfully"
else
    echo -e "${RED}[ERROR]${NC} Quick torch-choice model run failed"
    exit 1
fi

# Print usage instructions and next steps for the user
echo
echo -e "${GREEN}[SUCCESS]${NC} Setup complete! Here's how to use your environment:"
echo
# Display what was installed and where
echo "📦 Installation & Verification Summary:"
echo "  • Complete environment created with uv in .venv"
echo "  • torch-choice installed from local source (editable mode, [complete] extras)"
echo "  • Import checks for torch-choice and PyTorch passed"
echo "  • Quick torch-choice model run completed successfully"
echo ""
echo "📂 Virtual Environment Location:"
echo "  • Directory: $(pwd)/.venv"
echo "  • Python executable: $(pwd)/.venv/bin/python"
echo "  • Pip executable: $(pwd)/.venv/bin/pip"
echo "  • Site packages: $(pwd)/.venv/lib/python*/site-packages/"
echo
# Show two ways to use the environment: activate or use uv run
echo "🔧 How to Use:"
echo "Option 1 - Activate the virtual environment:"
echo "  source .venv/bin/activate"
echo "  python your_script.py"
echo
echo "Option 2 - Run commands with uv (no activation needed):"
echo "  uv run python your_script.py"
echo "  uv run jupyter notebook"
echo

# Development tools are available
echo "Development tools available:"
echo "  uv run torch-choice-test          # Run tests"
echo "  uv run black torch_choice/        # Format code"
echo "  uv run pytest                     # Run pytest"
echo

# Benchmarking tools are available
echo "Benchmarking tools available:"
echo "  uv run torch-choice-benchmark --data_path ./data --output_path ./results"
echo

# Jupyter notebook tools are available
echo "Jupyter notebooks:"
echo "  uv run jupyter notebook           # Start Jupyter"
echo "  uv run jupyter lab                # Start JupyterLab"
echo

# Direct user to comprehensive documentation
echo "For more information, see UV_SETUP.md"
echo
# Final success message
echo -e "${GREEN}[SUCCESS]${NC} All done! Happy coding with torch-choice! 🚀"