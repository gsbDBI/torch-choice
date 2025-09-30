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

# Get environment type from first command-line argument
# Use "basic" as default if no argument provided (${1:-"basic"} syntax)
ENV_TYPE=${1:-"basic"}

# Check if user requested help documentation
# [[ ]] is bash's enhanced test command, supporting || (OR) operator
if [[ "$ENV_TYPE" == "-h" || "$ENV_TYPE" == "--help" || "$ENV_TYPE" == "help" ]]; then
    # Display comprehensive help documentation
    echo "torch-choice uv Setup Script"
    echo
    echo "This script sets up torch-choice with uv package management:"
    echo "• Installs all dependencies via uv from PyPI (fast resolution)"
    echo "• Installs torch-choice from local source in editable mode"
    echo
    echo "USAGE:"
    echo "  ./scripts/setup_uv.sh [ENVIRONMENT_TYPE]"
    echo
    echo "ENVIRONMENT TYPES:"
    echo "  basic      - Core dependencies only (default)"
    echo "  dev        - Development tools (pytest, black, etc.)"
    echo "  complete   - All dependencies (dev + docs + notebooks + benchmarks)"
    echo "  notebooks  - Jupyter notebook support"
    echo "  benchmarks - Performance benchmarking tools"
    echo
    echo "EXAMPLES:"
    echo "  ./scripts/setup_uv.sh              # Basic setup"
    echo "  ./scripts/setup_uv.sh dev          # Development setup"
    echo "  ./scripts/setup_uv.sh complete     # Full setup"
    echo
    echo "OPTIONS:"
    echo "  -h, --help    Show this help message"
    # Exit with success code (0) after displaying help
    exit 0
fi

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

# Install dependencies based on the specified environment type
echo -e "${BLUE}[INFO]${NC} Installing dependencies for '$ENV_TYPE' environment..."

# Validate that the environment type is one of the supported options
# =~ is the regex match operator in bash
# ^(basic|dev|complete|notebooks|benchmarks)$ matches only these exact strings
# ! negates the match (true if NOT matched)
if [[ ! "$ENV_TYPE" =~ ^(basic|dev|complete|notebooks|benchmarks)$ ]]; then
    # Invalid environment type provided
    echo -e "${RED}[ERROR]${NC} Unknown environment type: $ENV_TYPE"
    echo "Available types: basic, dev, complete, notebooks, benchmarks"
    # Exit with error code 1
    exit 1
fi

# Install core dependencies (required for all environment types)
echo -e "${BLUE}[INFO]${NC} Installing core dependencies..."
# Use uv pip install to install packages with version constraints
# \< escapes the < character to prevent shell interpretation as redirection
# These are the minimal packages needed to run torch-choice
uv pip install numpy>=1.22,\<2.0 termcolor>=1.1.0 scikit-learn pandas>=1.4.3 tabulate>=0.8.10 torch>=1.12.0 pytorch-lightning>=1.6.3

# Note: Additional dependencies (dev tools, notebooks, etc.) will be installed
# via torch-choice "extras" in the next step based on ENV_TYPE

echo -e "${GREEN}[SUCCESS]${NC} Dependencies installed"

# Install torch-choice from local source with appropriate extras
echo -e "${BLUE}[INFO]${NC} Installing torch-choice from local source..."

# Install torch-choice with appropriate "extras" based on environment type
# Extras are defined in pyproject.toml [project.optional-dependencies]
# -e flag installs in "editable" mode (changes to source code take effect immediately)
# . refers to current directory (where pyproject.toml is located)
if [[ "$ENV_TYPE" == "complete" ]]; then
    # Complete: all optional dependencies (dev + docs + notebooks + benchmarks)
    echo -e "${BLUE}[INFO]${NC} Installing torch-choice with complete extras..."
    uv pip install -e ".[complete]"
elif [[ "$ENV_TYPE" == "dev" ]]; then
    # Dev: development tools (pytest, black, flake8, etc.)
    echo -e "${BLUE}[INFO]${NC} Installing torch-choice with dev extras..."
    uv pip install -e ".[dev]"
elif [[ "$ENV_TYPE" == "benchmarks" ]]; then
    # Benchmarks: performance benchmarking tools
    echo -e "${BLUE}[INFO]${NC} Installing torch-choice with benchmarks extras..."
    uv pip install -e ".[benchmarks]"
elif [[ "$ENV_TYPE" == "notebooks" ]]; then
    # Notebooks: Jupyter notebook and lab support
    echo -e "${BLUE}[INFO]${NC} Installing torch-choice with notebooks extras..."
    uv pip install -e ".[notebooks]"
else
    # Basic: only core dependencies, no extras
    echo -e "${BLUE}[INFO]${NC} Installing torch-choice (basic)..."
    uv pip install -e .
fi

echo -e "${GREEN}[SUCCESS]${NC} torch-choice installed from source"

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

# Display installed package versions for verification
echo
echo -e "${BLUE}[INFO]${NC} Installed package versions:"
# Run Python script to print versions of all core dependencies
# Using a multi-line Python script embedded in the shell script
uv run python -c "
import sys
# Define core packages to check with their import names
packages = {
    'torch-choice': 'torch_choice',
    'numpy': 'numpy',
    'pandas': 'pandas',
    'torch': 'torch',
    'pytorch-lightning': 'pytorch_lightning',
    'scikit-learn': 'sklearn',
    'termcolor': 'termcolor',
    'tabulate': 'tabulate'
}

print('  Core Dependencies:')
for display_name, import_name in packages.items():
    try:
        mod = __import__(import_name)
        version = getattr(mod, '__version__', 'unknown')
        print(f'    • {display_name}: {version}')
    except ImportError:
        pass
"

# Display environment-specific package versions if applicable
if [[ "$ENV_TYPE" == "dev" || "$ENV_TYPE" == "complete" ]]; then
    # Show dev tool versions
    uv run python -c "
dev_packages = {
    'pytest': 'pytest',
    'black': 'black',
}

print('  Development Tools:')
for display_name, import_name in dev_packages.items():
    try:
        mod = __import__(import_name)
        version = getattr(mod, '__version__', 'unknown')
        print(f'    • {display_name}: {version}')
    except ImportError:
        pass
"
fi

if [[ "$ENV_TYPE" == "notebooks" || "$ENV_TYPE" == "complete" ]]; then
    # Show notebook tool versions
    uv run python -c "
notebook_packages = {
    'jupyter': 'jupyter',
    'notebook': 'notebook',
}

print('  Notebook Tools:')
for display_name, import_name in notebook_packages.items():
    try:
        mod = __import__(import_name)
        version = getattr(mod, '__version__', 'unknown')
        print(f'    • {display_name}: {version}')
    except ImportError:
        pass
"
fi

# Print usage instructions and next steps for the user
echo
echo -e "${GREEN}[SUCCESS]${NC} Setup complete! Here's how to use your environment:"
echo
# Display what was installed and where
echo "📦 Installation Summary:"
echo "  • Dependencies installed via uv from PyPI"
echo "  • torch-choice installed from local source (editable mode)"
echo "  • Environment type: $ENV_TYPE"
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

# Show environment-specific usage instructions based on what was installed
# Only display relevant commands for the chosen environment type
if [[ "$ENV_TYPE" == "dev" || "$ENV_TYPE" == "complete" ]]; then
    # Development tools are available
    echo "Development tools available:"
    echo "  uv run torch-choice-test          # Run tests"
    echo "  uv run black torch_choice/        # Format code"
    echo "  uv run pytest                     # Run pytest"
    echo
fi

if [[ "$ENV_TYPE" == "benchmarks" || "$ENV_TYPE" == "complete" ]]; then
    # Benchmarking tools are available
    echo "Benchmarking tools available:"
    echo "  uv run torch-choice-benchmark --data_path ./data --output_path ./results"
    echo
fi

if [[ "$ENV_TYPE" == "notebooks" || "$ENV_TYPE" == "complete" ]]; then
    # Jupyter notebook tools are available
    echo "Jupyter notebooks:"
    echo "  uv run jupyter notebook           # Start Jupyter"
    echo "  uv run jupyter lab                # Start JupyterLab"
    echo
fi

# Direct user to comprehensive documentation
echo "For more information, see UV_SETUP.md"
echo
# Final success message
echo -e "${GREEN}[SUCCESS]${NC} All done! Happy coding with torch-choice! 🚀"