# UV Package Management Setup for torch-choice

This guide explains how to use [uv](https://github.com/astral-sh/uv) - a fast Python package installer and resolver - with the torch-choice project.

## Prerequisites

1. **Install uv**: Follow the [official installation guide](https://github.com/astral-sh/uv#installation)
   ```bash
   # On macOS and Linux:
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # On Windows:
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

   # Or with pip:
   pip install uv
   ```

2. **Verify installation**:
   ```bash
   uv --version
   ```

## Quick Start

### 1. Install torch-choice and dependencies

**Option A: Use the setup script (Recommended)**
```bash
# Show help and available options
./scripts/setup_uv.sh --help

# Quick setup with all dependencies
./scripts/setup_uv.sh complete

# Or other environment types
./scripts/setup_uv.sh dev        # Development tools
./scripts/setup_uv.sh notebooks  # Jupyter support
./scripts/setup_uv.sh benchmarks # Benchmarking tools
./scripts/setup_uv.sh basic      # Core only (default)
```

The setup script will:
- Create a fresh virtual environment in `.venv/`
- Install all dependencies from PyPI (fast resolution)
- Install torch-choice from local source in editable mode
- Verify the installation

**Option A.2: Install torch-choice from PyPI instead of local source**

Use `setup_uv_pypi.sh` if you want the *published* version of `torch-choice` instead of an editable install from your local checkout. This is the right script for replicators reproducing a specific paper or archived release, or for anyone who just wants the package installed from PyPI in a fresh `.venv/`.

```bash
# Install whatever's currently latest on PyPI (default)
./scripts/setup_uv_pypi.sh

# Pin to a specific version (e.g. for replicating a paper or archived release)
./scripts/setup_uv_pypi.sh 1.0.7

# Use a narrower set of extras
TORCH_CHOICE_EXTRA=benchmarks ./scripts/setup_uv_pypi.sh
```

The only practical difference from Option A is the install spec: `setup_uv.sh` runs `uv pip install -e ".[complete]"` (editable, local source); `setup_uv_pypi.sh` runs `uv pip install "torch-choice[complete]==<VERSION>"` (PyPI). Everything else — the `.venv` creation, the verification steps, the smoke-test model fit — is identical.

**Option B: Manual installation**
```bash
# Create virtual environment
uv venv --python 3.12

# Install dependencies first, then torch-choice from source
# Note: Always quote version constraints to prevent shell interpretation
uv pip install "numpy>=1.22,<2.0" "termcolor>=1.1.0" "scikit-learn" "pandas>=1.4.3" "tabulate>=0.8.10" "torch>=1.12.0" "pytorch-lightning>=1.6.3"
uv pip install -e .

# Or install everything together using pyproject.toml
uv pip install -e ".[complete]"
```

> **💡 Key Difference**: The setup script (Option A) installs dependencies from PyPI first (fast), then torch-choice from local source. Option B gives you more control but requires manual steps.

> **⚠️ Important**: Always quote package specifications with version constraints (e.g., `"numpy>=1.22"`). Without quotes, bash may interpret `>=` as a shell redirection operator and create unwanted files.

### 2. Available dependency groups

- `dev`: Development tools (pytest, black, isort, etc.)
- `docs`: Documentation building tools
- `notebooks`: Jupyter notebook support
- `benchmarks`: Performance benchmarking tools
- `complete`: All of the above plus additional packages

### 3. Activate and use the environment

```bash
# Option 1: Activate the virtual environment
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
python your_script.py

# Option 2: Use uv run (no activation needed)
uv run python your_script.py
```

### 4. Run tests

```bash
# Using the console script
uv run torch-choice-test

# Or directly
uv run python -m ai_generated_tests.run_all_tests

# With pytest
uv run pytest
```

### 5. Run benchmarks

```bash
# Using the console script (requires benchmark data)
torch-choice-benchmark --data_path ./data --output_path ./results

# Or directly with uv
uv run python replication/paper_performance_benchmarks/run_torch_choice.py --data_path ./data --output_path ./results
```

### 6. Visualize training runs with TensorBoard

TensorBoard is included in the uv-managed environment, so you can inspect Lightning logs without any extra installs:

```bash
uv run tensorboard --logdir lightning_logs --port 6006
```

## Working with Virtual Environments

### Create and activate a virtual environment

```bash
# Create a virtual environment with specific Python version
uv venv --python 3.12

# Or use default Python version
uv venv

# Activate it (Linux/macOS)
source .venv/bin/activate

# Activate it (Windows)
.venv\Scripts\activate

# Install the project
uv pip install -e ".[complete]"
```

### Working without activation

uv can run commands in the virtual environment without activation:

```bash
# Run python scripts
uv run python your_script.py

# Run installed console scripts
uv run torch-choice-test

# Run Jupyter notebooks
uv run jupyter notebook
```

## Development Workflow

### 1. Set up development environment

```bash
# Clone the repository
git clone https://github.com/gsbDBI/torch-choice.git
cd torch-choice

# Option A: Use the setup script (recommended)
./scripts/setup_uv.sh dev

# Option B: Manual setup
uv venv --python 3.12
uv pip install -e ".[dev]"
```

### 2. Code formatting and linting

```bash
# Format code with black
uv run black torch_choice/

# Sort imports with isort
uv run isort torch_choice/

# Run linting with flake8
uv run flake8 torch_choice/

# Type checking with mypy
uv run mypy torch_choice/
```

### 3. Running tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=torch_choice

# Run specific test file
uv run pytest ai_generated_tests/test_coefficient.py

# Use the console script
uv run torch-choice-test --failfast
```

## Jupyter Notebooks

### Install notebook dependencies

```bash
uv pip install -e ".[notebooks]"
```

### Run notebooks

```bash
# Start Jupyter server
uv run jupyter notebook

# Or JupyterLab
uv run jupyter lab

# Run specific notebook (if using nbconvert)
uv run jupyter nbconvert --to notebook --execute tutorials/landing_page_short_tutorial.ipynb
```

## Performance Benchmarking

### Set up benchmarking environment

```bash
# Install benchmark dependencies
uv pip install -e ".[benchmarks]"

# Generate synthetic datasets (if needed)
uv run python tutorials/generate_benchmark_datasets.py
```

### Run benchmarks

```bash
# Run all benchmarks
uv run torch-choice-benchmark \
    --data_path ./synthetic_data \
    --output_path ./benchmark_results \
    --device cuda

# Run specific experiment
uv run torch-choice-benchmark \
    --data_path ./synthetic_data \
    --output_path ./benchmark_results \
    --experiment_name num_records_experiment_small \
    --num_seeds 3
```

## Building Documentation

```bash
# Install documentation dependencies
uv pip install -e ".[docs]"

# Build documentation
uv run mkdocs build

# Serve documentation locally
uv run mkdocs serve
```

## Why Use uv?

1. **Speed**: uv is 10-100x faster than pip for dependency resolution
2. **Better dependency resolution**: More reliable conflict resolution than pip
3. **Lockfiles**: Reproducible installations with `uv.lock`
4. **Virtual environment management**: Streamlined venv creation and management
5. **No activation needed**: Run commands with `uv run` without activating venvs
6. **Cross-platform**: Works consistently across macOS, Linux, and Windows
7. **Modern**: Built in Rust for performance and reliability

## Migrating from conda/pip

If you were using conda environments:

```bash
# Instead of:
# conda create -n torch-choice python=3.9
# conda activate torch-choice
# pip install -r requirements.txt

# Use:
uv venv --python 3.9
uv pip install -e ".[complete]"
```

## Best Practices

### Always quote version constraints

When installing packages with version constraints, always use quotes:

```bash
# ✅ Correct - quotes prevent shell interpretation
uv pip install "numpy>=1.22,<2.0" "pandas>=1.4.3"

# ❌ Wrong - bash interprets >= as redirection, creates files like "=1.22"
uv pip install numpy>=1.22,<2.0 pandas>=1.4.3
```

### Use the setup script for consistency

The `./scripts/setup_uv.sh` script ensures a clean, reproducible environment:
- Removes old virtual environments
- Installs exact versions from lockfile
- Verifies installation
- Shows all installed versions

### Pin Python version

Specify Python version for reproducibility:

```bash
# Recommended
uv venv --python 3.12

# Less specific (may use different versions on different machines)
uv venv
```

### Use uv.lock for reproducibility

The `uv.lock` file ensures everyone uses the same dependency versions:

```bash
# Install from lockfile (exact versions)
uv pip install -r uv.lock

# Update lockfile after adding dependencies
uv pip compile pyproject.toml -o uv.lock
```

## Troubleshooting

### Common issues and solutions

1. **Strange files like `=1.22` or `=0.8.10` created**: This happens when version constraints aren't quoted in shell commands. The shell interprets `>=` as output redirection.
   ```bash
   # Fix: Delete the files
   rm -f =*

   # Prevent: Always quote version constraints
   uv pip install "numpy>=1.22" "pandas>=1.4.3"

   # Note: The .gitignore now includes =* to prevent tracking these files
   ```

2. **CUDA/PyTorch installation**: If you need specific CUDA versions:
   ```bash
   # Install PyTorch with specific CUDA version first
   uv pip install "torch" "torchvision" "torchaudio" --index-url https://download.pytorch.org/whl/cu118
   # Then install the rest
   uv pip install -e .
   ```

3. **Package conflicts**: uv's resolver is generally better, but if issues persist:
   ```bash
   uv pip install --resolution=lowest-direct -e ".[complete]"
   ```

4. **Slow installation**: Use parallel downloads:
   ```bash
   uv pip install --concurrent-downloads 10 -e ".[complete]"
   ```

5. **Virtual environment not found**: Make sure you're in the project root directory:
   ```bash
   # Check if .venv exists
   ls -la .venv

   # If not, create it
   ./scripts/setup_uv.sh complete
   ```

## Integration with IDEs

### VS Code

Add to your `.vscode/settings.json`:

```json
{
    "python.defaultInterpreterPath": "./.venv/bin/python",
    "python.terminal.activateEnvironment": true
}
```

### PyCharm

1. Go to File → Settings → Project → Python Interpreter
2. Add interpreter → Existing environment
3. Select `.venv/bin/python` (or `.venv\Scripts\python.exe` on Windows)

## Next Steps

- Explore the [tutorials](./tutorials/) to get started with torch-choice
- Check out the [documentation](https://gsbdbi.github.io/torch-choice/) for detailed API reference
- Run the [benchmarks](./replication/paper_performance_benchmarks/) to test performance
- Contribute to the project following the development workflow above