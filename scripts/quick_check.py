#!/usr/bin/env python3
"""Minimal end-to-end torch-choice installation check."""

from __future__ import annotations

import os
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO

import torch
import torch_choice


TRAINING_KWARGS = dict(
    num_epochs=10,
    learning_rate=0.3,
    batch_size=32,
    model_optimizer="Adam",
)

VERBOSE = os.environ.get("TORCH_CHOICE_QUICK_CHECK_VERBOSE", "").lower() in {
    "1",
    "true",
    "yes",
}


def _capture_output(func) -> str:
    """Run callable while capturing stdout/stderr, re-raising on failure."""
    buffer = StringIO()
    try:
        with redirect_stdout(buffer), redirect_stderr(buffer):
            func()
    except Exception:
        print("[torch-choice quick check] ❌ failure; captured logs:\n")
        print(buffer.getvalue())
        raise
    return buffer.getvalue()


def _run_single_check(device_name: str) -> str:
    print(f"[torch-choice quick check] Running on device: {device_name}")

    def _run() -> None:
        device = torch.device(device_name)
        dataset = torch_choice.data.load_mode_canada_dataset().to(device)
        model = torch_choice.model.ConditionalLogitModel(
            formula="(itemsession_cost_freq_ovt|constant) + (session_income|item) + (itemsession_ivt|item-full) + (intercept|item)",
            dataset=dataset,
            num_items=4,
        ).to(device)
        model.fit(dataset, device=device_name, **TRAINING_KWARGS)

    return _capture_output(_run)


def _print_logs(label: str, logs: str) -> None:
    if VERBOSE and logs.strip():
        print(f"[torch-choice quick check] {label} logs:")
        print(logs)


def main() -> int:
    cpu_logs = _run_single_check("cpu")
    print("[torch-choice quick check] ✅ CPU run succeeded.\n")
    _print_logs("CPU", cpu_logs)

    if torch.cuda.is_available():
        gpu_logs = _run_single_check("cuda")
        print("[torch-choice quick check] ✅ GPU run succeeded.\n")
        _print_logs("GPU", gpu_logs)
    else:
        print("[torch-choice quick check] ⚠️ CUDA not available; skipped GPU test.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

