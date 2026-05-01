"""Helpers for comparing torch-choice fits against R/mlogit reference fits.

Each per-dataset tutorial folder (e.g. ``tutorials/yogurt/``) ships:
  - ``fit_mlogit.R`` — fits the model with mlogit and emits JSON to stdout
  - ``mlogit_output.json`` — cached R output, committed alongside the notebook

This module exposes two functions that every dataset notebook can import:
  - ``run_or_load_mlogit`` — runs the R script via subprocess; falls back to the
    cached JSON when R is unavailable.
  - ``compare_coefs`` — builds a side-by-side coefficient comparison table and
    prints a one-line PASS/FAIL summary.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from torch_choice.utils.estimation_output import EstimationOutput


def run_or_load_mlogit(
    r_script_path: str | Path,
    csv_path: str | Path,
    cache_path: str | Path,
) -> tuple[pd.DataFrame, float]:
    """Fit a reference mlogit model via R, with a cached fallback.

    Tries ``Rscript <r_script_path> <csv_path>`` first. On any failure
    (R not installed, packages missing, non-zero exit, malformed JSON),
    loads the JSON at ``cache_path`` instead and prints a short notice.

    Args:
        r_script_path: Path to a per-dataset R script following the contract
            documented in ``tutorials/yogurt/fit_mlogit.R``.
        csv_path: Path to the dataset CSV the R script reads.
        cache_path: Path to a JSON file with the same schema as the script's
            stdout. Used as a fallback when live R execution fails.

    Returns:
        ``(coef_df, train_log_likelihood)``. The DataFrame has columns
        ``name``, ``estimate``, ``std_err``, ``z_value``, ``p_value``.
    """
    r_script_path = Path(r_script_path)
    csv_path = Path(csv_path)
    cache_path = Path(cache_path)

    payload = _try_run_rscript(r_script_path, csv_path)
    if payload is None:
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Live R fit failed and no cache found at {cache_path}. "
                "Install R + mlogit, or commit a cached JSON file first."
            )
        print(f"[mlogit] Using cached output from {cache_path} (R unavailable).")
        payload = json.loads(cache_path.read_text())
    else:
        print(f"[mlogit] Live R fit succeeded ({r_script_path.name}).")

    coef_df = pd.DataFrame(payload["coefficients"])
    coef_df = coef_df[["name", "estimate", "std_err", "z_value", "p_value"]]
    return coef_df, float(payload["log_likelihood"])


def _try_run_rscript(r_script_path: Path, csv_path: Path) -> dict | None:
    """Run the R script; return parsed JSON on success, ``None`` on any failure."""
    try:
        proc = subprocess.run(
            ["Rscript", str(r_script_path), str(csv_path)],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
            cwd=r_script_path.parent,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        print(f"[mlogit] Could not invoke Rscript: {exc}")
        return None
    if proc.returncode != 0:
        print(f"[mlogit] Rscript exited with code {proc.returncode}; stderr:")
        print(proc.stderr.strip()[-500:])
        return None
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        print(f"[mlogit] Rscript stdout is not valid JSON: {exc}")
        return None


def compare_coefs(
    mlogit_df: pd.DataFrame,
    tc_result: "EstimationOutput",
    name_map: dict[str, str],
    *,
    est_abs_tol: float = 1e-3,
    se_abs_tol: float = 1e-3,
) -> pd.DataFrame:
    """Side-by-side comparison of torch-choice and mlogit coefficients.

    Reports both the **point estimate** and the **standard error** for each
    coefficient. Both packages derive standard errors from the inverse Hessian
    at the optimum, so the SE match is a stronger consistency check than the
    point-estimate match alone.

    ``name_map`` is an explicit mapping of torch-choice coefficient names (rows
    in ``tc_result.coef_summary``) to mlogit coefficient names (rows in
    ``mlogit_df``). Every torch-choice coefficient must appear in the map; if
    one is missing, the function raises ``KeyError`` rather than silently
    skipping. This is intentional — silent misalignment is exactly the failure
    mode we want to avoid in a verification notebook.

    Args:
        mlogit_df: Output of :func:`run_or_load_mlogit`.
        tc_result: The :class:`EstimationOutput` returned by ``model.fit(...)``.
        name_map: ``{tc_coef_name: mlogit_coef_name}`` for every coefficient.
        est_abs_tol: Absolute-difference threshold for the estimate PASS line.
        se_abs_tol: Absolute-difference threshold for the std-err PASS line.

    Returns:
        DataFrame with columns ``coef``, ``mlogit_est``, ``tc_est``,
        ``est_abs_diff``, ``est_pct_diff``, ``mlogit_se``, ``tc_se``,
        ``se_abs_diff``, ``se_pct_diff``. Percent differences are computed
        as ``100 * |a - b| / max(|a|, |b|)`` and reported in percent units
        (e.g. ``0.003`` means 0.003%, *not* 0.3%).
    """
    tc_summary = tc_result.coef_summary  # pandas DataFrame indexed by tc name
    tc_names = list(tc_summary.index)
    mlogit_indexed = mlogit_df.set_index("name")

    missing = [n for n in tc_names if n not in name_map]
    if missing:
        raise KeyError(
            "name_map is missing entries for torch-choice coefficients: "
            + ", ".join(missing)
        )

    rows: list[dict] = []
    for tc_name in tc_names:
        ml_name = name_map[tc_name]
        if ml_name not in mlogit_indexed.index:
            raise KeyError(
                f"name_map points {tc_name!r} -> {ml_name!r}, "
                f"but {ml_name!r} is not a coefficient in mlogit_df."
            )
        tc_est = float(tc_summary.loc[tc_name, "Estimation"])
        ml_est = float(mlogit_indexed.loc[ml_name, "estimate"])
        tc_se = float(tc_summary.loc[tc_name, "Std. Err."])
        ml_se = float(mlogit_indexed.loc[ml_name, "std_err"])

        est_abs = abs(tc_est - ml_est)
        est_denom = max(abs(tc_est), abs(ml_est))
        est_pct = 100.0 * est_abs / est_denom if est_denom > 1e-12 else 0.0
        se_abs = abs(tc_se - ml_se)
        se_denom = max(abs(tc_se), abs(ml_se))
        se_pct = 100.0 * se_abs / se_denom if se_denom > 1e-12 else 0.0

        rows.append({
            "coef": tc_name,
            "mlogit_est": ml_est,
            "tc_est": tc_est,
            "est_abs_diff": est_abs,
            "est_pct_diff": est_pct,
            "mlogit_se": ml_se,
            "tc_se": tc_se,
            "se_abs_diff": se_abs,
            "se_pct_diff": se_pct,
        })

    diff_df = pd.DataFrame(rows)
    max_est_abs = float(diff_df["est_abs_diff"].max())
    max_est_pct = float(diff_df["est_pct_diff"].max())
    max_se_abs = float(diff_df["se_abs_diff"].max())
    max_se_pct = float(diff_df["se_pct_diff"].max())
    est_status = "PASS" if max_est_abs <= est_abs_tol else "FAIL"
    se_status = "PASS" if max_se_abs <= se_abs_tol else "FAIL"
    print(
        f"[compare_coefs] estimates: max |diff| = {max_est_abs:.3e} "
        f"({max_est_pct:.4f}%); tol {est_abs_tol:.0e} -> {est_status}"
    )
    print(
        f"[compare_coefs] std errs:  max |diff| = {max_se_abs:.3e} "
        f"({max_se_pct:.4f}%); tol {se_abs_tol:.0e} -> {se_status}"
    )
    return diff_df
