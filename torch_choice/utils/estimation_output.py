from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

# Avoid reporting p-values as 0 due to floating point underflow.
# 2e-16 matches the common R convention of printing "< 2.2e-16".
MIN_REPORTED_PVALUE = 2e-16
SCIENTIFIC_NOTATION_THRESHOLD = 1e-3


def _fmt_3digits(value: Any) -> str:
    """Format numbers with 3 digits, using scientific notation for tiny magnitudes."""
    if value is None:
        return "None"
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(x):
        return str(x)
    if x != 0.0 and abs(x) < SCIENTIFIC_NOTATION_THRESHOLD:
        return f"{x:.3e}"
    return f"{x:.3f}"


@dataclass
class EstimationOutput(dict):
    """Structured return object for model.fit, similar to HuggingFace's ModelOutput."""

    model: Any
    coef_summary: pd.DataFrame
    train_ll: float
    val_ll: Optional[float] = None
    test_ll: Optional[float] = None
    mean_dict: Optional[Dict[str, torch.Tensor]] = None
    std_dict: Optional[Dict[str, torch.Tensor]] = None
    epochs: Optional[int] = None
    learning_rate: Optional[float] = None
    batch_size: Optional[int] = None
    elapsed_time: Optional[float] = None
    backend: Optional[str] = None
    optimizer: Optional[str] = None

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if value is not None and field.name != "model":
                self[field.name] = value
        self["model"] = self.model

    def to_dict(self) -> Dict[str, Any]:
        """Return a shallow dictionary representation."""
        return dict(self)

    # ------------------------------------------------------------------
    # Pretty-print helpers for summarizing the estimation output in R/Stata style regression table format.
    # ------------------------------------------------------------------
    def _format_coef_table_for_display(self) -> Optional[pd.DataFrame]:
        if self.coef_summary is None:
            return None
        report_to_print = self.coef_summary.copy()
        if "Pr(>|z|)" in report_to_print.columns and "z-value" in report_to_print.columns:
            z_cutoff = float(norm.isf(MIN_REPORTED_PVALUE / 2.0))
            z_abs = report_to_print["z-value"].astype(float).abs()
            lt_mask = np.isfinite(z_abs.to_numpy()) & (z_abs.to_numpy() > z_cutoff)
            report_to_print["z-value"] = report_to_print["z-value"].map(_fmt_3digits)

            pvals = report_to_print["Pr(>|z|)"].astype(float).to_numpy()
            report_to_print["Pr(>|z|)"] = [
                (f"< {MIN_REPORTED_PVALUE:g}" if bool(is_lt) else _fmt_3digits(p))
                for p, is_lt in zip(pvals, lt_mask)
            ]
        return report_to_print

    def to_markdown_string(self) -> str:
        """Return a markdown-formatted regression summary (log-likelihood + coef table)."""
        lines = []
        lines.append("=" * 20 + " model results " + "=" * 20)
        lines.append(
            "Log-likelihood: "
            f"[Training] {_fmt_3digits(self.train_ll)}, "
            f"[Validation] {_fmt_3digits(self.val_ll)}, "
            f"[Test] {_fmt_3digits(self.test_ll)}"
        )

        report_to_print = self._format_coef_table_for_display()
        if report_to_print is not None:
            lines.append("")
            lines.append(report_to_print.to_markdown())
            lines.append("Significance codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_markdown_string()

    def __repr__(self) -> str:
        return self.to_markdown_string()





