from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, Optional

import pandas as pd
import torch


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





