"""
Legacy helper that now defers to `model.fit(...)`.
"""
import warnings
from copy import deepcopy

from torch_choice.model.conditional_logit_model import ConditionalLogitModel
from torch_choice.model.nested_logit_model import NestedLogitModel


def run(
    model,
    dataset,
    dataset_test=None,
    batch_size=-1,
    learning_rate=0.01,
    num_epochs=5000,
    report_frequency=None,
    compute_std=True,
    return_final_training_log_likelihood=False,
    model_optimizer="Adam",
    print_summary: bool = True,
):
    """Backward compatible functional runner."""
    warnings.warn(
        "torch_choice.utils.run_helper.run is deprecated. "
        "Please use `model.fit(...)` instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    if not isinstance(model, (ConditionalLogitModel, NestedLogitModel)):
        raise TypeError(
            f"A model of type {type(model)} is not supported by this runner."
        )

    if not compute_std or return_final_training_log_likelihood:
        warnings.warn(
            "`compute_std` and `return_final_training_log_likelihood` are ignored. "
            "`model.fit(...)` now always reports regression tables and returns "
            "an EstimationOutput object.",
            DeprecationWarning,
            stacklevel=2,
        )

    trained_model = deepcopy(model)
    return trained_model.fit(
        dataset,
        dataset_val=None,
        dataset_test=dataset_test,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        model_optimizer=model_optimizer,
        report_frequency=report_frequency,
        backend="torch",
        print_summary=print_summary,
    )
