"""Lightweight reimplementation of the v2 benchmark visualization notebook."""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import pandas as pd

try:
    import seaborn as sns  # type: ignore
except Exception:
    sns = None

try:
    import scienceplots  # noqa: F401

    plt.style.use(["science", "bright"])
except Exception as exc:
    warnings.warn(f"scienceplots style unavailable ({exc}); using default matplotlib style.")
    plt.style.use("default")

# configurations for the plots.
font = {"size": 22}
matplotlib.rc("font", **font)
plt.rcParams["legend.fontsize"] = 15
plt.rcParams["legend.loc"] = "upper left"
plt.rcParams["figure.dpi"] = 500
plt.rcParams["text.usetex"] = False

# ignore user warnings
warnings.filterwarnings("ignore", category=UserWarning)


R_FILES = {
    "num_items": "R_performance_items.csv",
    "num_params": "R_performance_params.csv",
    "num_records": "R_performance_records.csv",
}

TORCH_FILES = {
    "num_items": [
        "torch_choice_performance_num_items_experiment_small.csv",
        "torch_choice_performance_num_items_experiment_large.csv",
    ],
    "num_params": [
        "torch_choice_performance_num_params_experiment_small.csv",
        "torch_choice_performance_num_params_experiment_large.csv",
    ],
    "num_records": [
        "torch_choice_performance_num_records_experiment_small.csv",
        "torch_choice_performance_num_records_experiment_large.csv",
    ],
}


def parse_r_formula(formula: str) -> str:
    """Convert an R-style formula to our standard representation used in torch-choice."""
    if "user_latent_0" in formula and "item_latent_0" not in formula:
        return "(user_latents|item)"
    elif "item_latent_0" in formula and "user_latent_0" not in formula:
        return "(item_latents|constant)"
    elif "user_latent_0" in formula and "item_latent_0" in formula:
        return "(user_latents|item) + (item_latents|constant)"
    else:
        raise ValueError(f"Unknown formula: {formula}")


def generate_latex_representation_formula(input_formula: str) -> str:
    """Provide a nicer LaTeX string for figure legends."""
    if input_formula == "(item_latents|constant)":
        return r"$\mu_{uis} = \beta^\top \mathbf{z}_i$"
    elif input_formula == "(user_latents|item)":
        return r"$\mu_{uis} = \alpha_i^\top \mathbf{x}_u$"
    elif input_formula == "(user_latents|item) + (item_latents|constant)":
        return r"$\mu_{uis} = \alpha_i^\top \mathbf{x}_u + \beta^\top \mathbf{z}_i$"
    else:
        return input_formula


def _read_required(path: Path, label: str) -> Optional[pd.DataFrame]:
    """Read a CSV; fail fast normally, but in SMOKE_TEST tolerate missing/empty."""
    is_smoke = os.environ.get("SMOKE_TEST") == "1"
    if not path.exists():
        if is_smoke:
            print(f"[WARN] Skipping missing {label} (SMOKE_TEST=1): {path}")
            return None
        raise FileNotFoundError(f"Missing required input: {label} at {path}")
    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        if is_smoke:
            print(f"[WARN] Skipping empty {label} (SMOKE_TEST=1): {path}")
            return None
        raise ValueError(f"{label} at {path} is empty.")

    return df


def transform_time_to_ratio(df: pd.DataFrame, baseline_parameter: str) -> Tuple[pd.DataFrame, float]:
    """Match the notebook: normalize time by the baseline time (min parameter) per formula."""
    group_min = df.groupby("formula")[baseline_parameter].min()
    unique_baselines = group_min.unique()
    if len(unique_baselines) != 1:
        raise ValueError(
            f"Expected the same minimum '{baseline_parameter}' across all formulas, but got: {group_min.to_dict()}"
        )
    baseline_parameter_value = unique_baselines[0]
    baseline_times = (
        df[df[baseline_parameter] == baseline_parameter_value]
        .groupby("formula")["time"]
        .mean()
        .reset_index()
        .rename(columns={"time": "baseline_time"})
    )
    df = df.merge(baseline_times, on="formula", how="left", validate="m:1")
    if df["baseline_time"].isnull().any():
        raise ValueError(
            "Some formulas do not have a baseline time. Check that each formula has at least one record with the "
            f"minimum '{baseline_parameter}' value."
        )
    df["time"] = df["time"] / df["baseline_time"]
    return df.drop(columns=["baseline_time"]), baseline_parameter_value


def configure_axes(ax, log_scale: bool, y_log: bool = True) -> None:
    if log_scale:
        ax.set_xscale("log")
        if y_log:
            ax.set_yscale("log")
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_major_formatter(ScalarFormatter())
    else:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, linestyle="-", alpha=0.4)
    ax.tick_params(axis="both", which="major", labelsize=12, width=1.5, length=5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)


def plot_time_on_ax(
    ax,
    df: pd.DataFrame,
    param_display_name: str,
    color_mapping: Dict,
    report_ratio: bool,
    baseline_value: float | None,
    log_scale: bool,
) -> None:
    """Match the notebook: one curve per formula with mean±std error bars."""
    for formula in sorted(df["formula"].unique()):
        sub_df = df[df["formula"] == formula]
        stats = sub_df.groupby("parameter").agg({"time": ["mean", "std"]}).reset_index()
        stats.columns = ["parameter", "mean_time", "std_time"]
        stats = stats.sort_values("parameter")

        ax.errorbar(
            stats["parameter"],
            stats["mean_time"],
            yerr=stats["std_time"],
            label=sub_df["formula_display"].iloc[0],
            color=color_mapping[formula],
            marker="o",
            markersize=8,
            linewidth=2.5,
            capsize=5,
            alpha=0.9,
            markeredgewidth=1.5,
        )

    ax.set_xlabel(param_display_name, fontsize=14, labelpad=10)
    ylabel = "Time (seconds)" if not report_ratio else f"Time Ratio (Baseline: {int(baseline_value):,})"
    ax.set_ylabel(ylabel, fontsize=14, labelpad=10)
    configure_axes(ax, log_scale)
    ax.legend(
        title="Model Specification",
        fontsize=12,
        title_fontsize=13,
        frameon=True,
        shadow=False,
        edgecolor="black",
        loc="upper left",
    )
    ax.figure.tight_layout()


def create_epochs_figure(df: pd.DataFrame, param_display_name: str, color_mapping: Dict, log_scale: bool, plot_title: str):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    for formula in sorted(df["formula"].unique()):
        sub_df = df[df["formula"] == formula]
        stats = sub_df.groupby("parameter").agg({"epochs_run": ["mean", "std"]}).reset_index()
        stats.columns = ["parameter", "mean_epochs", "std_epochs"]
        stats = stats.sort_values("parameter")

        ax.errorbar(
            stats["parameter"],
            stats["mean_epochs"],
            yerr=stats["std_epochs"],
            label=sub_df["formula_display"].iloc[0],
            color=color_mapping[formula],
            marker="s",
            markersize=8,
            linewidth=2.5,
            capsize=5,
            alpha=0.9,
            markeredgewidth=1.5,
        )

    ax.set_xlabel(param_display_name, fontsize=14, labelpad=10)
    ax.set_ylabel("Epochs Until Convergence (Torch-Choice Only)", fontsize=14, labelpad=10)
    configure_axes(ax, log_scale)
    ax.legend(
        title="Model Specification",
        fontsize=12,
        title_fontsize=13,
        frameon=True,
        shadow=False,
        edgecolor="black",
        loc="upper left",
    )
    if plot_title:
        fig.suptitle(f"{plot_title} - Epochs Until Convergence", fontsize=16)
    fig.tight_layout()
    return fig


def create_loss_figure(
    df: pd.DataFrame,
    param_display_name: str,
    color_mapping: Dict,
    report_ratio: bool,
    log_scale: bool,
    plot_title: str,
):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    for formula in sorted(df["formula"].unique()):
        sub_df = df[df["formula"] == formula]
        stats = sub_df.groupby("parameter").agg({"loss": ["mean", "std"]}).reset_index()
        stats.columns = ["parameter", "mean_loss", "std_loss"]
        stats = stats.sort_values("parameter")

        ax.errorbar(
            stats["parameter"],
            stats["mean_loss"],
            yerr=stats["std_loss"],
            label=sub_df["formula_display"].iloc[0],
            color=color_mapping[formula],
            marker="D",
            markersize=8,
            linewidth=2.5,
            capsize=5,
            alpha=0.9,
            markeredgewidth=1.5,
        )

    loss_label = "Negative Log-Likelihood" + (" Ratio" if report_ratio else "")
    ax.set_xlabel(param_display_name, fontsize=14, labelpad=10)
    ax.set_ylabel(loss_label, fontsize=14, labelpad=10)
    configure_axes(ax, log_scale, y_log=False)
    ax.legend(
        title="Model Specification",
        fontsize=12,
        title_fontsize=13,
        frameon=True,
        shadow=False,
        edgecolor="black",
        loc="upper left",
    )
    if plot_title:
        fig.suptitle(f"{plot_title} - Loss Benchmark", fontsize=16)
    fig.tight_layout()
    return fig


def clean_title(original_title: str) -> str:
    return original_title.replace(" ", "_").replace("-", "_").lower()


def visualize_benchmarks_combined(
    csv_path: Path,
    parameter: str,
    time_ax,
    output_path: Path,
    report_ratio: bool = False,
    log_scale: bool = False,
    plot_title: str = "",
) -> Tuple[Optional[plt.Figure], Optional[plt.Figure]]:
    """Notebook-equivalent visualization for a single CSV on a provided axis."""
    assert parameter in ["num_items", "num_params", "num_records"]

    df = _read_required(csv_path, label=f"{plot_title} ({parameter})")
    if df is None:
        time_ax.text(
            0.5,
            0.5,
            f"Missing input\n{csv_path.name}",
            transform=time_ax.transAxes,
            ha="center",
            va="center",
        )
        time_ax.set_axis_off()
        return None, None

    # Homogenize the experiment log data from R and torch-choice.
    if "final_likelihood" in df.columns:
        df["loss"] = -df["final_likelihood"]
        df["formula"] = df["formula"].apply(parse_r_formula)
    elif "best_loss" in df.columns:
        df["loss"] = df["best_loss"]
    else:
        raise ValueError(
            "Unrecognized data format. CSV must contain either 'final_likelihood' (R) or 'best_loss' (torch-choice)."
        )

    # Create consistent parameter column and formula display.
    df[parameter] = pd.to_numeric(df[parameter])
    df["parameter"] = pd.to_numeric(df[parameter])
    df["formula_display"] = df["formula"].apply(generate_latex_representation_formula)

    # Time transformations.
    if report_ratio:
        df, baseline_parameter_value = transform_time_to_ratio(df, parameter)
    else:
        baseline_parameter_value = None

    param_display_name = {
        "num_records": "Number of Records",
        "num_params": "Number of Latent Dimensions",
        "num_items": "Number of Items",
    }[parameter]

    unique_formulas = sorted(df["formula"].unique())
    if sns is not None:
        palette = sns.color_palette("deep", len(unique_formulas))
        formula_color_mapping = {formula: palette[i] for i, formula in enumerate(unique_formulas)}
    else:
        colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3"])
        formula_color_mapping = {formula: colors[i % len(colors)] for i, formula in enumerate(unique_formulas)}

    plot_time_on_ax(time_ax, df, param_display_name, formula_color_mapping, report_ratio, baseline_parameter_value, log_scale)
    if plot_title:
        gpu_info = ""
        if "gpu_name" in df.columns:
            non_null_gpu_names = df["gpu_name"].dropna().unique()
            if len(non_null_gpu_names) == 0:
                gpu_info = " - GPU: N/A"
            elif len(non_null_gpu_names) == 1:
                gpu_info = f" - GPU: {non_null_gpu_names[0]}"
            else:
                raise ValueError(f"Inconsistent GPU models in benchmarks, got {df['gpu_name'].unique()}")
        time_ax.set_title(f"{plot_title} - Time Benchmark{gpu_info}", fontsize=16)

    epochs_fig = create_epochs_figure(df, param_display_name, formula_color_mapping, log_scale, plot_title) if "epochs_run" in df.columns else None
    loss_fig = create_loss_figure(df, param_display_name, formula_color_mapping, report_ratio, log_scale, plot_title)

    # Debug model size plot (notebook style: one panel per formula).
    if "parameter_count" in df.columns:
        fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(12, 6), dpi=300)
        for i, fml in enumerate(df["formula_display"].unique()):
            sub_df = df[df["formula_display"] == fml]
            if sns is None:
                raise RuntimeError("seaborn is required for debug_model_size plots to match the notebook.")
            sns.lineplot(data=sub_df, x=parameter, y="parameter_count", markers=True, dashes=False, ax=axes[i])
            axes[i].set_title(f"{plot_title} - Parameter Count \n {fml}", fontsize=16)
        fig.savefig(output_path / f"debug_model_size_{parameter}.pdf")
        plt.close(fig)

    return epochs_fig, loss_fig


def _concat_existing(*dfs: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    frames = [df for df in dfs if df is not None]
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _write_likelihood_alignment(
    parameter: str,
    r_df: pd.DataFrame,
    torch_df: pd.DataFrame,
    output_path: Path,
) -> Optional[pd.DataFrame]:
    """Compare R final_likelihood vs Torch best_loss and write a CSV."""
    param_col = parameter
    if param_col not in r_df.columns or param_col not in torch_df.columns:
        return None

    # Standardize R formula strings to match torch-choice formulas for joining.
    r_df = r_df.copy()
    if "final_likelihood" in r_df.columns and "formula" in r_df.columns:
        r_df["formula"] = r_df["formula"].apply(parse_r_formula)

    r_group = (
        r_df.groupby([param_col, "formula"])
        .agg(
            r_final_likelihood=("final_likelihood", "mean"),
            r_seeds=("seed", "nunique") if "seed" in r_df.columns else ("time", "count"),
        )
        .reset_index()
    )
    torch_group = (
        torch_df.groupby([param_col, "formula"])
        .agg(
            torch_best_loss=("best_loss", "mean"),
            torch_seeds=("seed", "nunique") if "seed" in torch_df.columns else ("time", "count"),
        )
        .reset_index()
    )
    torch_group["torch_log_likelihood"] = -torch_group["torch_best_loss"]

    merged = pd.merge(torch_group, r_group, on=[param_col, "formula"], how="outer")
    merged["parameter"] = parameter
    merged["loglik_delta_r_minus_torch"] = merged["r_final_likelihood"] - merged["torch_log_likelihood"]
    merged = merged[
        [
            "parameter",
            param_col,
            "formula",
            "torch_best_loss",
            "torch_log_likelihood",
            "r_final_likelihood",
            "loglik_delta_r_minus_torch",
            "torch_seeds",
            "r_seeds",
        ]
    ]

    output_path.mkdir(parents=True, exist_ok=True)
    out_file = output_path / f"likelihood_alignment_{parameter}.csv"
    merged.to_csv(out_file, index=False)
    return merged


def visualize(
    torch_results: Path,
    r_results: Path,
    output_path: Path,
) -> None:
    """Generate figures following the notebook's layout + filenames."""
    output_path.mkdir(parents=True, exist_ok=True)

    # Main notebook figures.
    for report_ratio in [True, False]:
        for parameter in ["num_items", "num_records", "num_params"]:
            fig = plt.figure(tight_layout=False, figsize=(15, 8), dpi=300)
            gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.2], hspace=0.3)
            axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]

            for i, (title, csv_path) in enumerate(
                [
                    ("Torch-Choice Small Scale", torch_results / TORCH_FILES[parameter][0]),
                    ("R Small Scale", r_results / R_FILES[parameter]),
                    ("Torch-Choice Large Scale", torch_results / TORCH_FILES[parameter][1]),
                ]
            ):
                ax = axes[i]
                epochs_fig, loss_fig = visualize_benchmarks_combined(
                    csv_path=csv_path,
                    parameter=parameter,
                    time_ax=ax,
                    output_path=output_path,
                    report_ratio=report_ratio,
                    log_scale=False,
                    plot_title=title,
                )

                prefix = "time_ratio" if report_ratio else "absolute_time"
                if epochs_fig is not None:
                    epochs_fig.savefig(
                        output_path / f"{prefix}_{clean_title(title)}_{parameter}_epochs.pdf",
                        bbox_inches="tight",
                        dpi=300,
                    )
                    plt.close(epochs_fig)
                if loss_fig is not None:
                    loss_fig.savefig(
                        output_path / f"{prefix}_{clean_title(title)}_{parameter}_loss.pdf",
                        bbox_inches="tight",
                        dpi=300,
                    )
                    plt.close(loss_fig)

            plt.subplots_adjust(top=0.95, bottom=0.1)
            prefix = "time_ratio" if report_ratio else "absolute_time"
            fig.savefig(
                output_path / f"{prefix}_{parameter}_time_cost_benchmark.pdf",
                bbox_inches="tight",
                dpi=300,
            )
            plt.close(fig)

    # Likelihood alignment table(s) (extra output vs notebook; requested for validation).
    alignment_frames: List[pd.DataFrame] = []
    for parameter in ["num_items", "num_records", "num_params"]:
        r_df = _read_required(r_results / R_FILES[parameter], label=f"R results for {parameter}")
        torch_small = _read_required(
            torch_results / TORCH_FILES[parameter][0],
            label=f"Torch-Choice small results for {parameter}",
        )
        torch_large = _read_required(
            torch_results / TORCH_FILES[parameter][1],
            label=f"Torch-Choice large results for {parameter}",
        )
        torch_all = _concat_existing(torch_small, torch_large)

        # In SMOKE_TEST mode, we may be missing large files; only write alignment if we have enough.
        if r_df is None or torch_all is None:
            continue

        alignment_df = _write_likelihood_alignment(parameter=parameter, r_df=r_df, torch_df=torch_all, output_path=output_path)
        if alignment_df is not None:
            alignment_frames.append(alignment_df)

    if alignment_frames:
        combined = pd.concat(alignment_frames, ignore_index=True)
        combined.to_csv(output_path / "likelihood_alignment.csv", index=False)


def build_arg_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate benchmark visualizations (v2).", add_help=add_help)
    parser.add_argument(
        "--torch-results",
        type=Path,
        required=True,
        help="Directory containing torch_choice_performance_*.csv outputs.",
    )
    parser.add_argument(
        "--r-results",
        type=Path,
        required=True,
        help="Directory containing R_performance_*.csv outputs.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("./benchmark_figures"),
        help="Directory to write PDF figures.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    visualize(
        torch_results=args.torch_results,
        r_results=args.r_results,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()

