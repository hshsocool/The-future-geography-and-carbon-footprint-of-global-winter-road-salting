```python
"""
Generate Taylor diagrams for climate model evaluation.

This script compares observed air temperature with multiple model outputs
stored in CSV files. For each location, it generates a Taylor diagram and
identifies the best-performing model based on normalized standard deviation
difference, correlation, and normalized RMSE.



from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.projections import PolarAxes
from mpl_toolkits.axisartist import floating_axes, grid_finder


def setup_taylor_axes(fig: plt.Figure, subplot_location: int):
    """Create Taylor diagram axes."""
    transform = PolarAxes.PolarTransform()

    corr_locs = np.hstack((np.arange(1, 10) / 10.0, [0.95, 0.99]))
    theta_locs = np.arccos(corr_locs)

    corr_locator = grid_finder.FixedLocator(theta_locs)
    corr_formatter = grid_finder.DictFormatter(
        dict(zip(theta_locs, map(str, corr_locs)))
    )

    std_locs = np.arange(0, 2.0, 0.25)
    std_labels = [
        "0", "0.25", "0.50", "0.75", "REF", "1.25", "1.50", "1.75"
    ]

    std_locator = grid_finder.FixedLocator(std_locs)
    std_formatter = grid_finder.DictFormatter(
        dict(zip(std_locs, std_labels))
    )

    grid_helper = floating_axes.GridHelperCurveLinear(
        transform,
        extremes=(0, np.pi / 2, 0, 1.75),
        grid_locator1=corr_locator,
        tick_formatter1=corr_formatter,
        grid_locator2=std_locator,
        tick_formatter2=std_formatter,
    )

    ax = floating_axes.FloatingSubplot(fig, subplot_location, grid_helper=grid_helper)
    fig.add_subplot(ax)

    ax.axis["top"].set_axis_direction("bottom")
    ax.axis["top"].toggle(ticklabels=True, label=True)
    ax.axis["top"].major_ticklabels.set_axis_direction("top")
    ax.axis["top"].label.set_axis_direction("top")
    ax.axis["top"].label.set_text("Correlation")
    ax.axis["top"].label.set_fontsize(14)

    ax.axis["left"].set_axis_direction("bottom")
    ax.axis["left"].label.set_text("Normalized standard deviation")
    ax.axis["left"].label.set_fontsize(14)

    ax.axis["right"].set_axis_direction("top")
    ax.axis["right"].toggle(ticklabels=True)
    ax.axis["right"].major_ticklabels.set_axis_direction("left")

    ax.axis["bottom"].set_visible(False)
    ax.grid(True)

    polar_ax = ax.get_aux_axes(transform)

    rs, ts = np.meshgrid(
        np.linspace(0, 1.75, 100),
        np.linspace(0, np.pi / 2, 100),
    )

    rms = np.sqrt(1 + rs**2 - 2 * rs * np.cos(ts))
    contours = polar_ax.contour(ts, rs, rms, colors="gray", linestyles="--")
    plt.clabel(contours, inline=True, fontsize=10)

    theta = np.linspace(0, np.pi / 2)
    polar_ax.plot(theta, np.ones_like(theta), "k--")

    polar_ax.text(
        np.pi / 2 + 0.032,
        1.02,
        "1.00",
        size=10,
        ha="right",
        va="top",
        bbox=dict(boxstyle="square", ec="white", fc="white"),
    )

    return polar_ax


def calculate_model_metrics(
    observed: pd.Series,
    simulated: pd.Series,
) -> dict[str, float] | None:
    """Calculate standard deviation ratio, correlation, RMSE, and normalized RMSE."""
    valid = observed.notna() & simulated.notna()
    obs = observed[valid].astype(float)
    sim = simulated[valid].astype(float)

    if len(obs) < 2:
        return None

    obs_std = np.std(obs, ddof=0)
    sim_std = np.std(sim, ddof=0)

    if obs_std == 0 or sim_std == 0:
        return None

    std_ratio = sim_std / obs_std
    corr = np.corrcoef(obs, sim)[0, 1]
    rmse = np.sqrt(np.mean((sim - obs) ** 2))
    nrmse = rmse / obs_std

    return {
        "std_ratio": std_ratio,
        "correlation": corr,
        "rmse": rmse,
        "nrmse": nrmse,
    }


def plot_model_on_taylor_diagram(
    ax,
    metrics: dict[str, float],
    label: str,
    color: str,
) -> None:
    """Plot one model on a Taylor diagram."""
    theta = np.arccos(np.clip(metrics["correlation"], -1, 1))
    radius = metrics["std_ratio"]

    ax.plot(
        theta,
        radius,
        marker="o",
        markersize=8,
        linestyle="None",
        label=label,
        color=color,
    )


def infer_location_name(csv_path: Path) -> str:
    """Infer location name from a CSV filename."""
    parts = csv_path.stem.split("_")
    if len(parts) >= 4:
        return "_".join(parts[1:4])
    return csv_path.stem


def evaluate_csv_file(
    csv_path: Path,
    output_dir: Path,
    observed_column: str,
    model_start_column: int,
    model_end_column: int,
    dpi: int,
) -> dict[str, object] | None:
    """Evaluate all model columns in one CSV file and save a Taylor diagram."""
    df = pd.read_csv(csv_path)

    if observed_column not in df.columns:
        logging.warning("Observed column '%s' not found in %s", observed_column, csv_path)
        return None

    observed = df[observed_column]
    model_columns = list(df.columns[model_start_column:model_end_column])

    if not model_columns:
        logging.warning("No model columns found in %s", csv_path)
        return None

    location = infer_location_name(csv_path)

    fig = plt.figure(figsize=(16, 9), dpi=dpi)
    ax = setup_taylor_axes(fig, 111)

    colors = [
        "#E64B35FF", "#4DBBD5FF", "#00A087FF", "#3C5488FF", "#F39B7FFF",
        "#8491B4FF", "#91D1C2FF", "#DC0000FF", "#7E6148FF", "#B09C85FF",
        "#7CAE00", "#00BFC4", "#C77CFF", "#F8766D", "#619CFF",
    ]

    best_model = None
    best_metrics = None
    best_score = np.inf

    for idx, model_column in enumerate(model_columns):
        simulated = df[model_column]
        metrics = calculate_model_metrics(observed, simulated)

        if metrics is None:
            logging.warning("Skipping %s in %s due to insufficient valid data", model_column, csv_path)
            continue

        score = (
            abs(metrics["std_ratio"] - 1.0)
            + (1.0 - metrics["correlation"])
            + metrics["nrmse"]
        )

        if score < best_score:
            best_score = score
            best_model = model_column
            best_metrics = metrics

        plot_model_on_taylor_diagram(
            ax=ax,
            metrics=metrics,
            label=model_column,
            color=colors[idx % len(colors)],
        )

    if best_model is None or best_metrics is None:
        plt.close(fig)
        return None

    ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.10), fontsize=8)
    fig.text(0.40, 0.90, location, fontsize=16)

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / f"{csv_path.stem}.png"
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "Location": csv_path.stem,
        "Best Model": best_model,
        "Std Ratio": best_metrics["std_ratio"],
        "Correlation": best_metrics["correlation"],
        "RMSE": best_metrics["rmse"],
        "NRMSE": best_metrics["nrmse"],
        "Score": best_score,
    }


def process_all_subdirectories(
    input_dir: Path,
    output_dir: Path,
    observed_column: str,
    model_start_column: int,
    model_end_column: int,
    dpi: int,
) -> None:
    """Process all CSV files in all subdirectories."""
    subdirs = [p for p in input_dir.iterdir() if p.is_dir()]

    if not subdirs:
        raise FileNotFoundError(f"No subdirectories found in {input_dir}")

    for subdir in subdirs:
        logging.info("Processing folder: %s", subdir.name)

        subdir_output = output_dir / subdir.name
        best_model_records = []

        csv_files = sorted(subdir.glob("*.csv"))

        for csv_path in csv_files:
            result = evaluate_csv_file(
                csv_path=csv_path,
                output_dir=subdir_output,
                observed_column=observed_column,
                model_start_column=model_start_column,
                model_end_column=model_end_column,
                dpi=dpi,
            )

            if result is not None:
                best_model_records.append(result)

        if best_model_records:
            best_models_df = pd.DataFrame(best_model_records)
            best_models_path = output_dir / f"{subdir.name}_BestModels.csv"
            best_models_df.to_csv(best_models_path, index=False)
            logging.info("Saved best-model summary: %s", best_models_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Taylor diagrams and identify the best climate model."
    )

    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--observed-column", default="temp_air")
    parser.add_argument("--model-start-column", default=2, type=int)
    parser.add_argument("--model-end-column", default=21, type=int)
    parser.add_argument("--dpi", default=600, type=int)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    process_all_subdirectories(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        observed_column=args.observed_column,
        model_start_column=args.model_start_column,
        model_end_column=args.model_end_column,
        dpi=args.dpi,
    )

    logging.info("Taylor diagram generation completed.")


if __name__ == "__main__":
    main()
```
