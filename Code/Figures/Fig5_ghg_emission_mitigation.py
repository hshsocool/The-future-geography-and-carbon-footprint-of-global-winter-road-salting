"""
Plot cumulative GHG emission mitigation diagrams.

This script generates a 2 x 2 waterfall figure showing cumulative GHG emissions
from winter road salt production and transport operation under different
decarbonization scenarios and time horizons.

Expected Excel input
--------------------
The workbook should contain sheets named after scenarios, e.g.:
- SSP2-4.5
- SSP1-2.6

Each sheet should contain:
- Year or year
- BAU-Salt
- BAU-Full
- BAU-Empty
- Difference-Salt
- Difference-Operation

Units are assumed to be Mt CO2-eq per year unless otherwise specified.

Example
-------
python plot_ghg_emission_waterfall.py \
    --input-excel data/Climate-Technosphere.xlsx \
    --output outputs/figure5_mitigation.png \
    --output-svg outputs/figure5_mitigation.svg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# =============================================================================
# Default visual parameters
# =============================================================================

COLORS = {
    "production": "#6C8EA3",
    "operation": "#C7D6E0",
    "production_reduction": "#F4D999",
    "operation_reduction": "#C1D386",
    "operation_increase": "#F2958A",
    "budget_2c": "#5DAD9D",
    "budget_27c": "#225EA8",
    "exceedance": "#C0392B",
    "surplus": "#225EA8",
}

BAR_ALPHA = 0.75
GRID_ALPHA = 0.25


DEFAULT_BUDGETS = {
    "budget_27_2050": 468.22,
    "budget_2_2050": 395.96,
    "budget_27_2100": 1165.0,
    "budget_2_2100": 583.0,
}


def configure_matplotlib() -> None:
    """Set publication-style matplotlib defaults."""
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
        }
    )


def get_year_column(df: pd.DataFrame) -> str:
    """Return the year column name."""
    for column in ["Year", "year", "YEAR"]:
        if column in df.columns:
            return column

    raise ValueError("No year column found. Expected one of: 'Year', 'year', 'YEAR'.")


def filter_years(df: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    """Filter a dataframe to a year range."""
    year_column = get_year_column(df)
    return df[(df[year_column] >= start_year) & (df[year_column] <= end_year)].copy()


def read_scenario_data(input_excel: Path, sheet_name: str, start_year: int, end_year: int) -> pd.DataFrame:
    """Read and filter one scenario sheet."""
    df = pd.read_excel(input_excel, sheet_name=sheet_name)
    return filter_years(df, start_year, end_year)


def calculate_waterfall_components(df: pd.DataFrame) -> dict[str, float]:
    """Calculate cumulative emissions and mitigation components."""
    required_columns = [
        "BAU-Salt",
        "BAU-Full",
        "BAU-Empty",
        "Difference-Salt",
        "Difference-Operation",
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    bau_production = df["BAU-Salt"].sum()
    bau_operation = df["BAU-Full"].sum() + df["BAU-Empty"].sum()
    bau_total = bau_production + bau_operation

    production_reduction = df["Difference-Salt"].sum()
    operation_reduction = df["Difference-Operation"].sum()

    after_production = bau_total - production_reduction
    final_total = after_production - operation_reduction

    return {
        "bau_production": bau_production,
        "bau_operation": bau_operation,
        "bau_total": bau_total,
        "production_reduction": production_reduction,
        "operation_reduction": operation_reduction,
        "after_production": after_production,
        "final_total": final_total,
    }


def add_budget_gap_annotation(
    ax,
    x_final: float,
    x_left: float,
    x_right: float,
    final_total: float,
    budget_value: float,
    sheet_name: str,
    end_year: int,
) -> None:
    """Add exceedance or surplus annotation relative to a carbon budget."""
    y_min, y_max = ax.get_ylim()
    gap = final_total - budget_value
    gap_abs = abs(gap)

    arrow_color = COLORS["exceedance"] if final_total > budget_value else COLORS["surplus"]

    # For selected panels, place text under the final bar instead of drawing arrow.
    use_text_only = (
        (sheet_name == "SSP2-4.5" and end_year == 2050)
        or (sheet_name == "SSP1-2.6" and end_year == 2050)
        or (sheet_name == "SSP1-2.6" and end_year == 2100)
    )

    if use_text_only:
        y_shift = (y_max - y_min) * 0.055

        ax.text(
            x_final + 0.12,
            final_total - y_shift,
            f"{gap_abs:.1f} Mt",
            color=arrow_color,
            fontsize=16,
            ha="left",
            va="center",
            fontweight="bold",
            zorder=8,
        )
        return

    x_arrow = (x_left + x_right) / 2 - 0.03
    y_mid = (final_total + budget_value) / 2

    ax.annotate(
        "",
        xy=(x_arrow, final_total),
        xytext=(x_arrow, budget_value),
        arrowprops=dict(
            arrowstyle="<->",
            color=arrow_color,
            linewidth=1.6,
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=7,
    )

    ax.text(
        x_arrow + 0.105,
        y_mid,
        f"{gap_abs:.1f} Mt",
        color=arrow_color,
        fontsize=16,
        ha="left",
        va="center",
        fontweight="bold",
        zorder=8,
    )


def plot_technosphere_waterfall(
    ax,
    input_excel: Path,
    sheet_name: str,
    start_year: int,
    end_year: int,
    budget_value: float,
    budget_label: str,
    budget_color: str,
    panel_title: str | None,
    show_budget: bool = True,
) -> float:
    """Plot one cumulative-emission waterfall panel."""
    df = read_scenario_data(input_excel, sheet_name, start_year, end_year)
    components = calculate_waterfall_components(df)

    bau_production = components["bau_production"]
    bau_operation = components["bau_operation"]
    bau_total = components["bau_total"]
    production_reduction = components["production_reduction"]
    operation_reduction = components["operation_reduction"]
    after_production = components["after_production"]
    final_total = components["final_total"]

    x = np.array([0.0, 1.45, 2.90])
    bar_width = 0.73

    # BAU stacked bar
    ax.bar(
        x[0],
        bau_production,
        width=bar_width,
        color=COLORS["production"],
        alpha=BAR_ALPHA,
        edgecolor="none",
        zorder=3,
    )
    ax.bar(
        x[0],
        bau_operation,
        width=bar_width,
        bottom=bau_production,
        color=COLORS["operation"],
        alpha=BAR_ALPHA,
        edgecolor="none",
        zorder=3,
    )

    # Production-side reduction
    ax.bar(
        x[1],
        production_reduction,
        width=bar_width,
        bottom=after_production,
        color=COLORS["production_reduction"],
        alpha=BAR_ALPHA,
        edgecolor="none",
        zorder=3,
    )

    # Operation-side reduction or increase
    operation_color = (
        COLORS["operation_reduction"]
        if operation_reduction >= 0
        else COLORS["operation_increase"]
    )

    ax.bar(
        x[2],
        abs(operation_reduction),
        width=bar_width,
        bottom=final_total if operation_reduction >= 0 else after_production,
        color=operation_color,
        alpha=BAR_ALPHA,
        edgecolor="none",
        zorder=3,
    )

    # Connector lines
    ax.plot(
        [x[0] + bar_width / 2, x[1] - bar_width / 2],
        [bau_total, bau_total],
        color="#A8A8A8",
        linestyle="--",
        linewidth=1.0,
        zorder=4,
    )
    ax.plot(
        [x[1] + bar_width / 2, x[2] - bar_width / 2],
        [after_production, after_production],
        color="#A8A8A8",
        linestyle="--",
        linewidth=1.0,
        zorder=4,
    )

    # Final residual line
    half_width = bar_width / 2
    extend = 0.08
    x_left = x[2] - half_width - extend
    x_right = x[2] + half_width + extend

    ax.hlines(
        final_total,
        x_left,
        x_right,
        color="#333333",
        linewidth=1.6,
        zorder=5,
        capstyle="butt",
    )

    # Axes limits
    ax.set_xlim(-0.65, 3.45)
    ax.set_ylim(0, bau_total * 1.22)

    y_min, y_max = ax.get_ylim()
    x_min, x_max = ax.get_xlim()

    # Budget line
    if show_budget:
        ax.axhline(
            y=budget_value,
            color=budget_color,
            linestyle="--",
            linewidth=1.6,
            zorder=6,
        )

        budget_label_offset = (y_max - y_min) * 0.02
        ax.text(
            (x_min + x_max) / 2,
            budget_value - budget_label_offset,
            budget_label,
            color=budget_color,
            fontsize=16,
            ha="center",
            va="top",
            zorder=8,
        )

        add_budget_gap_annotation(
            ax=ax,
            x_final=x[2],
            x_left=x_left,
            x_right=x_right,
            final_total=final_total,
            budget_value=budget_value,
            sheet_name=sheet_name,
            end_year=end_year,
        )

    # Bar labels
    offset = bau_total * 0.025

    ax.text(
        x[0],
        bau_total + offset,
        f"{bau_total:.1f}",
        ha="center",
        va="bottom",
        fontsize=16,
        fontweight="bold",
    )

    ax.text(
        x[1],
        after_production + production_reduction / 2,
        f"-{production_reduction:.1f}",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
    )

    if sheet_name == "SSP2-4.5":
        label_offset = bau_total * 0.020

        ax.text(
            x[2],
            final_total + operation_reduction + label_offset,
            f"{abs(operation_reduction):.1f}",
            ha="center",
            va="bottom",
            fontsize=16,
            fontweight="bold",
        )
    else:
        ax.text(
            x[2],
            final_total + operation_reduction / 2,
            f"-{operation_reduction:.1f}",
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
        )

    if panel_title is not None:
        ax.set_title(panel_title, fontsize=16, pad=10)

    if sheet_name == "SSP1-2.6":
        first_label = "SSP1-2.6-BAU"
    elif sheet_name == "SSP2-4.5":
        first_label = "SSP2-4.5-BAU"
    else:
        first_label = "Baseline"

    ax.set_xticks(x)
    ax.set_xticklabels([first_label, "Production-side", "Operation-side"])

    ax.grid(axis="y", linestyle="-", linewidth=0.6, alpha=GRID_ALPHA)
    ax.set_axisbelow(True)

    return final_total


def add_panel_labels_and_axes_style(axes) -> None:
    """Add subplot labels and shared axis styling."""
    for ax, label in zip(axes, ["a", "b", "c", "d"]):
        ax.text(
            -0.10,
            1.08,
            label,
            transform=ax.transAxes,
            fontsize=24,
            fontweight="bold",
            va="top",
            ha="left",
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.tick_params(axis="both", width=0.8, length=4)


def build_legends():
    """Build legend handles."""
    patch_legend = [
        Patch(facecolor=COLORS["production"], alpha=BAR_ALPHA, label="Salt production"),
        Patch(facecolor=COLORS["operation"], alpha=BAR_ALPHA, label="Transport operation"),
        Patch(
            facecolor=COLORS["production_reduction"],
            alpha=BAR_ALPHA,
            label="Production-side reduction",
        ),
        Patch(
            facecolor=COLORS["operation_reduction"],
            alpha=BAR_ALPHA,
            label="Operation-side reduction",
        ),
        Patch(
            facecolor=COLORS["operation_increase"],
            alpha=BAR_ALPHA,
            label="Operation-side increase",
        ),
    ]

    line_legend = [
        Line2D(
            [0],
            [0],
            color=COLORS["exceedance"],
            linewidth=1.8,
            label="Carbon budget exceedance",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["surplus"],
            linewidth=1.8,
            label="Mitigation surplus",
        ),
    ]

    return patch_legend, line_legend


def plot_figure5(
    input_excel: Path,
    output: Path,
    output_svg: Path | None = None,
    dpi: int = 900,
    budget_27_2050: float = DEFAULT_BUDGETS["budget_27_2050"],
    budget_2_2050: float = DEFAULT_BUDGETS["budget_2_2050"],
    budget_27_2100: float = DEFAULT_BUDGETS["budget_27_2100"],
    budget_2_2100: float = DEFAULT_BUDGETS["budget_2_2100"],
) -> None:
    """Create the full 2 x 2 GHG mitigation figure."""
    configure_matplotlib()

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14.5, 11.0),
        facecolor="white",
    )

    ax_a, ax_b, ax_c, ax_d = axes.flatten()

    plot_technosphere_waterfall(
        ax=ax_a,
        input_excel=input_excel,
        sheet_name="SSP2-4.5",
        start_year=2024,
        end_year=2050,
        budget_value=budget_27_2050,
        budget_label="2.7°C budget (2024–2050)",
        budget_color=COLORS["budget_27c"],
        panel_title="Mitigation under SSP2-4.5",
        show_budget=True,
    )

    plot_technosphere_waterfall(
        ax=ax_b,
        input_excel=input_excel,
        sheet_name="SSP1-2.6",
        start_year=2024,
        end_year=2050,
        budget_value=budget_2_2050,
        budget_label="2°C budget (2024–2050)",
        budget_color=COLORS["budget_2c"],
        panel_title="Mitigation under SSP1-2.6",
        show_budget=True,
    )

    plot_technosphere_waterfall(
        ax=ax_c,
        input_excel=input_excel,
        sheet_name="SSP2-4.5",
        start_year=2024,
        end_year=2100,
        budget_value=budget_27_2100,
        budget_label="2.7°C budget (2024–2100)",
        budget_color=COLORS["budget_27c"],
        panel_title=None,
        show_budget=True,
    )

    plot_technosphere_waterfall(
        ax=ax_d,
        input_excel=input_excel,
        sheet_name="SSP1-2.6",
        start_year=2024,
        end_year=2100,
        budget_value=budget_2_2100,
        budget_label="2°C budget (2024–2100)",
        budget_color=COLORS["budget_2c"],
        panel_title=None,
        show_budget=True,
    )

    patch_legend, line_legend = build_legends()

    ax_a.legend(
        handles=patch_legend,
        frameon=False,
        fontsize=16,
        loc="lower center",
        bbox_to_anchor=(0.75, 0.09),
        ncol=1,
        handlelength=1.6,
        handletextpad=0.7,
        labelspacing=0.8,
    )

    ax_b.legend(
        handles=line_legend,
        frameon=False,
        fontsize=16,
        loc="lower center",
        bbox_to_anchor=(0.75, 0.25),
        ncol=1,
        handlelength=2.2,
        handletextpad=0.7,
        labelspacing=0.8,
    )

    add_panel_labels_and_axes_style([ax_a, ax_b, ax_c, ax_d])

    ax_a.set_ylabel("Cumulative emissions (2024–2050)\n(Mt CO$_2$-eq)")
    ax_c.set_ylabel("Cumulative emissions (2024–2100)\n(Mt CO$_2$-eq)")
    ax_b.set_ylabel("")
    ax_d.set_ylabel("")

    plt.subplots_adjust(
        left=0.07,
        right=0.98,
        top=0.94,
        bottom=0.07,
        wspace=0.17,
        hspace=0.24,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")

    if output_svg is not None:
        output_svg.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_svg,
            format="svg",
            dpi=300,
            bbox_inches="tight",
            metadata={"Date": None},
        )

    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot cumulative GHG emission mitigation waterfall diagrams."
    )

    parser.add_argument("--input-excel", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--output-svg", default=None, type=Path)
    parser.add_argument("--dpi", default=900, type=int)

    parser.add_argument("--budget-27-2050", default=DEFAULT_BUDGETS["budget_27_2050"], type=float)
    parser.add_argument("--budget-2-2050", default=DEFAULT_BUDGETS["budget_2_2050"], type=float)
    parser.add_argument("--budget-27-2100", default=DEFAULT_BUDGETS["budget_27_2100"], type=float)
    parser.add_argument("--budget-2-2100", default=DEFAULT_BUDGETS["budget_2_2100"], type=float)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    plot_figure5(
        input_excel=args.input_excel,
        output=args.output,
        output_svg=args.output_svg,
        dpi=args.dpi,
        budget_27_2050=args.budget_27_2050,
        budget_2_2050=args.budget_2_2050,
        budget_27_2100=args.budget_27_2100,
        budget_2_2100=args.budget_2_2100,
    )


if __name__ == "__main__":
    main()
