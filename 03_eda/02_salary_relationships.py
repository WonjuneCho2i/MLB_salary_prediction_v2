"""EDA 3-2: contract-stage salary distributions and current-to-next salary relationships."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


STAGE_ORDER = ["pre_arb", "arbitration", "guaranteed", "no_longer_40man", "multiple"]
STAGE_LABELS = {
    "pre_arb": "Pre-Arb",
    "arbitration": "Arbitration",
    "guaranteed": "Guaranteed",
    "no_longer_40man": "No longer\non 40-man",
    "multiple": "Multiple",
}
STAGE_COLORS = {
    "pre_arb": "#4c78a8",
    "arbitration": "#f2a541",
    "guaranteed": "#7a5aa6",
    "no_longer_40man": "#c45b5b",
    "multiple": "#4f8b6d",
}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root / "02_data_preprocessing" / "model_ready" / "training_model_ready.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    return parser.parse_args()


def configure_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def load_main_sample(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, encoding="utf-8-sig")
    sample = data.loc[data["train_eligible_main_pa50"].eq(1)].copy()
    numeric = ["log1p_current_salary", "target_salary_total", "log1p_target_salary"]
    sample[numeric] = sample[numeric].apply(pd.to_numeric, errors="raise")
    sample["current_salary_total_reconstructed"] = np.expm1(sample["log1p_current_salary"])
    unknown = sorted(set(sample["current_contract_stage"]) - set(STAGE_ORDER))
    if unknown:
        raise ValueError(f"알 수 없는 계약 단계: {unknown}")
    return sample


def save_contract_stage_figure(sample: pd.DataFrame, path: Path) -> None:
    plot_data = sample.assign(salary_millions=sample["target_salary_total"] / 1_000_000)
    figure, axis = plt.subplots(figsize=(11.5, 6.2), constrained_layout=True)
    sns.boxplot(
        data=plot_data,
        x="current_contract_stage",
        y="salary_millions",
        order=STAGE_ORDER,
        palette=STAGE_COLORS,
        hue="current_contract_stage",
        hue_order=STAGE_ORDER,
        legend=False,
        showfliers=False,
        width=0.62,
        ax=axis,
    )
    sns.stripplot(
        data=plot_data,
        x="current_contract_stage",
        y="salary_millions",
        order=STAGE_ORDER,
        color="#25364a",
        alpha=0.13,
        size=2.1,
        jitter=0.22,
        ax=axis,
    )
    summaries = plot_data.groupby("current_contract_stage")["target_salary_total"].agg(["count", "median"])
    upper = plot_data.groupby("current_contract_stage")["salary_millions"].quantile(0.75)
    for index, stage in enumerate(STAGE_ORDER):
        if stage not in summaries.index:
            continue
        axis.text(
            index,
            upper.loc[stage] + 1.1,
            f"${summaries.loc[stage, 'median'] / 1_000_000:.2f}M\nn={int(summaries.loc[stage, 'count']):,}",
            ha="center",
            va="bottom",
        )
    axis.set_xticks(range(len(STAGE_ORDER)), [STAGE_LABELS[stage] for stage in STAGE_ORDER])
    axis.set(
        title="Next-season salary by current contract stage",
        xlabel="Current contract stage",
        ylabel="Next-season salary ($ millions)",
    )
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_relationship(axis: plt.Axes, data: pd.DataFrame, title: str, color: str) -> None:
    x = data["log1p_current_salary"]
    y = data["log1p_target_salary"]
    axis.scatter(x, y, s=14, alpha=0.24, color=color, edgecolors="none")
    low = min(x.min(), y.min())
    high = max(x.max(), y.max())
    axis.plot([low, high], [low, high], color="#5f6368", linewidth=1.2, linestyle="--")
    if len(data) >= 2 and x.nunique() > 1:
        slope, intercept = np.polyfit(x, y, 1)
        line_x = np.linspace(low, high, 100)
        axis.plot(line_x, slope * line_x + intercept, color=color, linewidth=2)
    pearson = x.corr(y)
    axis.set_title(f"{title}\nn={len(data):,}, Pearson r={pearson:.2f}")
    axis.set_xlim(low, high)
    axis.set_ylim(low, high)


def save_current_next_figure(sample: pd.DataFrame, path: Path) -> pd.DataFrame:
    positive = sample.loc[
        sample["log1p_current_salary"].gt(0) & sample["log1p_target_salary"].gt(0)
    ].copy()
    figure, axes = plt.subplots(2, 3, figsize=(14, 9.2), sharex=True, sharey=True, constrained_layout=True)
    panels = [("All stages", positive, "#4c78a8")]
    panels.extend(
        (STAGE_LABELS[stage].replace("\n", " "), positive.loc[positive["current_contract_stage"].eq(stage)], STAGE_COLORS[stage])
        for stage in STAGE_ORDER
    )
    for axis, (title, data, color) in zip(axes.flat, panels):
        plot_relationship(axis, data, title, color)
    for axis in axes[:, 0]:
        axis.set_ylabel("log1p(next-season salary)")
    for axis in axes[-1, :]:
        axis.set_xlabel("log1p(current salary)")
    figure.suptitle("Current salary vs. next-season salary (positive salaries only)")
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return positive


def correlation_row(label: str, data: pd.DataFrame) -> dict[str, object]:
    current = data["log1p_current_salary"]
    target = data["log1p_target_salary"]
    salary_change = data["target_salary_total"] - data["current_salary_total_reconstructed"]
    return {
        "contract_stage": label,
        "n": len(data),
        "pearson_log_salary": current.corr(target),
        "spearman_log_salary": current.rank().corr(target.rank()),
        "median_current_salary": data["current_salary_total_reconstructed"].median(),
        "median_target_salary": data["target_salary_total"].median(),
        "median_salary_change": salary_change.median(),
        "mean_absolute_salary_change": salary_change.abs().mean(),
    }


def save_tables(sample: pd.DataFrame, positive: pd.DataFrame, table_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    table_dir.mkdir(parents=True, exist_ok=True)
    contract_summary = (
        sample.groupby("current_contract_stage")
        .agg(
            n=("playerid", "count"),
            current_salary_mean=("current_salary_total_reconstructed", "mean"),
            current_salary_median=("current_salary_total_reconstructed", "median"),
            target_salary_mean=("target_salary_total", "mean"),
            target_salary_median=("target_salary_total", "median"),
            target_salary_p25=("target_salary_total", lambda values: values.quantile(0.25)),
            target_salary_p75=("target_salary_total", lambda values: values.quantile(0.75)),
        )
        .reindex(STAGE_ORDER)
        .reset_index()
    )
    correlation_rows = [correlation_row("all", positive)]
    correlation_rows.extend(
        correlation_row(stage, positive.loc[positive["current_contract_stage"].eq(stage)])
        for stage in STAGE_ORDER
    )
    correlations = pd.DataFrame(correlation_rows)
    contract_summary.to_csv(table_dir / "salary_by_contract_stage.csv", index=False, encoding="utf-8-sig")
    correlations.to_csv(table_dir / "current_next_salary_correlations.csv", index=False, encoding="utf-8-sig")
    return contract_summary, correlations


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    figure_dir = output_dir / "figures"
    table_dir = output_dir / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    sample = load_main_sample(args.input.resolve())
    save_contract_stage_figure(sample, figure_dir / "03_salary_by_contract_stage.png")
    positive = save_current_next_figure(sample, figure_dir / "04_current_vs_next_salary.png")
    contract_summary, correlations = save_tables(sample, positive, table_dir)
    print("Salary relationship EDA complete")
    print(f"main_sample_rows: {len(sample)}")
    print(f"positive_salary_rows_for_correlation: {len(positive)}")
    print(f"overall_pearson_log_salary: {correlations.loc[0, 'pearson_log_salary']:.4f}")
    print(f"overall_spearman_log_salary: {correlations.loc[0, 'spearman_log_salary']:.4f}")
    print("contract_stage_counts:")
    for row in contract_summary.itertuples(index=False):
        print(f"  {row.current_contract_stage}: {int(row.n)}")
    print(f"output_dir: {output_dir}")


if __name__ == "__main__":
    main()
