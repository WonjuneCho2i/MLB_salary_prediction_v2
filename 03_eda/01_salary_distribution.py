"""EDA 3-1: visualize contractual salary and log-salary distributions."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


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
    sample["target_salary_total"] = pd.to_numeric(sample["target_salary_total"], errors="raise")
    sample["log1p_target_salary"] = pd.to_numeric(sample["log1p_target_salary"], errors="raise")
    sample["target_salary_season"] = pd.to_numeric(sample["target_salary_season"], errors="raise").astype(int)
    if sample.empty or sample["target_salary_total"].lt(0).any():
        raise ValueError("주 모델 표본의 연봉이 비어 있거나 음수입니다.")
    return sample


def save_distribution_figure(sample: pd.DataFrame, path: Path) -> None:
    salary_millions = sample["target_salary_total"] / 1_000_000
    positive_sample = sample.loc[sample["target_salary_total"].gt(0)]
    log_salary = positive_sample["log1p_target_salary"]
    zero_count = int(sample["target_salary_total"].eq(0).sum())
    raw_mean, raw_median = salary_millions.mean(), salary_millions.median()
    log_mean, log_median = log_salary.mean(), log_salary.median()

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    sns.histplot(salary_millions, bins=35, color="#3568a8", edgecolor="white", ax=axes[0])
    axes[0].axvline(raw_mean, color="#c44e52", linewidth=2, label=f"Mean ${raw_mean:.1f}M")
    axes[0].axvline(raw_median, color="#2f7f5f", linewidth=2, label=f"Median ${raw_median:.1f}M")
    axes[0].set(title="Contractual salary", xlabel="Next-season salary ($ millions)", ylabel="Player-seasons")
    axes[0].legend(frameon=False)

    sns.histplot(log_salary, bins=35, color="#7a5aa6", edgecolor="white", ax=axes[1])
    axes[1].axvline(log_mean, color="#c44e52", linewidth=2, label=f"Mean {log_mean:.2f}")
    axes[1].axvline(log_median, color="#2f7f5f", linewidth=2, label=f"Median {log_median:.2f}")
    axes[1].set(
        title=f"Log-transformed positive salary (zero salary excluded: {zero_count})",
        xlabel="log1p(next-season salary)",
        ylabel="Player-seasons",
    )
    axes[1].legend(frameon=False)

    figure.suptitle(f"Main model sample (PA ≥ 50, n={len(sample):,})")
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_year_figure(sample: pd.DataFrame, path: Path) -> None:
    plot_data = sample.assign(salary_millions=sample["target_salary_total"] / 1_000_000)
    order = sorted(plot_data["target_salary_season"].unique())
    figure, axis = plt.subplots(figsize=(11.5, 5.8), constrained_layout=True)
    sns.boxplot(
        data=plot_data,
        x="target_salary_season",
        y="salary_millions",
        order=order,
        color="#6f93bd",
        width=0.62,
        showfliers=False,
        ax=axis,
    )
    sns.stripplot(
        data=plot_data,
        x="target_salary_season",
        y="salary_millions",
        order=order,
        color="#24364b",
        alpha=0.14,
        size=2.2,
        jitter=0.22,
        ax=axis,
    )
    axis.set(
        title="Next-season contractual salary by target year",
        xlabel="Salary season",
        ylabel="Salary ($ millions)",
    )
    medians = plot_data.groupby("target_salary_season")["target_salary_total"].median()
    upper_quartiles = plot_data.groupby("target_salary_season")["salary_millions"].quantile(0.75)
    for index, season in enumerate(order):
        axis.text(
            index,
            upper_quartiles.loc[season] + 1.1,
            f"${medians.loc[season] / 1_000_000:.2f}M",
            ha="center",
            va="bottom",
        )
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_tables(sample: pd.DataFrame, table_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    table_dir.mkdir(parents=True, exist_ok=True)
    salary = sample["target_salary_total"]
    log_salary = sample["log1p_target_salary"]
    positive = sample.loc[salary.gt(0)]
    overall = pd.DataFrame(
        [
            {
                "sample": "main_pa50",
                "n": len(sample),
                "salary_mean": salary.mean(),
                "salary_median": salary.median(),
                "salary_std": salary.std(),
                "salary_p25": salary.quantile(0.25),
                "salary_p75": salary.quantile(0.75),
                "salary_p90": salary.quantile(0.90),
                "salary_p95": salary.quantile(0.95),
                "salary_max": salary.max(),
                "salary_skewness": salary.skew(),
                "zero_salary_count": salary.eq(0).sum(),
                "log_salary_mean": log_salary.mean(),
                "log_salary_median": log_salary.median(),
                "log_salary_std": log_salary.std(),
                "log_salary_skewness": log_salary.skew(),
                "positive_salary_skewness": positive["target_salary_total"].skew(),
                "positive_log_salary_skewness": positive["log1p_target_salary"].skew(),
            }
        ]
    )
    yearly = (
        sample.groupby("target_salary_season")["target_salary_total"]
        .agg(n="count", mean="mean", median="median", minimum="min", maximum="max")
        .join(sample.groupby("target_salary_season")["target_salary_total"].quantile([0.25, 0.75, 0.90]).unstack())
        .rename(columns={0.25: "p25", 0.75: "p75", 0.90: "p90"})
        .reset_index()
    )
    overall.to_csv(table_dir / "salary_distribution_summary.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(table_dir / "salary_by_year_summary.csv", index=False, encoding="utf-8-sig")
    sample.loc[salary.eq(0), [
        "Season", "target_salary_season", "Player", "playerid", "current_contract_stage",
        "log1p_current_salary", "target_salary_total",
    ]].to_csv(table_dir / "zero_salary_cases.csv", index=False, encoding="utf-8-sig")
    return overall, yearly


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    figure_dir = output_dir / "figures"
    table_dir = output_dir / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    sample = load_main_sample(args.input.resolve())
    save_distribution_figure(sample, figure_dir / "01_salary_raw_vs_log.png")
    save_year_figure(sample, figure_dir / "02_salary_by_year.png")
    overall, yearly = save_tables(sample, table_dir)
    print("Salary distribution EDA complete")
    print(f"main_sample_rows: {len(sample)}")
    print(f"salary_mean: {overall.loc[0, 'salary_mean']:.2f}")
    print(f"salary_median: {overall.loc[0, 'salary_median']:.2f}")
    print(f"salary_skewness: {overall.loc[0, 'salary_skewness']:.4f}")
    print(f"log_salary_skewness: {overall.loc[0, 'log_salary_skewness']:.4f}")
    print(f"target_years: {yearly['target_salary_season'].min()}-{yearly['target_salary_season'].max()}")
    print(f"output_dir: {output_dir}")


if __name__ == "__main__":
    main()
