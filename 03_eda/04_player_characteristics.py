"""EDA 3-4: age, service time, and next-season salary relationships."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import seaborn as sns


AGE_ORDER = ["Under 25", "25-29", "30-34", "35+"]
SERVICE_ORDER = ["Under 3 years", "3-5 years", "6+ years", "Missing"]
STAGE_ORDER = ["pre_arb", "arbitration", "guaranteed", "no_longer_40man", "multiple"]


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


def load_main_sample(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, encoding="utf-8-sig")
    sample = data.loc[data["train_eligible_main_pa50"].eq(1)].copy()
    numeric = ["current_age", "current_service_total_days", "target_salary_total", "log1p_target_salary"]
    sample[numeric] = sample[numeric].apply(pd.to_numeric, errors="coerce")
    sample["service_years_equivalent"] = sample["current_service_total_days"] / 172
    sample["age_band"] = pd.cut(
        sample["current_age"],
        bins=[-np.inf, 25, 30, 35, np.inf],
        right=False,
        labels=AGE_ORDER,
    )
    sample["service_band"] = pd.cut(
        sample["service_years_equivalent"],
        bins=[-np.inf, 3, 6, np.inf],
        right=False,
        labels=SERVICE_ORDER[:3],
    ).astype("object")
    sample["service_band"] = sample["service_band"].fillna("Missing")
    if sample.empty or sample["target_salary_total"].lt(0).any():
        raise ValueError("The PA>=50 training sample is empty or contains a negative salary.")
    return sample


def salary_tick(log_value: float, _position: float) -> str:
    salary_millions = np.expm1(log_value) / 1_000_000
    return "" if salary_millions < 0 else f"${salary_millions:g}M"


def correlation_values(data: pd.DataFrame, feature: str) -> tuple[int, float, float]:
    clean = data[[feature, "log1p_target_salary"]].dropna()
    pearson = clean[feature].corr(clean["log1p_target_salary"])
    spearman = clean[feature].rank().corr(clean["log1p_target_salary"].rank())
    return len(clean), pearson, spearman


def save_continuous_figure(sample: pd.DataFrame, path: Path) -> pd.DataFrame:
    positive = sample.loc[sample["target_salary_total"].gt(0)].copy()
    sns.set_theme(style="whitegrid", context="notebook")
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.7), sharey=True, constrained_layout=True)
    specifications = [
        ("current_age", "Age", axes[0]),
        ("service_years_equivalent", "MLB service time (172-day years)", axes[1]),
    ]
    salary_ticks = np.log1p(np.array([0.5, 1, 2, 5, 10, 20, 40, 60]) * 1_000_000)
    rows: list[dict[str, object]] = []

    for feature, label, axis in specifications:
        panel = positive[[feature, "log1p_target_salary"]].dropna()
        n, pearson, spearman = correlation_values(positive, feature)
        sns.regplot(
            data=panel,
            x=feature,
            y="log1p_target_salary",
            order=2,
            ci=None,
            truncate=True,
            scatter_kws={"s": 14, "alpha": 0.20, "edgecolors": "none"},
            line_kws={"linewidth": 2.2, "color": "#b14b4b"},
            color="#4c78a8",
            ax=axis,
        )
        axis.set_title(f"{label}\nSpearman r={spearman:.2f}, n={n:,}")
        axis.set_xlabel(label)
        axis.set_ylabel("")
        axis.set_yticks(salary_ticks)
        axis.yaxis.set_major_formatter(FuncFormatter(salary_tick))
        rows.append(
            {
                "group": "all",
                "feature": feature,
                "n": n,
                "pearson_vs_log_salary": pearson,
                "spearman_vs_log_salary": spearman,
            }
        )

    axes[0].set_ylabel("Next-season salary ($ millions; log scale)")
    figure.suptitle("Player age and service time vs. next-season salary (PA >= 50)")
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return pd.DataFrame(rows)


def annotate_boxes(axis: plt.Axes, data: pd.DataFrame, group: str, order: list[str]) -> None:
    summary = data.groupby(group, observed=False)["target_salary_total"].agg(["count", "median"])
    upper = data.groupby(group, observed=False)["salary_millions"].quantile(0.75)
    for index, category in enumerate(order):
        if category not in summary.index or summary.loc[category, "count"] == 0:
            continue
        axis.text(
            index,
            upper.loc[category] + 1.0,
            f"${summary.loc[category, 'median'] / 1_000_000:.2f}M\nn={int(summary.loc[category, 'count']):,}",
            ha="center",
            va="bottom",
        )


def save_grouped_figure(sample: pd.DataFrame, path: Path) -> None:
    plot_data = sample.assign(salary_millions=sample["target_salary_total"] / 1_000_000)
    sns.set_theme(style="whitegrid", context="notebook")
    figure, axes = plt.subplots(1, 2, figsize=(14.0, 6.0), sharey=True, constrained_layout=True)
    panels = [
        ("age_band", AGE_ORDER, "Age band", "Next-season salary by age band"),
        ("service_band", SERVICE_ORDER, "MLB service-time band", "Next-season salary by service time"),
    ]
    colors = ["#4c78a8", "#f2a541", "#7a5aa6", "#4f8b6d"]

    for axis, (group, order, xlabel, title) in zip(axes, panels):
        sns.boxplot(
            data=plot_data,
            x=group,
            y="salary_millions",
            order=order,
            hue=group,
            hue_order=order,
            palette=colors,
            legend=False,
            showfliers=False,
            width=0.64,
            ax=axis,
        )
        sns.stripplot(
            data=plot_data,
            x=group,
            y="salary_millions",
            order=order,
            color="#25364a",
            alpha=0.12,
            size=2.0,
            jitter=0.22,
            ax=axis,
        )
        annotate_boxes(axis, plot_data, group, order)
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("")

    axes[0].set_ylabel("Next-season salary ($ millions)")
    figure.suptitle("Next-season salary across player age and service-time groups (PA >= 50)")
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def grouped_summary(sample: pd.DataFrame, group: str, order: list[str]) -> pd.DataFrame:
    summary = (
        sample.groupby(group, observed=False)["target_salary_total"]
        .agg(
            n="count",
            mean="mean",
            median="median",
            p25=lambda values: values.quantile(0.25),
            p75=lambda values: values.quantile(0.75),
        )
        .reindex(order)
        .reset_index()
        .rename(columns={group: "category"})
    )
    summary.insert(0, "characteristic", group)
    return summary


def correlations_by_stage(sample: pd.DataFrame) -> pd.DataFrame:
    positive = sample.loc[sample["target_salary_total"].gt(0)]
    rows: list[dict[str, object]] = []
    for stage in ["all", *STAGE_ORDER]:
        stage_data = positive if stage == "all" else positive.loc[positive["current_contract_stage"].eq(stage)]
        for feature in ["current_age", "service_years_equivalent"]:
            n, pearson, spearman = correlation_values(stage_data, feature)
            rows.append(
                {
                    "contract_stage": stage,
                    "feature": feature,
                    "n": n,
                    "pearson_vs_log_salary": pearson,
                    "spearman_vs_log_salary": spearman,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    figure_dir = output_dir / "figures"
    table_dir = output_dir / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    sample = load_main_sample(args.input.resolve())
    overall_correlations = save_continuous_figure(sample, figure_dir / "06_salary_vs_age_service.png")
    save_grouped_figure(sample, figure_dir / "07_salary_by_age_service_group.png")

    summaries = pd.concat(
        [grouped_summary(sample, "age_band", AGE_ORDER), grouped_summary(sample, "service_band", SERVICE_ORDER)],
        ignore_index=True,
    )
    summaries.to_csv(table_dir / "salary_by_age_service_group.csv", index=False, encoding="utf-8-sig")
    stage_correlations = correlations_by_stage(sample)
    stage_correlations.to_csv(
        table_dir / "age_service_salary_correlations.csv", index=False, encoding="utf-8-sig"
    )
    sample.loc[sample["current_service_total_days"].isna(), [
        "Season", "target_salary_season", "Player", "playerid", "current_contract_stage",
        "current_age", "target_salary_total",
    ]].to_csv(table_dir / "missing_service_time_cases.csv", index=False, encoding="utf-8-sig")

    print("Player-characteristic EDA complete")
    print(f"main_sample_rows: {len(sample)}")
    print(f"missing_service_time_rows: {sample['current_service_total_days'].isna().sum()}")
    for row in overall_correlations.itertuples(index=False):
        print(f"  {row.feature}: Spearman r={row.spearman_vs_log_salary:.4f}, n={row.n}")
    print(f"output_dir: {output_dir}")


if __name__ == "__main__":
    main()
