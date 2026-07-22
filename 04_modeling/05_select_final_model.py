"""Modeling 4-5: diagnose candidates and select the final forecasting model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import seaborn as sns


FINAL_MODEL = "gradient_boosting__log_change_from_current"
NAIVE_MODEL = "naive_current_salary"
STAGE_ORDER = ["pre_arb", "arbitration", "guaranteed", "no_longer_40man", "multiple"]
STAGE_LABELS = {
    "pre_arb": "Pre-Arb",
    "arbitration": "Arbitration",
    "guaranteed": "Guaranteed",
    "no_longer_40man": "No longer\non 40-man",
    "multiple": "Multiple",
}
SALARY_BAND_ORDER = ["Under $1M", "$1M-$5M", "$5M-$10M", "$10M-$20M", "$20M+"]
MODEL_LABELS = {
    "gradient_boosting__log_change_from_current": "Gradient Boosting\n(change)",
    "extra_trees__log_change_from_current": "Extra Trees\n(change)",
    "naive_current_salary": "Current salary\ncarry-forward",
    "random_forest__direct_log_salary": "Random Forest",
    "lightgbm__direct_log_salary": "LightGBM",
    "xgboost__direct_log_salary": "XGBoost",
    "lasso": "Lasso",
    "elastic_net": "Elastic Net",
    "ridge": "Ridge",
    "linear_previous_salary": "Previous-salary\nlinear",
    "dummy_median_log": "Median dummy",
}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    modeling = project_root / "04_modeling"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison",
        type=Path,
        default=modeling / "results" / "all_model_comparison_through_4_4.csv",
    )
    parser.add_argument(
        "--tree-predictions",
        type=Path,
        default=modeling / "predictions" / "tree_out_of_fold_predictions.csv",
    )
    parser.add_argument(
        "--baseline-predictions",
        type=Path,
        default=modeling / "predictions" / "baseline_out_of_fold_predictions.csv",
    )
    parser.add_argument(
        "--tree-fold-metrics",
        type=Path,
        default=modeling / "results" / "tree_selected_fold_metrics.csv",
    )
    parser.add_argument(
        "--baseline-fold-metrics",
        type=Path,
        default=modeling / "results" / "baseline_fold_metrics.csv",
    )
    parser.add_argument(
        "--tree-tuning",
        type=Path,
        default=modeling / "results" / "tree_hyperparameter_summary.csv",
    )
    parser.add_argument(
        "--feature-importance",
        type=Path,
        default=modeling / "results" / "tree_feature_importance_by_fold.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=modeling)
    return parser.parse_args()


def add_salary_band(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["salary_band"] = pd.cut(
        result["target_salary_total"],
        bins=[-np.inf, 1_000_000, 5_000_000, 10_000_000, 20_000_000, np.inf],
        right=False,
        labels=SALARY_BAND_ORDER,
    )
    return result


def error_summary(data: pd.DataFrame, group: str, order: list[object] | None = None) -> pd.DataFrame:
    result = (
        data.groupby(group, observed=False)
        .agg(
            n=("playerid", "count"),
            mae_dollars=("absolute_error_dollars", "mean"),
            median_absolute_error_dollars=("absolute_error_dollars", "median"),
            mean_error_dollars=("signed_error_dollars", "mean"),
            actual_salary_median=("target_salary_total", "median"),
            predicted_salary_median=("predicted_salary_dollars", "median"),
        )
    )
    if order is not None:
        result = result.reindex(order)
    return result.reset_index()


def salary_tick(log_value: float, _position: float) -> str:
    millions = np.expm1(log_value) / 1_000_000
    return "" if millions < 0 else f"${millions:g}M"


def save_comparison_figure(comparison: pd.DataFrame, path: Path) -> None:
    plot = comparison.sort_values("fold_mean_mae_dollars", ascending=False).copy()
    plot["label"] = plot["model"].map(MODEL_LABELS).fillna(plot["model"])
    plot["mae_millions"] = plot["fold_mean_mae_dollars"] / 1_000_000
    colors = ["#b14b4b" if model == FINAL_MODEL else "#6f93bd" for model in plot["model"]]
    sns.set_theme(style="whitegrid", context="notebook")
    figure, axis = plt.subplots(figsize=(11.5, 7.0), constrained_layout=True)
    bars = axis.barh(plot["label"], plot["mae_millions"], color=colors)
    for bar, value in zip(bars, plot["mae_millions"]):
        axis.text(value + 0.05, bar.get_y() + bar.get_height() / 2, f"${value:.2f}M", va="center")
    naive = float(comparison.loc[comparison["model"].eq(NAIVE_MODEL), "fold_mean_mae_dollars"].iloc[0]) / 1_000_000
    axis.axvline(naive, color="#555555", linestyle="--", linewidth=1.4, label=f"Carry-forward baseline ${naive:.2f}M")
    axis.set(title="Temporal-validation MAE across all candidate models", xlabel="Four-fold mean MAE ($ millions)", ylabel="")
    axis.legend(frameon=False)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_diagnostic_figure(
    final_predictions: pd.DataFrame,
    yearly_comparison: pd.DataFrame,
    stage_summary: pd.DataFrame,
    band_summary: pd.DataFrame,
    path: Path,
) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    figure, axes = plt.subplots(2, 2, figsize=(14.0, 10.2), constrained_layout=True)

    axis = axes[0, 0]
    scatter_data = final_predictions.loc[final_predictions["target_salary_total"].gt(0)]
    axis.scatter(
        scatter_data["log1p_target_salary"],
        scatter_data["predicted_log1p_salary"],
        s=15,
        alpha=0.24,
        color="#4c78a8",
        edgecolors="none",
    )
    low = min(scatter_data["log1p_target_salary"].min(), scatter_data["predicted_log1p_salary"].min())
    high = max(scatter_data["log1p_target_salary"].max(), scatter_data["predicted_log1p_salary"].max())
    axis.plot([low, high], [low, high], linestyle="--", linewidth=1.3, color="#555555")
    ticks = np.log1p(np.array([0.5, 2, 5, 10, 20, 60]) * 1_000_000)
    axis.set_xticks(ticks)
    axis.set_yticks(ticks)
    axis.xaxis.set_major_formatter(FuncFormatter(salary_tick))
    axis.yaxis.set_major_formatter(FuncFormatter(salary_tick))
    axis.tick_params(axis="x", rotation=25)
    axis.set(title="Actual vs. predicted salary (positive actual salaries)", xlabel="Actual salary (log scale)", ylabel="Predicted salary (log scale)")

    axis = axes[0, 1]
    year_plot = yearly_comparison.copy()
    year_plot["mae_millions"] = year_plot["mae_dollars"] / 1_000_000
    sns.barplot(
        data=year_plot,
        x="validation_target_year",
        y="mae_millions",
        hue="display_model",
        palette=["#b14b4b", "#6f93bd"],
        ax=axis,
    )
    axis.set(title="MAE by validation year", xlabel="Salary year", ylabel="MAE ($ millions)")
    axis.legend(title="", frameon=False)

    axis = axes[1, 0]
    stage_plot = stage_summary.copy()
    stage_plot["label"] = stage_plot["current_contract_stage"].map(STAGE_LABELS)
    stage_plot["mae_millions"] = stage_plot["mae_dollars"] / 1_000_000
    bars = axis.bar(stage_plot["label"], stage_plot["mae_millions"], color="#7a5aa6")
    for bar, value, n in zip(bars, stage_plot["mae_millions"], stage_plot["n"]):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.06, f"${value:.2f}M\nn={int(n):,}", ha="center", va="bottom")
    axis.set(title="Final-model MAE by contract stage", xlabel="Current contract stage", ylabel="MAE ($ millions)")
    axis.set_ylim(0, stage_plot["mae_millions"].max() * 1.24)

    axis = axes[1, 1]
    band_plot = band_summary.copy()
    band_plot["mae_millions"] = band_plot["mae_dollars"] / 1_000_000
    bars = axis.bar(band_plot["salary_band"].astype(str), band_plot["mae_millions"], color="#4f8b6d")
    for bar, value, n in zip(bars, band_plot["mae_millions"], band_plot["n"]):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.06, f"${value:.2f}M\nn={int(n):,}", ha="center", va="bottom")
    axis.set(title="Final-model MAE by actual salary band", xlabel="Actual next-season salary", ylabel="MAE ($ millions)")
    axis.set_ylim(0, band_plot["mae_millions"].max() * 1.18)

    figure.suptitle("Final Gradient Boosting out-of-fold diagnostics")
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_importance_figure(importance: pd.DataFrame, path: Path) -> pd.DataFrame:
    selected = importance.loc[importance["model"].eq(FINAL_MODEL)]
    summary = (
        selected.groupby("feature")["importance"]
        .agg(mean_importance="mean", std_importance="std")
        .sort_values("mean_importance", ascending=False)
        .reset_index()
    )
    top = summary.head(15).sort_values("mean_importance", ascending=True)
    sns.set_theme(style="whitegrid", context="notebook")
    figure, axis = plt.subplots(figsize=(11.5, 7.0), constrained_layout=True)
    axis.barh(top["feature"], top["mean_importance"], xerr=top["std_importance"], color="#4c78a8", alpha=0.9)
    axis.set(title="Final Gradient Boosting feature importance across folds", xlabel="Mean impurity importance", ylabel="")
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return summary


def json_safe(value: object) -> object:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    result_dir = output_dir / "results"
    figure_dir = output_dir / "figures"
    config_dir = output_dir / "config"
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    comparison = pd.read_csv(args.comparison.resolve(), encoding="utf-8-sig")
    tree_predictions = pd.read_csv(args.tree_predictions.resolve(), encoding="utf-8-sig")
    baseline_predictions = pd.read_csv(args.baseline_predictions.resolve(), encoding="utf-8-sig")
    tree_folds = pd.read_csv(args.tree_fold_metrics.resolve(), encoding="utf-8-sig")
    baseline_folds = pd.read_csv(args.baseline_fold_metrics.resolve(), encoding="utf-8-sig")
    tuning = pd.read_csv(args.tree_tuning.resolve(), encoding="utf-8-sig")
    importance = pd.read_csv(args.feature_importance.resolve(), encoding="utf-8-sig")

    if comparison.iloc[0]["model"] != FINAL_MODEL:
        raise ValueError(f"Expected {FINAL_MODEL} to rank first by fold mean MAE.")
    final_predictions = tree_predictions.loc[tree_predictions["model"].eq(FINAL_MODEL)].copy()
    final_predictions["signed_error_dollars"] = (
        final_predictions["predicted_salary_dollars"] - final_predictions["target_salary_total"]
    )
    final_predictions = add_salary_band(final_predictions)

    yearly = error_summary(final_predictions, "target_salary_season")
    yearly = yearly.rename(columns={"target_salary_season": "validation_target_year"})
    stage = error_summary(final_predictions, "current_contract_stage", STAGE_ORDER)
    band = error_summary(final_predictions, "salary_band", SALARY_BAND_ORDER)

    final_year_metrics = tree_folds.loc[tree_folds["model"].eq(FINAL_MODEL), [
        "validation_target_year", "mae_dollars"
    ]].copy()
    final_year_metrics["display_model"] = "Gradient Boosting"
    naive_year_metrics = baseline_folds.loc[baseline_folds["model"].eq(NAIVE_MODEL), [
        "validation_target_year", "mae_dollars"
    ]].copy()
    naive_year_metrics["display_model"] = "Current salary baseline"
    yearly_comparison = pd.concat([final_year_metrics, naive_year_metrics], ignore_index=True)
    year_check = final_year_metrics.merge(naive_year_metrics, on="validation_target_year", suffixes=("_final", "_naive"))
    if not year_check["mae_dollars_final"].lt(year_check["mae_dollars_naive"]).all():
        raise ValueError("Final model does not outperform the naive baseline in every validation year.")

    save_comparison_figure(comparison, figure_dir / "10_all_model_mae_comparison.png")
    save_diagnostic_figure(
        final_predictions,
        yearly_comparison,
        stage,
        band,
        figure_dir / "11_final_model_diagnostics.png",
    )
    importance_summary = save_importance_figure(
        importance, figure_dir / "12_final_model_feature_importance.png"
    )

    final_tuning = tuning.loc[tuning["candidate_id"].eq(26)].iloc[0]
    selection = {
        "selected_model": FINAL_MODEL,
        "model_family": "gradient_boosting",
        "target_mode": "log_change_from_current",
        "selection_metric": "four_fold_mean_mae_dollars",
        "selection_value": float(comparison.iloc[0]["fold_mean_mae_dollars"]),
        "improvement_vs_current_salary_baseline_percent": float(
            comparison.iloc[0]["mae_improvement_vs_naive_percent"]
        ),
        "beats_current_salary_baseline_in_every_validation_year": True,
        "candidate_id": 26,
        "hyperparameters": {
            key: json_safe(final_tuning[key])
            for key in ["n_estimators", "learning_rate", "max_depth", "min_samples_leaf", "loss"]
        },
        "final_fit_feature_set": "tree_reduced_main",
        "final_fit_target_salary_years": [2020, 2026],
        "final_prediction_feature_year": 2026,
        "final_prediction_target_year": 2027,
    }
    (config_dir / "final_model_selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    comparison.to_csv(result_dir / "final_model_ranking.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(result_dir / "final_model_error_by_year.csv", index=False, encoding="utf-8-sig")
    stage.to_csv(result_dir / "final_model_error_by_contract_stage.csv", index=False, encoding="utf-8-sig")
    band.to_csv(result_dir / "final_model_error_by_salary_band.csv", index=False, encoding="utf-8-sig")
    yearly_comparison.to_csv(result_dir / "final_vs_naive_by_year.csv", index=False, encoding="utf-8-sig")
    importance_summary.to_csv(result_dir / "final_model_feature_importance.csv", index=False, encoding="utf-8-sig")
    final_predictions.nlargest(25, "absolute_error_dollars").to_csv(
        result_dir / "final_model_largest_errors.csv", index=False, encoding="utf-8-sig"
    )

    print("Final model selection complete")
    print(f"selected_model: {FINAL_MODEL}")
    print(f"fold_mean_mae: ${selection['selection_value']:,.0f}")
    print(f"improvement_vs_naive: {selection['improvement_vs_current_salary_baseline_percent']:.2f}%")
    print("MAE by contract stage:")
    for row in stage.itertuples(index=False):
        print(f"  {row.current_contract_stage}: ${row.mae_dollars:,.0f} (n={int(row.n)})")
    print("MAE by actual salary band:")
    for row in band.itertuples(index=False):
        print(f"  {row.salary_band}: ${row.mae_dollars:,.0f} (n={int(row.n)})")
    print(f"output_dir: {output_dir}")


if __name__ == "__main__":
    main()
