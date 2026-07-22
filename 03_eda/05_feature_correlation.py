"""EDA 3-5: feature correlations and multicollinearity diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


DISPLAY_NAMES = {
    "log1p_target_salary": "Next salary",
    "log1p_current_salary": "Current salary",
    "current_age": "Age",
    "current_service_total_days": "Service time",
    "pa_per_game": "PA per game",
    "AVG": "AVG",
    "home_run_rate": "HR rate",
    "walk_rate": "BB rate",
    "strikeout_rate": "SO rate",
    "stolen_base_success_rate": "SB success",
    "value_batting_per_600_pa": "Batting/600",
    "value_base_running_per_600_pa": "Baserunning/600",
    "value_fielding_per_600_pa": "Fielding/600",
    "value_war_per_600_pa": "WAR/600",
    "value_dollars_per_600_pa": "Dollars/600",
}

SELECTED_FEATURES = list(DISPLAY_NAMES)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root / "02_data_preprocessing" / "model_ready" / "training_model_ready.csv",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=project_root / "02_data_preprocessing" / "model_ready" / "feature_manifest.json",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    return parser.parse_args()


def load_data(data_path: Path, manifest_path: Path) -> tuple[pd.DataFrame, list[str]]:
    data = pd.read_csv(data_path, encoding="utf-8-sig")
    sample = data.loc[data["train_eligible_main_pa50"].eq(1)].copy()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    numeric_features = manifest["feature_sets"]["full_main_rate_safe"]["numeric"]
    missing = sorted(set(numeric_features + ["log1p_target_salary"]) - set(sample.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    sample[numeric_features + ["log1p_target_salary"]] = sample[
        numeric_features + ["log1p_target_salary"]
    ].apply(pd.to_numeric, errors="coerce")
    return sample, numeric_features


def save_selected_heatmap(sample: pd.DataFrame, path: Path) -> pd.DataFrame:
    correlations = sample[SELECTED_FEATURES].corr(method="spearman")
    renamed = correlations.rename(index=DISPLAY_NAMES, columns=DISPLAY_NAMES)
    mask = np.triu(np.ones_like(renamed, dtype=bool), k=1)

    sns.set_theme(style="white", context="notebook")
    figure, axis = plt.subplots(figsize=(12.5, 10.0), constrained_layout=True)
    sns.heatmap(
        renamed,
        mask=mask,
        cmap="vlag",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8},
        cbar_kws={"label": "Spearman correlation", "shrink": 0.8},
        ax=axis,
    )
    axis.set_title("Selected feature correlations in the main PA >= 50 sample")
    axis.set_xlabel("")
    axis.set_ylabel("")
    axis.tick_params(axis="x", rotation=55)
    axis.tick_params(axis="y", rotation=0)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return correlations


def pairwise_correlations(sample: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    correlations = sample[features].corr(method="spearman")
    rows: list[dict[str, object]] = []
    for left_index, left in enumerate(features):
        for right in features[left_index + 1 :]:
            value = correlations.loc[left, right]
            if pd.isna(value):
                continue
            rows.append(
                {
                    "feature_1": left,
                    "feature_2": right,
                    "spearman_correlation": value,
                    "absolute_correlation": abs(value),
                }
            )
    return pd.DataFrame(rows).sort_values("absolute_correlation", ascending=False).reset_index(drop=True)


def save_top_pairs_figure(pairs: pd.DataFrame, path: Path) -> None:
    top = pairs.head(15).sort_values("absolute_correlation", ascending=True).copy()
    top["pair"] = top["feature_1"] + "  |  " + top["feature_2"]
    colors = np.where(top["spearman_correlation"].ge(0), "#4c78a8", "#d17b49")

    sns.set_theme(style="whitegrid", context="notebook")
    figure, axis = plt.subplots(figsize=(12.5, 7.3), constrained_layout=True)
    bars = axis.barh(top["pair"], top["absolute_correlation"], color=colors)
    for bar, signed_value in zip(bars, top["spearman_correlation"]):
        axis.text(
            min(bar.get_width() + 0.012, 0.985),
            bar.get_y() + bar.get_height() / 2,
            f"r={signed_value:.2f}",
            va="center",
            ha="left" if bar.get_width() < 0.94 else "right",
        )
    axis.set_xlim(0, 1.03)
    axis.set_xlabel("Absolute Spearman correlation")
    axis.set_ylabel("")
    axis.set_title("Most strongly correlated numeric feature pairs in the main model")
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def calculate_vif(sample: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    numeric = sample[features].copy()
    missing_counts = numeric.isna().sum()
    numeric = numeric.fillna(numeric.median(numeric_only=True))
    variances = numeric.var(ddof=0)
    usable = variances.loc[variances.gt(0)].index.tolist()
    constant = variances.loc[variances.le(0)].index.tolist()

    matrix = numeric[usable].to_numpy(dtype=float)
    means = matrix.mean(axis=0)
    standard_deviations = matrix.std(axis=0)
    matrix = (matrix - means) / standard_deviations
    rows: list[dict[str, object]] = []

    for index, feature in enumerate(usable):
        target = matrix[:, index]
        predictors = np.delete(matrix, index, axis=1)
        coefficients, *_ = np.linalg.lstsq(predictors, target, rcond=None)
        residual = target - predictors @ coefficients
        residual_sum_squares = float(np.square(residual).sum())
        total_sum_squares = float(np.square(target).sum())
        r_squared = max(0.0, min(1.0, 1 - residual_sum_squares / total_sum_squares))
        vif = np.inf if 1 - r_squared < 1e-10 else 1 / (1 - r_squared)
        rows.append(
            {
                "feature": feature,
                "missing_count": int(missing_counts[feature]),
                "r_squared_from_other_features": r_squared,
                "vif": vif,
                "vif_flag": "infinite" if np.isinf(vif) else ("high_10_plus" if vif >= 10 else ("moderate_5_plus" if vif >= 5 else "acceptable")),
            }
        )

    rows.extend(
        {
            "feature": feature,
            "missing_count": int(missing_counts[feature]),
            "r_squared_from_other_features": np.nan,
            "vif": np.nan,
            "vif_flag": "constant",
        }
        for feature in constant
    )
    result = pd.DataFrame(rows)
    result["sort_vif"] = result["vif"].replace(np.inf, 1e12).fillna(-1)
    return result.sort_values("sort_vif", ascending=False).drop(columns="sort_vif").reset_index(drop=True)


def write_notes(path: Path, pairs: pd.DataFrame, vif: pd.DataFrame) -> None:
    high_pairs = pairs.loc[pairs["absolute_correlation"].ge(0.85)]
    high_vif = vif.loc[vif["vif_flag"].isin(["infinite", "high_10_plus"])]
    lines = [
        "# 3-5 correlation and multicollinearity notes",
        "",
        f"- Numeric features checked: {len(vif)}",
        f"- Feature pairs with |Spearman r| >= 0.85: {len(high_pairs)}",
        f"- Features with VIF >= 10 or infinite: {len(high_vif)}",
        "- VIF is diagnostic only; missing numeric values were median-filled for this calculation.",
        "- The actual model must fit all imputers inside each training fold.",
        "",
        "## Modeling decisions to test",
        "",
        "- Keep current_contract_stage as the primary contract representation and avoid duplicating all stage flags in linear models.",
        "- Do not use WAR/600 and Dollars/600 together when they encode essentially the same value scale.",
        "- Avoid combining component rates with their exact or near-exact totals without regularization.",
        "- Age and age squared are intentionally related; retain both only for models that need an explicit nonlinear age term.",
        "- Ridge/Elastic Net and tree models can be compared with a reduced-feature linear model.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    figure_dir = output_dir / "figures"
    table_dir = output_dir / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    sample, numeric_features = load_data(args.input.resolve(), args.manifest.resolve())
    selected = save_selected_heatmap(sample, figure_dir / "08_selected_feature_correlation.png")
    selected.to_csv(table_dir / "selected_feature_spearman.csv", encoding="utf-8-sig")

    pairs = pairwise_correlations(sample, numeric_features)
    pairs.to_csv(table_dir / "full_main_numeric_correlation_pairs.csv", index=False, encoding="utf-8-sig")
    pairs.loc[pairs["absolute_correlation"].ge(0.85)].to_csv(
        table_dir / "high_correlation_pairs.csv", index=False, encoding="utf-8-sig"
    )
    save_top_pairs_figure(pairs, figure_dir / "09_top_collinear_feature_pairs.png")

    vif = calculate_vif(sample, numeric_features)
    vif.to_csv(table_dir / "numeric_feature_vif.csv", index=False, encoding="utf-8-sig")
    write_notes(output_dir / "05_feature_correlation_notes.md", pairs, vif)

    print("Feature-correlation EDA complete")
    print(f"main_sample_rows: {len(sample)}")
    print(f"numeric_features: {len(numeric_features)}")
    print(f"high_correlation_pairs_abs_r_ge_0.85: {(pairs['absolute_correlation'] >= 0.85).sum()}")
    print(f"vif_10_plus_or_infinite: {vif['vif_flag'].isin(['infinite', 'high_10_plus']).sum()}")
    print(f"output_dir: {output_dir}")


if __name__ == "__main__":
    main()
