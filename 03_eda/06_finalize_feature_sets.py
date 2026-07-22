"""EDA 3-6: finalize de-duplicated feature sets for the modeling stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BASELINE_NUMERIC = [
    "target_salary_season",
    "log1p_current_salary",
]

PERFORMANCE_ONLY_NUMERIC = [
    "target_salary_season",
    "AVG",
    "pa_per_game",
    "double_rate",
    "triple_rate",
    "home_run_rate",
    "walk_rate",
    "strikeout_rate",
    "hbp_rate",
    "sac_fly_rate",
    "gidp_rate",
    "stolen_base_rate",
    "caught_stealing_rate",
    "value_base_running_per_600_pa",
    "value_fielding_per_600_pa",
    "value_positional_per_600_pa",
]

LINEAR_REDUCED_NUMERIC = [
    *PERFORMANCE_ONLY_NUMERIC,
    "current_age",
    "current_age_squared",
    "current_service_total_days",
    "service_time_missing",
    "log1p_current_salary",
    "log1p_current_retained_salary",
    "feature_is_covid_2020",
    "target_is_covid_2020",
]

TREE_REDUCED_NUMERIC = [
    "target_salary_season",
    "AVG",
    "pa_per_game",
    "single_rate",
    "double_rate",
    "triple_rate",
    "home_run_rate",
    "run_rate",
    "rbi_rate",
    "walk_rate",
    "strikeout_rate",
    "hbp_rate",
    "sac_fly_rate",
    "sac_hit_rate",
    "gidp_rate",
    "stolen_base_rate",
    "caught_stealing_rate",
    "stolen_base_success_rate",
    "value_batting_per_600_pa",
    "value_base_running_per_600_pa",
    "value_fielding_per_600_pa",
    "value_positional_per_600_pa",
    "value_war_per_600_pa",
    "current_age",
    "current_service_total_days",
    "service_time_missing",
    "current_paying_team_count",
    "current_has_retained_salary",
    "log1p_current_salary",
    "log1p_current_retained_salary",
    "feature_is_covid_2020",
    "target_is_covid_2020",
]

CATEGORICAL = ["player_position", "current_contract_stage"]


REMOVAL_DECISIONS = [
    ("hit_rate", "drop_linear", "Overlaps with AVG and hit-type rates."),
    ("single_rate", "drop_linear", "Hit composition is represented by AVG plus extra-base-hit rates."),
    ("value_offense_per_600_pa", "drop", "Near-deterministic sum of batting and baserunning components."),
    ("value_batting_per_600_pa", "tree_only", "Strongly reconstructed by AVG and batting-event rates in the reduced linear set."),
    ("value_defense_per_600_pa", "drop", "Near-deterministic sum of fielding and positional components."),
    ("value_rar_per_600_pa", "drop", "Near-duplicate of WAR/600 and Dollars/600."),
    ("value_dollars_per_600_pa", "drop", "Economic scaling of WAR; would duplicate WAR information."),
    ("value_war_per_600_pa", "tree_only", "Retained as an overall value summary for nonlinear models."),
    ("current_is_*", "drop", "Duplicated by the categorical current_contract_stage variable."),
    ("current_listed_team_count", "drop", "Exactly matched other payroll-count fields in this sample."),
    ("current_salary_component_count", "drop", "Exactly matched other payroll-count fields in this sample."),
    ("current_has_multiple_contract_groups", "drop", "Already represented by current_contract_stage=multiple."),
    ("current_age_squared", "linear_only", "Intentional nonlinear age term; trees learn the curve directly."),
    ("raw cumulative batting/value totals", "sensitivity_only", "Not snapshot-safe for partial 2026 data."),
]


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root / "02_data_preprocessing" / "model_ready" / "training_model_ready.csv",
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=project_root / "02_data_preprocessing" / "model_ready" / "feature_manifest.json",
    )
    parser.add_argument(
        "--prediction-input",
        type=Path,
        default=project_root / "02_data_preprocessing" / "model_ready" / "prediction_2027_model_ready.csv",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=project_root / "02_data_preprocessing" / "model_ready" / "feature_manifest_final.json",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    return parser.parse_args()


def calculate_vif(sample: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    numeric = sample[features].apply(pd.to_numeric, errors="coerce")
    missing_counts = numeric.isna().sum()
    numeric = numeric.fillna(numeric.median(numeric_only=True))
    variances = numeric.var(ddof=0)
    usable = variances.loc[variances.gt(0)].index.tolist()
    matrix = numeric[usable].to_numpy(dtype=float)
    matrix = (matrix - matrix.mean(axis=0)) / matrix.std(axis=0)
    rows: list[dict[str, object]] = []

    for index, feature in enumerate(usable):
        target = matrix[:, index]
        predictors = np.delete(matrix, index, axis=1)
        coefficients, *_ = np.linalg.lstsq(predictors, target, rcond=None)
        residual = target - predictors @ coefficients
        r_squared = 1 - float(np.square(residual).sum() / np.square(target).sum())
        r_squared = max(0.0, min(1.0, r_squared))
        vif = np.inf if 1 - r_squared < 1e-10 else 1 / (1 - r_squared)
        rows.append(
            {
                "feature": feature,
                "missing_count": int(missing_counts[feature]),
                "vif": vif,
                "vif_flag": "infinite" if np.isinf(vif) else ("high_10_plus" if vif >= 10 else ("moderate_5_plus" if vif >= 5 else "acceptable")),
            }
        )
    result = pd.DataFrame(rows)
    result["sort_vif"] = result["vif"].replace(np.inf, 1e12)
    return result.sort_values("sort_vif", ascending=False).drop(columns="sort_vif").reset_index(drop=True)


def validate_feature_sets(columns: set[str], feature_sets: dict[str, dict[str, list[str]]]) -> None:
    for set_name, definition in feature_sets.items():
        all_features = definition["numeric"] + definition["categorical"]
        missing = sorted(set(all_features) - columns)
        duplicates = sorted({feature for feature in all_features if all_features.count(feature) > 1})
        if missing:
            raise ValueError(f"{set_name} contains missing columns: {missing}")
        if duplicates:
            raise ValueError(f"{set_name} contains duplicate columns: {duplicates}")


def write_selection_notes(path: Path, feature_sets: dict[str, dict[str, list[str]]], vif: pd.DataFrame) -> None:
    high_vif = vif.loc[vif["vif_flag"].isin(["infinite", "high_10_plus"])]
    lines = [
        "# 3-6 final feature selection",
        "",
        "The source model-ready CSV is preserved. Modeling columns are selected through feature_manifest_final.json.",
        "",
        "## Final comparison sets",
        "",
    ]
    for name, definition in feature_sets.items():
        lines.append(
            f"- {name}: {len(definition['numeric'])} numeric + {len(definition['categorical'])} categorical"
        )
    lines.extend(
        [
            "",
            "## Linear reduced-set diagnostic",
            "",
            f"- VIF >= 10 or infinite after reduction: {len(high_vif)}",
            "- Age and age squared are intentionally retained together for a quadratic age effect.",
            "- Regularized linear models remain the primary linear specification.",
            "- Tree models use a broader but still de-duplicated snapshot-safe set.",
            "- Raw cumulative totals remain sensitivity-only because 2026 is a partial-season snapshot.",
            "",
            "## Leakage safeguards",
            "",
            "- No next-season contract group, next-season age, future AAV, or contract text is included.",
            "- Numeric imputation and categorical encoding must be fitted inside each training fold.",
            "- 2026 features are used only for the final 2027 prediction after model selection.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    sample = pd.read_csv(args.input.resolve(), encoding="utf-8-sig")
    sample = sample.loc[sample["train_eligible_main_pa50"].eq(1)].copy()
    prediction = pd.read_csv(args.prediction_input.resolve(), encoding="utf-8-sig", nrows=5)
    source_manifest = json.loads(args.source_manifest.resolve().read_text(encoding="utf-8"))

    final_sets = {
        "baseline_previous_salary": {"numeric": BASELINE_NUMERIC, "categorical": []},
        "performance_only_reduced": {"numeric": PERFORMANCE_ONLY_NUMERIC, "categorical": ["player_position"]},
        "linear_reduced_main": {"numeric": LINEAR_REDUCED_NUMERIC, "categorical": CATEGORICAL},
        "tree_reduced_main": {"numeric": TREE_REDUCED_NUMERIC, "categorical": CATEGORICAL},
        "full_main_rate_safe_sensitivity": source_manifest["feature_sets"]["full_main_rate_safe"],
        "full_expanded_sensitivity": source_manifest["feature_sets"]["full_expanded_sensitivity"],
    }
    validate_feature_sets(set(sample.columns), final_sets)
    validate_feature_sets(set(prediction.columns), final_sets)

    final_manifest = {
        **{key: value for key, value in source_manifest.items() if key != "feature_sets"},
        "feature_sets": final_sets,
        "selection_policy": {
            "primary_linear_set": "linear_reduced_main",
            "primary_tree_set": "tree_reduced_main",
            "baseline_set": "baseline_previous_salary",
            "ablation_set": "performance_only_reduced",
            "selection_basis": "3-5 Spearman pair correlations, VIF diagnostics, snapshot safety, and leakage prevention",
            "raw_source_columns_preserved": True,
        },
    }
    args.output_manifest.resolve().write_text(
        json.dumps(final_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    output_dir = args.output_dir.resolve()
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for set_name, definition in final_sets.items():
        summary_rows.append(
            {
                "feature_set": set_name,
                "numeric_count": len(definition["numeric"]),
                "categorical_count": len(definition["categorical"]),
                "total_pre_encoding": len(definition["numeric"]) + len(definition["categorical"]),
                "intended_use": (
                    "primary_linear" if set_name == "linear_reduced_main" else
                    "primary_tree" if set_name == "tree_reduced_main" else
                    "baseline" if set_name == "baseline_previous_salary" else
                    "ablation" if set_name == "performance_only_reduced" else "sensitivity"
                ),
            }
        )
    pd.DataFrame(summary_rows).to_csv(
        table_dir / "final_feature_set_summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(REMOVAL_DECISIONS, columns=["feature_or_group", "decision", "reason"]).to_csv(
        table_dir / "feature_selection_decisions.csv", index=False, encoding="utf-8-sig"
    )

    vif = calculate_vif(sample, LINEAR_REDUCED_NUMERIC)
    vif.to_csv(table_dir / "linear_reduced_vif.csv", index=False, encoding="utf-8-sig")
    write_selection_notes(output_dir / "06_final_feature_selection_notes.md", final_sets, vif)

    print("Final feature sets complete")
    print(f"main_sample_rows: {len(sample)}")
    for row in summary_rows:
        print(f"  {row['feature_set']}: {row['total_pre_encoding']} pre-encoding features")
    print(f"linear_vif_10_plus_or_infinite: {vif['vif_flag'].isin(['infinite', 'high_10_plus']).sum()}")
    print(f"manifest: {args.output_manifest.resolve()}")


if __name__ == "__main__":
    main()
