"""Create leakage-safe model-ready tables and an explicit feature manifest."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


ID_COLUMNS = ["Season", "target_salary_season", "Player", "playerid"]
CATEGORICAL_COLUMNS = ["Team", "player_position", "current_contract_stage"]
RAW_BATTING_COLUMNS = [
    "G", "AB", "PA", "H", "1B", "2B", "3B", "HR", "R", "RBI", "BB", "IBB",
    "SO", "HBP", "SF", "SH", "GDP", "SB", "CS", "AVG",
]
RAW_VALUE_COLUMNS = [
    "value_batting", "value_base_running", "value_fielding", "value_positional",
    "value_offense", "value_defense", "value_league", "value_replacement",
    "value_rar", "value_war", "value_dollars",
]
PLAYER_COLUMNS = [
    "current_age", "current_age_squared", "age_missing", "age_30_plus", "age_35_plus",
    "current_service_years", "current_service_days", "current_service_total_days",
    "service_time_missing", "service_3_years_plus", "service_6_years_plus",
    "current_is_guaranteed", "current_is_arbitration", "current_is_pre_arb",
    "current_is_no_longer_40man", "current_has_multiple_contract_groups",
    "current_paying_team_count", "current_listed_team_count",
    "current_salary_component_count", "current_has_retained_salary",
    "log1p_current_salary", "feature_is_covid_2020", "target_is_covid_2020",
]
RATE_COLUMNS = [
    "pa_per_game", "hit_rate", "single_rate", "double_rate", "triple_rate",
    "home_run_rate", "run_rate", "rbi_rate", "walk_rate", "intentional_walk_rate",
    "strikeout_rate", "hbp_rate", "sac_fly_rate", "sac_hit_rate", "gidp_rate",
    "stolen_base_rate", "caught_stealing_rate", "stolen_base_success_rate",
]
VALUE_RATE_COLUMNS = [
    "value_batting_per_600_pa", "value_base_running_per_600_pa",
    "value_fielding_per_600_pa", "value_positional_per_600_pa",
    "value_offense_per_600_pa", "value_defense_per_600_pa",
    "value_rar_per_600_pa", "value_war_per_600_pa", "value_dollars_per_600_pa",
]
ENGINEERED_PAYROLL_COLUMNS = ["log1p_current_retained_salary"]
SAMPLE_FLAG_COLUMNS = [
    "train_eligible_all_hitters_pa1", "train_eligible_main_pa50",
    "train_eligible_stable_pa100", "train_eligible_regular_pa200",
    "predict_eligible_all_hitters_pa1", "predict_eligible_main_pa50",
    "predict_eligible_stable_pa100", "predict_eligible_regular_pa200",
]
TARGET_COLUMNS = ["target_salary_total", "log1p_target_salary"]


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training",
        type=Path,
        default=base / "model_input" / "training_hitters_2019_2025.csv",
    )
    parser.add_argument(
        "--prediction",
        type=Path,
        default=base / "model_input" / "prediction_hitters_2026_for_2027.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=base / "model_ready")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], field: str) -> float:
    value = row.get(field, "")
    if value in {"", None}:
        return 0.0
    return float(str(value).replace("$", "").replace(",", ""))


def safe_ratio(numerator: float, denominator: float, multiplier: float = 1.0) -> str | float:
    if denominator <= 0:
        return ""
    return round(numerator / denominator * multiplier, 10)


def engineer(row: dict[str, str]) -> dict[str, object]:
    pa = number(row, "PA")
    games = number(row, "G")
    sb = number(row, "SB")
    cs = number(row, "CS")
    output: dict[str, object] = {
        "pa_per_game": safe_ratio(pa, games),
        "hit_rate": safe_ratio(number(row, "H"), pa),
        "single_rate": safe_ratio(number(row, "1B"), pa),
        "double_rate": safe_ratio(number(row, "2B"), pa),
        "triple_rate": safe_ratio(number(row, "3B"), pa),
        "home_run_rate": safe_ratio(number(row, "HR"), pa),
        "run_rate": safe_ratio(number(row, "R"), pa),
        "rbi_rate": safe_ratio(number(row, "RBI"), pa),
        "walk_rate": safe_ratio(number(row, "BB"), pa),
        "intentional_walk_rate": safe_ratio(number(row, "IBB"), pa),
        "strikeout_rate": safe_ratio(number(row, "SO"), pa),
        "hbp_rate": safe_ratio(number(row, "HBP"), pa),
        "sac_fly_rate": safe_ratio(number(row, "SF"), pa),
        "sac_hit_rate": safe_ratio(number(row, "SH"), pa),
        "gidp_rate": safe_ratio(number(row, "GDP"), pa),
        "stolen_base_rate": safe_ratio(sb, pa),
        "caught_stealing_rate": safe_ratio(cs, pa),
        "stolen_base_success_rate": safe_ratio(sb, sb + cs),
        "value_batting_per_600_pa": safe_ratio(number(row, "value_batting"), pa, 600),
        "value_base_running_per_600_pa": safe_ratio(number(row, "value_base_running"), pa, 600),
        "value_fielding_per_600_pa": safe_ratio(number(row, "value_fielding"), pa, 600),
        "value_positional_per_600_pa": safe_ratio(number(row, "value_positional"), pa, 600),
        "value_offense_per_600_pa": safe_ratio(number(row, "value_offense"), pa, 600),
        "value_defense_per_600_pa": safe_ratio(number(row, "value_defense"), pa, 600),
        "value_rar_per_600_pa": safe_ratio(number(row, "value_rar"), pa, 600),
        "value_war_per_600_pa": safe_ratio(number(row, "value_war"), pa, 600),
        "value_dollars_per_600_pa": safe_ratio(number(row, "value_dollars"), pa, 600),
        "log1p_current_retained_salary": round(math.log1p(number(row, "current_retained_salary_total")), 10),
    }
    return output


def select_row(row: dict[str, str]) -> dict[str, object]:
    output: dict[str, object] = {}
    for field in ID_COLUMNS + CATEGORICAL_COLUMNS + RAW_BATTING_COLUMNS + RAW_VALUE_COLUMNS + PLAYER_COLUMNS:
        output[field] = number(row, field) if field == "value_dollars" else row.get(field, "")
    output.update(engineer(row))
    for field in SAMPLE_FLAG_COLUMNS + TARGET_COLUMNS:
        output[field] = row.get(field, "")
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def feature_manifest() -> dict[str, object]:
    batting_rate_numeric = ["target_salary_season", "AVG", *RATE_COLUMNS, *VALUE_RATE_COLUMNS]
    full_main_numeric = [
        *batting_rate_numeric,
        "current_age", "current_age_squared", "current_service_total_days",
        "service_time_missing", "current_is_guaranteed", "current_is_arbitration",
        "current_is_pre_arb", "current_is_no_longer_40man",
        "current_has_multiple_contract_groups", "current_paying_team_count",
        "current_listed_team_count", "current_salary_component_count",
        "current_has_retained_salary", "log1p_current_salary",
        "log1p_current_retained_salary", "feature_is_covid_2020", "target_is_covid_2020",
    ]
    expanded_numeric = list(dict.fromkeys([
        "target_salary_season", *RAW_BATTING_COLUMNS, *RAW_VALUE_COLUMNS,
        *RATE_COLUMNS, *VALUE_RATE_COLUMNS, *PLAYER_COLUMNS, *ENGINEERED_PAYROLL_COLUMNS,
    ]))
    return {
        "target": "log1p_target_salary",
        "target_raw_dollars": "target_salary_total",
        "id_columns": ID_COLUMNS,
        "sample_flags": SAMPLE_FLAG_COLUMNS,
        "feature_sets": {
            "baseline_previous_salary": {
                "numeric": ["target_salary_season", "log1p_current_salary"],
                "categorical": [],
            },
            "batting_value_rate": {
                "numeric": batting_rate_numeric,
                "categorical": ["player_position"],
            },
            "full_main_rate_safe": {
                "numeric": full_main_numeric,
                "categorical": ["player_position", "current_contract_stage"],
            },
            "full_expanded_sensitivity": {
                "numeric": expanded_numeric,
                "categorical": CATEGORICAL_COLUMNS,
            },
        },
        "preprocessing": {
            "numeric_missing": "median fitted inside each training fold",
            "categorical_missing": "most_frequent fitted inside each training fold",
            "categorical_encoding": "one-hot with unknown categories ignored",
            "linear_model_scaling": "standardize numeric features inside each training fold",
            "salary_target_transform": "log1p; convert predictions back with expm1",
        },
        "notes": [
            "full_main_rate_safe avoids raw cumulative batting/value totals because 2026 is a partial-season snapshot",
            "full_expanded_sensitivity retains raw cumulative features for comparison only",
            "identifiers, source metadata, eligibility flags, and future payroll attributes are not model inputs",
        ],
    }


def validate(rows: list[dict[str, object]], manifest: dict[str, object], is_training: bool) -> None:
    fields = set(rows[0])
    forbidden = {"Contract", "AAV", "target_contract_groups", "target_age", "target_service_time"}
    if fields & forbidden:
        raise ValueError(f"누수 위험 열 포함: {sorted(fields & forbidden)}")
    for spec in manifest["feature_sets"].values():
        missing = (set(spec["numeric"]) | set(spec["categorical"])) - fields
        if missing:
            raise ValueError(f"Feature manifest 열 누락: {sorted(missing)}")
    if is_training and any(not row["log1p_target_salary"] for row in rows):
        raise ValueError("학습 데이터 목표 연봉 결측")
    if not is_training and any(row["log1p_target_salary"] for row in rows):
        raise ValueError("예측 데이터에 목표 연봉이 포함됨")


def main() -> None:
    args = parse_args()
    training = [select_row(row) for row in read_csv(args.training.resolve())]
    prediction = [select_row(row) for row in read_csv(args.prediction.resolve())]
    manifest = feature_manifest()
    validate(training, manifest, is_training=True)
    validate(prediction, manifest, is_training=False)

    output_dir = args.output_dir.resolve()
    write_csv(output_dir / "training_model_ready.csv", training)
    write_csv(output_dir / "prediction_2027_model_ready.csv", prediction)
    with (output_dir / "feature_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    print("Model feature preparation complete")
    print(f"training_rows: {len(training)}")
    print(f"prediction_rows: {len(prediction)}")
    print(f"columns: {len(training[0])}")
    print(f"output_dir: {output_dir}")


if __name__ == "__main__":
    main()
