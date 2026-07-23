"""Build leakage-safe training and 2027 prediction datasets."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batting",
        type=Path,
        default=base / "interim" / "batting_standard_value_2018_2026_flagged.csv",
    )
    parser.add_argument(
        "--payroll-features",
        type=Path,
        default=base / "interim" / "payroll_player_features_2019_2026.csv",
    )
    parser.add_argument(
        "--salary",
        type=Path,
        default=base / "interim" / "payroll_player_season_salary_2019_2026.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base / "model_input",
    )
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        if not fields:
            raise ValueError(f"헤더가 없는 파일: {path}")
        if len(fields) != len(set(fields)):
            raise ValueError(f"중복 열 이름이 있는 파일: {path}")
        return fields, list(reader)


def index_unique(rows: list[dict[str, str]], label: str) -> dict[tuple[str, str], dict[str, str]]:
    output: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["Season"], row["playerid"])
        if key in output:
            raise ValueError(f"{label} 중복 Season+playerid: {key}")
        output[key] = row
    return output


def log_salary(value: str) -> str | float:
    if not value:
        return ""
    return round(math.log1p(int(value)), 10)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build(
    batting_path: Path,
    payroll_path: Path,
    salary_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    batting_fields, batting_rows = read_csv(batting_path)
    payroll_fields, payroll_rows = read_csv(payroll_path)
    _, salary_rows = read_csv(salary_path)
    payroll_by_key = index_unique(payroll_rows, "payroll features")
    salary_by_key = index_unique(salary_rows, "salary")
    payroll_names_by_playerid: dict[str, set[str]] = {}
    for payroll in payroll_rows:
        payroll_names_by_playerid.setdefault(payroll["playerid"], set()).add(payroll["Player"])
    canonical_payroll_name = {
        playerid: sorted(names, key=lambda name: (name.endswith("..."), -len(name), name))[0]
        for playerid, names in payroll_names_by_playerid.items()
    }

    payroll_copy_fields = [field for field in payroll_fields if field not in {"Season", "Player", "playerid"}]
    batting_copy_fields = [field for field in batting_fields if field not in {"Name"}]
    master_rows: list[dict[str, object]] = []

    for batting in batting_rows:
        feature_season = int(batting["Season"])
        if feature_season < 2019 or feature_season > 2026 or batting["is_hitter"] != "1":
            continue

        key = (batting["Season"], batting["playerid"])
        current = payroll_by_key.get(key)
        target_year = feature_season + 1
        target = salary_by_key.get((str(target_year), batting["playerid"])) if target_year <= 2026 else None
        has_current = current is not None
        current_complete = has_current and current["current_salary_is_complete"] == "1"
        has_target = target is not None
        target_complete = has_target and target["salary_is_complete"] == "1"
        training_period = 2019 <= feature_season <= 2025
        prediction_period = feature_season == 2026
        train_base = training_period and current_complete and target_complete
        predict_base = prediction_period and current_complete

        row: dict[str, object] = {
            "Season": batting["Season"],
            "target_salary_season": target_year,
            "Player": current["Player"] if current else canonical_payroll_name.get(batting["playerid"], batting["Name"]),
            "batting_name": batting["Name"],
        }
        row.update({field: batting[field] for field in batting_copy_fields if field != "Season"})
        for field in payroll_copy_fields:
            row[field] = current[field] if current else ""

        current_salary = current["current_salary_total"] if current else ""
        target_salary = target["season_salary_total"] if target else ""
        row.update(
            {
                "log1p_current_salary": log_salary(current_salary),
                "target_salary_total": target_salary,
                "log1p_target_salary": log_salary(target_salary),
                "target_salary_is_complete": target["salary_is_complete"] if target else "",
                "target_salary_status": target["salary_status"] if target else "",
                "has_current_payroll": int(has_current),
                "has_target_payroll_record": int(has_target),
                "is_training_period": int(training_period),
                "is_2027_prediction_period": int(prediction_period),
                "feature_is_covid_2020": int(feature_season == 2020),
                "target_is_covid_2020": int(target_year == 2020),
                "train_base_eligible": int(train_base),
                "predict_base_eligible": int(predict_base),
                "train_eligible_all_hitters_pa1": int(train_base and batting["eligible_all_hitters_pa1"] == "1"),
                "train_eligible_main_pa50": int(train_base and batting["eligible_main_pa50"] == "1"),
                "train_eligible_stable_pa100": int(train_base and batting["eligible_stable_pa100"] == "1"),
                "train_eligible_regular_pa200": int(train_base and batting["eligible_regular_pa200"] == "1"),
                "predict_eligible_all_hitters_pa1": int(predict_base and batting["eligible_all_hitters_pa1"] == "1"),
                "predict_eligible_main_pa50": int(predict_base and batting["eligible_main_pa50"] == "1"),
                "predict_eligible_stable_pa100": int(predict_base and batting["eligible_stable_pa100"] == "1"),
                "predict_eligible_regular_pa200": int(predict_base and batting["eligible_regular_pa200"] == "1"),
            }
        )
        master_rows.append(row)

    if not master_rows:
        raise ValueError("모델링 마스터 데이터가 비어 있습니다.")

    fields = list(master_rows[0])
    for row in master_rows:
        if list(row) != fields:
            raise ValueError("행별 열 순서가 일치하지 않습니다.")

    training_rows = [row for row in master_rows if row["train_base_eligible"] == 1]
    prediction_rows = [row for row in master_rows if row["predict_base_eligible"] == 1]
    write_csv(output_dir / "modeling_master_hitters_2019_2026.csv", fields, master_rows)
    write_csv(output_dir / "training_hitters_2019_2025.csv", fields, training_rows)
    write_csv(output_dir / "prediction_hitters_2026_for_2027.csv", fields, prediction_rows)

    return {
        "master_rows": len(master_rows),
        "training_rows": len(training_rows),
        "prediction_rows": len(prediction_rows),
        "master_missing_current_payroll": sum(row["has_current_payroll"] == 0 for row in master_rows),
        "training_missing_target_record": sum(
            row["is_training_period"] == 1 and row["has_target_payroll_record"] == 0 for row in master_rows
        ),
        "train_all_pa1": sum(row["train_eligible_all_hitters_pa1"] for row in master_rows),
        "train_main_pa50": sum(row["train_eligible_main_pa50"] for row in master_rows),
        "train_stable_pa100": sum(row["train_eligible_stable_pa100"] for row in master_rows),
        "train_regular_pa200": sum(row["train_eligible_regular_pa200"] for row in master_rows),
        "predict_all_pa1": sum(row["predict_eligible_all_hitters_pa1"] for row in master_rows),
        "predict_main_pa50": sum(row["predict_eligible_main_pa50"] for row in master_rows),
        "predict_stable_pa100": sum(row["predict_eligible_stable_pa100"] for row in master_rows),
        "predict_regular_pa200": sum(row["predict_eligible_regular_pa200"] for row in master_rows),
        "output_dir": str(output_dir.resolve()),
    }


def main() -> None:
    args = parse_args()
    result = build(
        args.batting.resolve(),
        args.payroll_features.resolve(),
        args.salary.resolve(),
        args.output_dir.resolve(),
    )
    print("Modeling datasets complete")
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
