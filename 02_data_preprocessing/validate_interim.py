"""Validate preprocessing outputs before building the modeling dataset."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path


BASE = Path(__file__).resolve().parent
RAW_DIR = BASE.parent / "01_data_collection" / "downloads" / "payroll"
COMBINED = BASE / "interim" / "payroll_2019_2026_raw_combined.csv"
SALARY = BASE / "interim" / "payroll_player_season_salary_2019_2026.csv"
BATTING = BASE / "interim" / "batting_standard_value_2018_2026_flagged.csv"
PAYROLL_FEATURES = BASE / "interim" / "payroll_player_features_2019_2026.csv"


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        if not fields:
            raise AssertionError(f"헤더 없음: {path}")
        if len(fields) != len(set(fields)):
            raise AssertionError(f"중복 헤더: {path}")
        return fields, list(reader)


def unique_by(rows: list[dict[str, str]], fields: tuple[str, ...], label: str) -> dict[tuple[str, ...], dict[str, str]]:
    output: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        if key in output:
            raise AssertionError(f"{label} 중복 키: {key}")
        output[key] = row
    return output


def main() -> None:
    raw_files = sorted(RAW_DIR.glob("fangraphs_payroll_*.csv"))
    if len(raw_files) != 240:
        raise AssertionError(f"Payroll 원본 파일 수: {len(raw_files)} != 240")

    _, combined = read_csv(COMBINED)
    _, salary = read_csv(SALARY)
    batting_fields, batting = read_csv(BATTING)
    payroll_fields, payroll = read_csv(PAYROLL_FEATURES)

    raw_row_counts: Counter[str] = Counter()
    for path in raw_files:
        _, rows = read_csv(path)
        raw_row_counts[path.name] = len(rows)
    combined_row_counts = Counter(row["_source_file"] for row in combined)
    if raw_row_counts != combined_row_counts:
        raise AssertionError("원본 파일별 행 수와 통합 파일의 _source_file 행 수가 다름")

    salary_by_key = unique_by(salary, ("Season", "playerid"), "salary")
    batting_by_key = unique_by(batting, ("Season", "playerid"), "batting")
    payroll_by_key = unique_by(payroll, ("Season", "playerid"), "payroll features")
    if set(salary_by_key) != set(payroll_by_key):
        raise AssertionError("salary와 payroll features의 선수·시즌 키가 다름")

    raw_groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in combined:
        raw_groups[(row["Season"], row["playerid"])].append(row)
    if set(raw_groups) != set(salary_by_key):
        raise AssertionError("combined와 salary의 선수·시즌 키가 다름")

    duplicate_components_removed = 0
    for key, rows in raw_groups.items():
        numeric_components = {
            (row["Team"], int(row["Salary"]))
            for row in rows
            if row["Salary"].isdigit()
        }
        non_numeric_components = {
            (row["Team"], row["Salary_raw"])
            for row in rows
            if not row["Salary"].isdigit()
        }
        expected_total = sum(amount for _, amount in numeric_components)
        actual_raw = salary_by_key[key]["season_salary_total"]
        actual_total = int(actual_raw) if actual_raw else 0
        if expected_total != actual_total:
            raise AssertionError(f"총연봉 합계 불일치: {key} {expected_total} != {actual_total}")
        duplicate_components_removed += len(rows) - len(numeric_components) - len(non_numeric_components)
    if duplicate_components_removed != 1:
        raise AssertionError(f"예상 중복 제거 건수 1과 다름: {duplicate_components_removed}")

    if "Contract" in payroll_fields or any("AAV" in field.upper() for field in payroll_fields):
        raise AssertionError("Payroll feature에 Contract 또는 AAV가 포함됨")

    flag_errors = 0
    for row in batting:
        partition = int(row["is_pitcher"]) + int(row["is_shohei_ohtani"]) + int(row["is_hitter"])
        nested = (
            int(row["eligible_regular_pa200"]) <= int(row["eligible_stable_pa100"])
            <= int(row["eligible_main_pa50"])
            <= int(row["eligible_all_hitters_pa1"])
            <= int(row["is_hitter"])
        )
        if partition != 1 or not nested:
            flag_errors += 1
    if flag_errors:
        raise AssertionError(f"타자/PA 플래그 논리 오류: {flag_errors}")

    shared_keys = set(batting_by_key) & set(payroll_by_key)
    name_mismatches = [
        (key, batting_by_key[key]["Name"], payroll_by_key[key]["Player"])
        for key in sorted(shared_keys)
        if batting_by_key[key]["Name"] != payroll_by_key[key]["Player"]
    ]
    overlaps = Counter(key[0] for key in shared_keys)

    print("Interim data audit PASSED")
    print(f"raw_files: {len(raw_files)}")
    print(f"combined_rows: {len(combined)}")
    print(f"salary_player_seasons: {len(salary_by_key)}")
    print(f"batting_player_seasons: {len(batting_by_key)}")
    print(f"payroll_feature_player_seasons: {len(payroll_by_key)}")
    print(f"duplicate_salary_components_removed: {duplicate_components_removed}")
    print(f"incomplete_salary_rows: {sum(row['salary_is_complete'] == '0' for row in salary)}")
    print(f"age_missing: {sum(row['age_missing'] == '1' for row in payroll)}")
    print(f"service_time_missing: {sum(row['service_time_missing'] == '1' for row in payroll)}")
    print(f"batting_payroll_shared_keys: {len(shared_keys)}")
    print(f"name_mismatches_on_shared_playerid: {len(name_mismatches)}")
    print("shared_keys_by_season:")
    for season in sorted(overlaps):
        print(f"  {season}: {overlaps[season]}")
    if name_mismatches:
        print("name_mismatch_examples (joins remain playerid-based):")
        for item in name_mismatches[:10]:
            print(f"  {item}")


if __name__ == "__main__":
    main()
