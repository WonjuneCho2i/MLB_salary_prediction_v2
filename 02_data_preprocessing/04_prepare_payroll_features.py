"""Create leakage-safe current-season player features from FanGraphs Payroll."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


GROUP_GUARANTEED = "Players With Guaranteed Salaries"
GROUP_ARBITRATION = "Players Eligible For Arbitration"
GROUP_PRE_ARB = "Notable Players Not Yet Eligible For Arbitration"
GROUP_RETAINED = "Players No Longer On 40-Man Roster or No Longer In Organization"
EXPECTED_GROUPS = {GROUP_GUARANTEED, GROUP_ARBITRATION, GROUP_PRE_ARB, GROUP_RETAINED}
SERVICE_DAYS_PER_YEAR = 172


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-payroll",
        type=Path,
        default=base / "interim" / "payroll_2019_2026_raw_combined.csv",
    )
    parser.add_argument(
        "--salary",
        type=Path,
        default=base / "interim" / "payroll_player_season_salary_2019_2026.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=base / "interim" / "payroll_player_features_2019_2026.csv",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"헤더가 없는 파일: {path}")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"중복 열 이름이 있는 파일: {path}")
        return list(reader)


def one_value(rows: list[dict[str, str]], field: str, key: tuple[str, str]) -> str:
    values = {row.get(field, "").strip() for row in rows if row.get(field, "").strip()}
    if len(values) > 1:
        raise ValueError(f"동일 선수·시즌의 {field} 값 충돌: {key} {sorted(values)}")
    return next(iter(values), "")


def parse_age(raw: str) -> float | None:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def parse_service_time(raw: str, key: tuple[str, str]) -> tuple[int, int, int] | None:
    if not raw:
        return None
    parts = raw.split(".")
    if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
        raise ValueError(f"Service Time 형식 오류: {key} {raw}")
    years, days = int(parts[0]), int(parts[1])
    if days >= SERVICE_DAYS_PER_YEAR:
        raise ValueError(f"Service Time 일수 범위 오류: {key} {raw}")
    return years, days, years * SERVICE_DAYS_PER_YEAR + days


def contract_stage(groups: set[str]) -> str:
    if len(groups) != 1:
        return "multiple"
    group = next(iter(groups))
    return {
        GROUP_GUARANTEED: "guaranteed",
        GROUP_ARBITRATION: "arbitration",
        GROUP_PRE_ARB: "pre_arb",
        GROUP_RETAINED: "no_longer_40man",
    }[group]


def build(raw_path: Path, salary_path: Path, output_path: Path) -> dict[str, int]:
    raw_rows = read_csv(raw_path)
    salary_rows = read_csv(salary_path)

    raw_groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in raw_rows:
        raw_groups[(row["Season"], row["playerid"])].append(row)

    salary_by_key: dict[tuple[str, str], dict[str, str]] = {}
    for row in salary_rows:
        key = (row["Season"], row["playerid"])
        if key in salary_by_key:
            raise ValueError(f"연봉 테이블 중복 키: {key}")
        salary_by_key[key] = row

    if set(raw_groups) != set(salary_by_key):
        raise ValueError(
            f"원본/연봉 선수·시즌 키 불일치: raw={len(raw_groups)}, salary={len(salary_by_key)}"
        )

    output_rows: list[dict[str, object]] = []
    age_missing_count = 0
    service_missing_count = 0

    for key in sorted(raw_groups, key=lambda item: (int(item[0]), salary_by_key[item]["Player"], item[1])):
        rows = raw_groups[key]
        salary = salary_by_key[key]
        age_raw = one_value(rows, "Age", key)
        service_raw = one_value(rows, "Service_Time", key)
        age = parse_age(age_raw)
        service = parse_service_time(service_raw, key)
        groups = {row["contract_group"] for row in rows if row.get("contract_group")}
        unknown_groups = groups - EXPECTED_GROUPS
        if unknown_groups:
            raise ValueError(f"알 수 없는 계약 그룹: {key} {sorted(unknown_groups)}")
        if "|".join(sorted(groups)) != salary["contract_groups"]:
            raise ValueError(f"연봉 테이블과 계약 그룹 불일치: {key}")

        if age is None:
            age_missing_count += 1
        if service is None:
            service_missing_count += 1
        service_years, service_days, service_total_days = service or ("", "", "")

        output_rows.append(
            {
                "Season": key[0],
                "Player": salary["Player"],
                "playerid": key[1],
                "current_age_raw": age_raw,
                "current_age": "" if age is None else age,
                "current_age_squared": "" if age is None else round(age * age, 6),
                "age_missing": int(age is None),
                "age_30_plus": "" if age is None else int(age >= 30),
                "age_35_plus": "" if age is None else int(age >= 35),
                "current_service_time_raw": service_raw,
                "current_service_years": service_years,
                "current_service_days": service_days,
                "current_service_total_days": service_total_days,
                "service_time_missing": int(service is None),
                "service_3_years_plus": "" if service is None else int(service_total_days >= 3 * SERVICE_DAYS_PER_YEAR),
                "service_6_years_plus": "" if service is None else int(service_total_days >= 6 * SERVICE_DAYS_PER_YEAR),
                "current_contract_stage": contract_stage(groups),
                "current_contract_groups": salary["contract_groups"],
                "current_is_guaranteed": int(GROUP_GUARANTEED in groups),
                "current_is_arbitration": int(GROUP_ARBITRATION in groups),
                "current_is_pre_arb": int(GROUP_PRE_ARB in groups),
                "current_is_no_longer_40man": int(GROUP_RETAINED in groups),
                "current_has_multiple_contract_groups": int(len(groups) > 1),
                "current_salary_total": salary["season_salary_total"],
                "current_salary_is_complete": salary["salary_is_complete"],
                "current_salary_status": salary["salary_status"],
                "current_paying_teams": salary["paying_teams"],
                "current_paying_team_count": salary["paying_team_count"],
                "current_listed_teams": salary["listed_teams"],
                "current_listed_team_count": salary["listed_team_count"],
                "current_salary_component_count": salary["salary_component_count"],
                "current_retained_salary_total": salary["retained_salary_total"],
                "current_has_retained_salary": salary["has_retained_salary"],
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(output_rows[0])
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)

    return {
        "rows": len(output_rows),
        "age_missing": age_missing_count,
        "service_time_missing": service_missing_count,
        "guaranteed": sum(int(row["current_is_guaranteed"]) for row in output_rows),
        "arbitration": sum(int(row["current_is_arbitration"]) for row in output_rows),
        "pre_arb": sum(int(row["current_is_pre_arb"]) for row in output_rows),
        "no_longer_40man": sum(int(row["current_is_no_longer_40man"]) for row in output_rows),
        "multiple_contract_groups": sum(int(row["current_has_multiple_contract_groups"]) for row in output_rows),
    }


def main() -> None:
    args = parse_args()
    result = build(args.raw_payroll.resolve(), args.salary.resolve(), args.output.resolve())
    print("Payroll player feature table complete")
    for key, value in result.items():
        print(f"{key}: {value}")
    print(f"output: {args.output.resolve()}")


if __name__ == "__main__":
    main()
