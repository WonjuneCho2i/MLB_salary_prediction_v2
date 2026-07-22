"""Build one contractual salary target per player and season."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


RETAINED_GROUP = "Players No Longer On 40-Man Roster or No Longer In Organization"


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=base / "interim" / "payroll_2019_2026_raw_combined.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=base / "interim" / "payroll_player_season_salary_2019_2026.csv",
    )
    return parser.parse_args()


def is_integer(value: str | None) -> bool:
    return bool(value) and value.isdigit()


def joined(values: set[str]) -> str:
    return "|".join(sorted(value for value in values if value))


def choose_player_name(rows: list[dict[str, str]]) -> str:
    counts = Counter(row["Player"] for row in rows if row.get("Player"))
    if not counts:
        return ""
    highest = max(counts.values())
    return sorted(name for name, count in counts.items() if count == highest)[0]


def aggregate_group(season: str, playerid: str, rows: list[dict[str, str]]) -> dict[str, object]:
    # 동일 팀이 동일 금액을 두 계약 그룹에 중복 기재한 경우 하나의 지급액으로 봅니다.
    numeric_components: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    non_numeric_components: set[tuple[str, str]] = set()
    salary_statuses: set[str] = set()

    for row in rows:
        salary = row.get("Salary", "")
        team = row.get("Team", "")
        if is_integer(salary):
            numeric_components[(team, int(salary))].append(row)
        else:
            status = row.get("Salary_raw", "").strip()
            non_numeric_components.add((team, status))
            if status:
                salary_statuses.add(status)

    salary_total = sum(amount for _, amount in numeric_components)
    retained_total = sum(
        amount
        for (_, amount), component_rows in numeric_components.items()
        if any(row.get("contract_group") == RETAINED_GROUP for row in component_rows)
    )
    paying_teams = {team for team, _ in numeric_components}
    listed_teams = {row.get("Team", "") for row in rows}
    contract_groups = {row.get("contract_group", "") for row in rows}
    unique_payment_or_status_count = len(numeric_components) + len(non_numeric_components)
    duplicate_rows_removed = len(rows) - unique_payment_or_status_count
    salary_is_complete = bool(numeric_components) and not salary_statuses

    return {
        "Season": season,
        "Player": choose_player_name(rows),
        "playerid": playerid,
        "season_salary_total": salary_total if numeric_components else "",
        "salary_is_complete": int(salary_is_complete),
        "salary_status": joined(salary_statuses),
        "paying_teams": joined(paying_teams),
        "paying_team_count": len(paying_teams),
        "listed_teams": joined(listed_teams),
        "listed_team_count": len(listed_teams),
        "salary_component_count": len(numeric_components),
        "retained_salary_total": retained_total,
        "has_retained_salary": int(retained_total > 0),
        "contract_groups": joined(contract_groups),
        "raw_row_count": len(rows),
        "duplicate_salary_rows_removed": duplicate_rows_removed,
    }


def build(input_path: Path, output_path: Path) -> dict[str, int]:
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("통합 Payroll 파일에 데이터 행이 없습니다.")

    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for line_number, row in enumerate(rows, start=2):
        season = row.get("Season", "")
        playerid = row.get("playerid", "")
        if not season or not playerid:
            raise ValueError(f"Season 또는 playerid 누락: {input_path.name}:{line_number}")
        groups[(season, playerid)].append(row)

    output_rows = [
        aggregate_group(season, playerid, group_rows)
        for (season, playerid), group_rows in sorted(
            groups.items(), key=lambda item: (int(item[0][0]), item[1][0].get("Player", ""), item[0][1])
        )
    ]
    fields = list(output_rows[0])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)

    complete = sum(int(row["salary_is_complete"]) for row in output_rows)
    multi_team = sum(int(row["paying_team_count"]) > 1 for row in output_rows)
    retained = sum(int(row["has_retained_salary"]) for row in output_rows)
    duplicates = sum(int(row["duplicate_salary_rows_removed"]) for row in output_rows)
    return {
        "raw_rows": len(rows),
        "player_seasons": len(output_rows),
        "complete_salary": complete,
        "incomplete_salary": len(output_rows) - complete,
        "multi_team": multi_team,
        "retained_salary": retained,
        "duplicates_removed": duplicates,
    }


def main() -> None:
    args = parse_args()
    result = build(args.input.resolve(), args.output.resolve())
    print("Player-season salary table complete")
    for key, value in result.items():
        print(f"{key}: {value}")
    print(f"output: {args.output.resolve()}")


if __name__ == "__main__":
    main()
