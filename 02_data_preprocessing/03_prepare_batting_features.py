"""Join FanGraphs Standard and Value data and add hitter/model-sample flags."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


OHTANI_PLAYERID = "19755"
VALUE_COLUMNS = {
    "Batting": "value_batting",
    "Base Running": "value_base_running",
    "Fielding": "value_fielding",
    "Positional": "value_positional",
    "Offense": "value_offense",
    "Defense": "value_defense",
    "League": "value_league",
    "Replacement": "value_replacement",
    "RAR": "value_rar",
    "WAR": "value_war",
    "Dollars": "value_dollars",
}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--standard",
        type=Path,
        default=project_root / "01_data_collection" / "downloads" / "standard" / "fangraphs_standard_2018_2026.csv",
    )
    parser.add_argument(
        "--value",
        type=Path,
        default=project_root / "01_data_collection" / "downloads" / "value" / "fangraphs_value_2018_2026.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "interim" / "batting_standard_value_2018_2026_flagged.csv",
    )
    return parser.parse_args()


def read_unique_header_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        if not fields:
            raise ValueError(f"헤더가 없는 파일: {path}")
        if len(fields) != len(set(fields)):
            raise ValueError(f"중복 열 이름이 있는 파일: {path}")
        return fields, list(reader)


def read_value_csv(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        headers = next(reader)
        season_index = headers.index("Season")
        playerid_index = headers.index("playerid")
        name_index = headers.index("Name")
        team_index = headers.index("Team")
        position_index = headers.index("player_position")
        source_index = headers.index("_source_url")
        collected_index = headers.index("_collected_at")
        metric_indices = {output: headers.index(source) for source, output in VALUE_COLUMNS.items()}

        output: dict[tuple[str, str], dict[str, str]] = {}
        for line_number, row in enumerate(reader, start=2):
            if len(row) != len(headers):
                raise ValueError(f"열 개수 불일치: {path.name}:{line_number}")
            key = (row[season_index], row[playerid_index])
            if key in output:
                raise ValueError(f"중복 Season+playerid: {path.name}:{line_number} {key}")
            values = {name: row[index] for name, index in metric_indices.items()}
            values.update(
                {
                    "value_name": row[name_index],
                    "value_team": row[team_index],
                    "value_position": row[position_index],
                    "value_source_url": row[source_index],
                    "value_collected_at": row[collected_index],
                }
            )
            output[key] = values
        return output


def pa_band(is_hitter: bool, is_ohtani: bool, pa: int) -> str:
    if is_ohtani:
        return "excluded_ohtani"
    if not is_hitter:
        return "pitcher"
    if pa == 0:
        return "pa_0"
    if pa < 50:
        return "pa_1_49"
    if pa < 100:
        return "pa_50_99"
    if pa < 200:
        return "pa_100_199"
    return "pa_200_plus"


def build(standard_path: Path, value_path: Path, output_path: Path) -> dict[str, int]:
    standard_fields, standard_rows = read_unique_header_csv(standard_path)
    value_by_key = read_value_csv(value_path)

    standard_by_key: dict[tuple[str, str], dict[str, str]] = {}
    for line_number, row in enumerate(standard_rows, start=2):
        key = (row["Season"], row["playerid"])
        if key in standard_by_key:
            raise ValueError(f"Standard 중복 Season+playerid: {standard_path.name}:{line_number} {key}")
        standard_by_key[key] = row

    missing_value = sorted(set(standard_by_key) - set(value_by_key))
    missing_standard = sorted(set(value_by_key) - set(standard_by_key))
    if missing_value or missing_standard:
        raise ValueError(
            f"Standard/Value 키 불일치: Value 누락={len(missing_value)}, Standard 누락={len(missing_standard)}"
        )

    ignored_standard = {"#", "_source_url", "_collected_at"}
    kept_standard_fields = [field for field in standard_fields if field not in ignored_standard]
    output_rows: list[dict[str, object]] = []

    for key in sorted(standard_by_key, key=lambda item: (int(item[0]), standard_by_key[item]["Name"], item[1])):
        standard = standard_by_key[key]
        value = value_by_key[key]
        if standard["Name"] != value["value_name"]:
            raise ValueError(f"선수명 불일치: {key} {standard['Name']} / {value['value_name']}")
        if standard["Team"] != value["value_team"]:
            raise ValueError(f"팀 불일치: {key} {standard['Team']} / {value['value_team']}")
        if standard["player_position"] != value["value_position"]:
            raise ValueError(f"포지션 불일치: {key}")

        try:
            pa = int(standard["PA"])
        except ValueError as error:
            raise ValueError(f"PA가 정수가 아님: {key} {standard['PA']}") from error

        is_pitcher = standard["player_position"] == "P"
        is_ohtani = standard["playerid"] == OHTANI_PLAYERID
        is_hitter = not is_pitcher and not is_ohtani
        row: dict[str, object] = {field: standard[field] for field in kept_standard_fields}
        row.update({column: value[column] for column in VALUE_COLUMNS.values()})
        row.update(
            {
                "is_pitcher": int(is_pitcher),
                "is_shohei_ohtani": int(is_ohtani),
                "is_hitter": int(is_hitter),
                "pa_band": pa_band(is_hitter, is_ohtani, pa),
                "eligible_all_hitters_pa1": int(is_hitter and pa >= 1),
                "eligible_main_pa50": int(is_hitter and pa >= 50),
                "eligible_stable_pa100": int(is_hitter and pa >= 100),
                "eligible_regular_pa200": int(is_hitter and pa >= 200),
                "is_partial_2026_snapshot": int(standard["Season"] == "2026"),
                "standard_source_url": standard["_source_url"],
                "standard_collected_at": standard["_collected_at"],
                "value_source_url": value["value_source_url"],
                "value_collected_at": value["value_collected_at"],
            }
        )
        output_rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(output_rows[0])
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)

    return {
        "rows": len(output_rows),
        "pitcher_rows": sum(int(row["is_pitcher"]) for row in output_rows),
        "ohtani_rows": sum(int(row["is_shohei_ohtani"]) for row in output_rows),
        "hitter_rows": sum(int(row["is_hitter"]) for row in output_rows),
        "all_hitters_pa1": sum(int(row["eligible_all_hitters_pa1"]) for row in output_rows),
        "main_pa50": sum(int(row["eligible_main_pa50"]) for row in output_rows),
        "stable_pa100": sum(int(row["eligible_stable_pa100"]) for row in output_rows),
        "regular_pa200": sum(int(row["eligible_regular_pa200"]) for row in output_rows),
    }


def main() -> None:
    args = parse_args()
    result = build(args.standard.resolve(), args.value.resolve(), args.output.resolve())
    print("Batting Standard/Value table complete")
    for key, value in result.items():
        print(f"{key}: {value}")
    print(f"output: {args.output.resolve()}")


if __name__ == "__main__":
    main()
