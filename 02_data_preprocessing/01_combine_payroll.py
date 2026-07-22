"""Combine the 240 raw FanGraphs Payroll CSV files without cleaning values."""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path


SEASONS = tuple(range(2019, 2027))
TEAMS = (
    "ARI", "ATH", "ATL", "BAL", "BOS", "CHC", "CHW", "CIN", "CLE", "COL",
    "DET", "HOU", "KCR", "LAA", "LAD", "MIA", "MIL", "MIN", "NYM", "NYY",
    "PHI", "PIT", "SDP", "SEA", "SFG", "STL", "TBR", "TEX", "TOR", "WSN",
)
FILE_PATTERN = re.compile(r"^fangraphs_payroll_([A-Z]+)_(20\d{2})\.csv$")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=project_root / "01_data_collection" / "downloads" / "payroll",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "interim" / "payroll_2019_2026_raw_combined.csv",
    )
    return parser.parse_args()


def expected_keys() -> set[tuple[str, str]]:
    return {(team, str(season)) for season in SEASONS for team in TEAMS}


def discover_files(input_dir: Path) -> dict[tuple[str, str], Path]:
    found: dict[tuple[str, str], Path] = {}
    unexpected: list[str] = []

    for path in sorted(input_dir.glob("*.csv")):
        match = FILE_PATTERN.fullmatch(path.name)
        if not match:
            unexpected.append(path.name)
            continue
        team, season = match.groups()
        key = (team, season)
        if key in found:
            raise ValueError(f"중복 팀·시즌 파일: {path.name}")
        found[key] = path

    missing = sorted(expected_keys() - set(found))
    extra = sorted(set(found) - expected_keys())
    if unexpected or missing or extra:
        messages = []
        if unexpected:
            messages.append(f"예상하지 않은 CSV: {unexpected}")
        if missing:
            messages.append(f"누락 팀·시즌: {missing}")
        if extra:
            messages.append(f"범위 밖 팀·시즌: {extra}")
        raise ValueError("\n".join(messages))

    return found


def combine(input_dir: Path, output: Path) -> dict[str, object]:
    files = discover_files(input_dir)
    reference_fields: list[str] | None = None
    combined_rows: list[dict[str, str]] = []
    season_counts: Counter[str] = Counter()

    for (team, season), path in sorted(files.items(), key=lambda item: (item[0][1], item[0][0])):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            if not fields:
                raise ValueError(f"헤더가 없는 파일: {path.name}")
            if len(fields) != len(set(fields)):
                raise ValueError(f"중복 열 이름이 있는 파일: {path.name}")
            if reference_fields is None:
                reference_fields = fields
            elif fields != reference_fields:
                raise ValueError(f"열 구조가 다른 파일: {path.name}")

            file_rows = 0
            for line_number, row in enumerate(reader, start=2):
                if row.get("Team") != team or row.get("Season") != season:
                    raise ValueError(
                        f"파일명과 행의 팀·시즌 불일치: {path.name}:{line_number}"
                    )
                if not row.get("playerid"):
                    raise ValueError(f"playerid 누락: {path.name}:{line_number}")
                row["_source_file"] = path.name
                combined_rows.append(row)
                season_counts[season] += 1
                file_rows += 1

            if file_rows == 0:
                raise ValueError(f"데이터 행이 없는 파일: {path.name}")

    assert reference_fields is not None
    output.parent.mkdir(parents=True, exist_ok=True)
    output_fields = [*reference_fields, "_source_file"]
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(combined_rows)

    return {
        "file_count": len(files),
        "row_count": len(combined_rows),
        "column_count": len(output_fields),
        "season_counts": dict(sorted(season_counts.items())),
        "output": str(output.resolve()),
    }


def main() -> None:
    args = parse_args()
    result = combine(args.input_dir.resolve(), args.output.resolve())
    print("Payroll 원본 통합 완료")
    print(f"파일 수: {result['file_count']}")
    print(f"행 수: {result['row_count']}")
    print(f"열 수: {result['column_count']}")
    for season, count in result["season_counts"].items():
        print(f"  {season}: {count}행")
    print(f"출력: {result['output']}")


if __name__ == "__main__":
    main()
