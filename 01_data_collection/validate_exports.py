"""Validate manually exported FanGraphs Standard and Value CSV files.

This script does not modify the raw files.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


EXPECTED_START_SEASON = 2018
EXPECTED_END_SEASON = 2026
STANDARD_REQUIRED = {"Season", "Name", "Team", "PA", "HR", "playerid"}
VALUE_REQUIRED = {"Season", "Name", "Team", "WAR", "Dollars", "playerid"}


@dataclass
class CsvSummary:
    path: Path
    row_count: int
    headers: list[str]
    seasons: set[int]
    players: set[str]
    keys: list[tuple[str, int]]
    missing_player_ids: int

    @property
    def duplicate_keys(self) -> int:
        return len(self.keys) - len(set(self.keys))


def read_csv(path: Path) -> CsvSummary:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(8192)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(handle, dialect=dialect)
        headers = [str(value).strip() for value in (reader.fieldnames or [])]
        seasons: set[int] = set()
        players: set[str] = set()
        keys: list[tuple[str, int]] = []
        missing_player_ids = 0
        row_count = 0

        for raw_row in reader:
            row_count += 1
            player_id = str(raw_row.get("playerid", "") or "").strip()
            season_text = str(raw_row.get("Season", "") or "").strip()
            if not player_id:
                missing_player_ids += 1
            else:
                players.add(player_id)
            try:
                season = int(float(season_text))
            except ValueError:
                continue
            seasons.add(season)
            if player_id:
                keys.append((player_id, season))

    return CsvSummary(
        path=path,
        row_count=row_count,
        headers=headers,
        seasons=seasons,
        players=players,
        keys=keys,
        missing_player_ids=missing_player_ids,
    )


def validate_report(summary: CsvSummary, report: str, required: set[str]) -> list[str]:
    errors: list[str] = []
    missing = sorted(required - set(summary.headers))
    if missing:
        errors.append(f"{report}: missing required columns: {missing}")
    if summary.row_count < 500:
        errors.append(f"{report}: only {summary.row_count:,} rows; expected at least 500")
    if not summary.seasons:
        errors.append(f"{report}: no valid Season values found")
    elif (
        min(summary.seasons) != EXPECTED_START_SEASON
        or max(summary.seasons) != EXPECTED_END_SEASON
    ):
        errors.append(
            f"{report}: season coverage is {min(summary.seasons)}-{max(summary.seasons)}, "
            f"expected {EXPECTED_START_SEASON}-{EXPECTED_END_SEASON}"
        )
    if summary.duplicate_keys:
        errors.append(
            f"{report}: {summary.duplicate_keys:,} duplicate playerid+Season rows; check Split Teams"
        )
    if summary.missing_player_ids:
        errors.append(f"{report}: {summary.missing_player_ids:,} rows have no playerid")
    return errors


def print_summary(summary: CsvSummary, report: str) -> None:
    season_range = (
        f"{min(summary.seasons)}-{max(summary.seasons)}" if summary.seasons else "not detected"
    )
    print(f"\n[{report}]")
    print(f"file: {summary.path.resolve()}")
    print(f"rows: {summary.row_count:,}")
    print(f"seasons: {season_range}")
    print(f"unique players: {len(summary.players):,}")
    print(f"duplicate player-seasons: {summary.duplicate_keys:,}")
    print(f"missing playerid rows: {summary.missing_player_ids:,}")
    print(f"columns ({len(summary.headers)}): {', '.join(summary.headers)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate FanGraphs Standard and Value CSV exports.")
    parser.add_argument("--standard", required=True, type=Path)
    parser.add_argument("--value", required=True, type=Path)
    args = parser.parse_args()

    standard = read_csv(args.standard)
    value = read_csv(args.value)
    print_summary(standard, "Standard")
    print_summary(value, "Value")

    errors = [
        *validate_report(standard, "Standard", STANDARD_REQUIRED),
        *validate_report(value, "Value", VALUE_REQUIRED),
    ]
    common_keys = set(standard.keys) & set(value.keys)
    print(f"\ncommon player-seasons: {len(common_keys):,}")
    if len(common_keys) < 500:
        errors.append(f"Standard/Value: only {len(common_keys):,} common player-seasons")

    if errors:
        print("\nVALIDATION FAILED")
        for error in errors:
            print(f"- {error}")
        return 1

    print("\nVALIDATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
