#!/usr/bin/env python3
"""Run the MLB salary preprocessing stages 1-6 as one reproducible pipeline.

Default behavior follows the original repository's 2019-2026 preprocessing while
preserving genuinely missing numeric values in Stage 6. Use
``--numeric-missing-policy legacy-zero`` for byte-for-byte legacy reproduction.

1. Combine 240 FanGraphs payroll CSV files.
2. Build one contractual salary row per player-season.
3. Join FanGraphs Standard and Value tables and create hitter/PA flags.
4. Build leakage-safe current-season payroll/player features.
5. Align current-season features with next-season salary targets.
6. Create model-ready tables and feature_manifest.json.

Only the Python standard library is required (Python 3.10+ recommended).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


# ---------------------------------------------------------------------------
# Project defaults
# ---------------------------------------------------------------------------

DEFAULT_PAYROLL_START_SEASON = 2019
DEFAULT_BATTING_START_SEASON = 2018
DEFAULT_SNAPSHOT_SEASON = 2026
DEFAULT_TEAMS = (
    "ARI", "ATH", "ATL", "BAL", "BOS", "CHC", "CHW", "CIN", "CLE", "COL",
    "DET", "HOU", "KCR", "LAA", "LAD", "MIA", "MIL", "MIN", "NYM", "NYY",
    "PHI", "PIT", "SDP", "SEA", "SFG", "STL", "TBR", "TEX", "TOR", "WSN",
)
PAYROLL_FILE_PATTERN = re.compile(r"^fangraphs_payroll_([A-Z]+)_(20\d{2})\.csv$")
OHTANI_PLAYERID = "19755"
SERVICE_DAYS_PER_YEAR = 172

GROUP_GUARANTEED = "Players With Guaranteed Salaries"
GROUP_ARBITRATION = "Players Eligible For Arbitration"
GROUP_PRE_ARB = "Notable Players Not Yet Eligible For Arbitration"
GROUP_RETAINED = "Players No Longer On 40-Man Roster or No Longer In Organization"
EXPECTED_CONTRACT_GROUPS = {
    GROUP_GUARANTEED,
    GROUP_ARBITRATION,
    GROUP_PRE_ARB,
    GROUP_RETAINED,
}

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

RAW_PAYROLL_REQUIRED = {
    "Season", "Team", "Player", "playerid", "contract_group", "Age",
    "Service_Time", "Salary", "Salary_raw",
}
STANDARD_REQUIRED = {
    "Season", "Name", "Team", "G", "AB", "PA", "H", "1B", "2B", "3B",
    "HR", "R", "RBI", "BB", "IBB", "SO", "HBP", "SF", "SH", "GDP",
    "SB", "CS", "AVG", "playerid", "player_position", "_source_url",
    "_collected_at",
}
VALUE_REQUIRED = {
    "Season", "Name", "Team", "playerid", "player_position", "_source_url",
    "_collected_at", *VALUE_COLUMNS.keys(),
}

# Stage 6 feature definitions are retained from the original scripts so that
# default output columns and feature_manifest.json stay compatible.
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


class PipelineError(RuntimeError):
    """Raised for a data-contract or preprocessing failure."""


@dataclass(frozen=True)
class PipelineConfig:
    project_root: Path
    preprocessing_dir: Path
    payroll_dir: Path
    standard_csv: Path
    value_csv: Path
    payroll_start_season: int = DEFAULT_PAYROLL_START_SEASON
    batting_start_season: int = DEFAULT_BATTING_START_SEASON
    snapshot_season: int = DEFAULT_SNAPSHOT_SEASON
    teams: tuple[str, ...] = DEFAULT_TEAMS
    include_ohtani: bool = False
    numeric_missing_policy: str = "preserve"

    @property
    def interim_dir(self) -> Path:
        return self.preprocessing_dir / "interim"

    @property
    def model_input_dir(self) -> Path:
        return self.preprocessing_dir / "model_input"

    @property
    def model_ready_dir(self) -> Path:
        return self.preprocessing_dir / "model_ready"

    @property
    def payroll_seasons(self) -> tuple[int, ...]:
        return tuple(range(self.payroll_start_season, self.snapshot_season + 1))

    @property
    def combined_payroll_csv(self) -> Path:
        return self.interim_dir / (
            f"payroll_{self.payroll_start_season}_{self.snapshot_season}_raw_combined.csv"
        )

    @property
    def salary_csv(self) -> Path:
        return self.interim_dir / (
            f"payroll_player_season_salary_{self.payroll_start_season}_{self.snapshot_season}.csv"
        )

    @property
    def batting_features_csv(self) -> Path:
        return self.interim_dir / (
            f"batting_standard_value_{self.batting_start_season}_{self.snapshot_season}_flagged.csv"
        )

    @property
    def payroll_features_csv(self) -> Path:
        return self.interim_dir / (
            f"payroll_player_features_{self.payroll_start_season}_{self.snapshot_season}.csv"
        )

    @property
    def modeling_master_csv(self) -> Path:
        return self.model_input_dir / (
            f"modeling_master_hitters_{self.payroll_start_season}_{self.snapshot_season}.csv"
        )

    @property
    def training_csv(self) -> Path:
        return self.model_input_dir / (
            f"training_hitters_{self.payroll_start_season}_{self.snapshot_season - 1}.csv"
        )

    @property
    def prediction_csv(self) -> Path:
        return self.model_input_dir / (
            f"prediction_hitters_{self.snapshot_season}_for_{self.snapshot_season + 1}.csv"
        )

    @property
    def training_model_ready_csv(self) -> Path:
        return self.model_ready_dir / "training_model_ready.csv"

    @property
    def prediction_model_ready_csv(self) -> Path:
        return self.model_ready_dir / (
            f"prediction_{self.snapshot_season + 1}_model_ready.csv"
        )

    @property
    def feature_manifest_json(self) -> Path:
        return self.model_ready_dir / "feature_manifest.json"


# ---------------------------------------------------------------------------
# Generic I/O and validation helpers
# ---------------------------------------------------------------------------


def ensure_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise PipelineError(f"{label} 파일을 찾을 수 없습니다: {path}")


def ensure_required_columns(
    headers: Sequence[str], required: Iterable[str], path: Path, *, exact_once: bool = True
) -> None:
    counts = Counter(headers)
    missing = sorted(set(required) - set(headers))
    duplicated_required = sorted(name for name in required if counts[name] > 1)
    if missing:
        raise PipelineError(f"필수 열 누락: {path} -> {missing}")
    if exact_once and duplicated_required:
        raise PipelineError(f"필수 열이 중복됨: {path} -> {duplicated_required}")


def read_unique_header_csv(
    path: Path,
    *,
    required: Iterable[str] = (),
    allow_empty: bool = False,
) -> tuple[list[str], list[dict[str, str]]]:
    """Read a CSV whose header names must all be unique and row widths must match."""
    ensure_file(path, "CSV")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            headers = next(reader)
        except StopIteration as exc:
            raise PipelineError(f"빈 CSV 파일: {path}") from exc
        if not headers:
            raise PipelineError(f"헤더가 없는 CSV: {path}")
        if len(headers) != len(set(headers)):
            duplicates = sorted(name for name, count in Counter(headers).items() if count > 1)
            raise PipelineError(f"중복 열 이름: {path} -> {duplicates}")
        ensure_required_columns(headers, required, path)

        rows: list[dict[str, str]] = []
        for line_number, values in enumerate(reader, start=2):
            if len(values) != len(headers):
                raise PipelineError(
                    f"열 개수 불일치: {path.name}:{line_number} "
                    f"({len(values)} != {len(headers)})"
                )
            rows.append(dict(zip(headers, values)))

    if not rows and not allow_empty:
        raise PipelineError(f"데이터 행이 없는 CSV: {path}")
    return headers, rows


def atomic_write_csv(
    path: Path,
    fields: Sequence[str],
    rows: Sequence[dict[str, Any]],
    *,
    allow_empty: bool = False,
) -> None:
    if not fields:
        raise PipelineError(f"출력 열 목록이 비어 있습니다: {path}")
    if not rows and not allow_empty:
        raise PipelineError(f"출력 행이 비어 있습니다: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8-sig",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fields),
            lineterminator="\n",
            extrasaction="raise",
        )
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    temp_path.replace(path)


def unique_index(
    rows: Sequence[dict[str, str]],
    key_fields: Sequence[str],
    label: str,
) -> dict[tuple[str, ...], dict[str, str]]:
    output: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        if key in output:
            raise PipelineError(f"{label} 중복 키 {tuple(key_fields)}: {key}")
        output[key] = row
    return output


def parse_int_strict(value: str, *, field: str, context: str) -> int:
    token = value.strip()
    if not token or not re.fullmatch(r"[+-]?\d+", token):
        raise PipelineError(f"정수 형식 오류: {context} {field}={value!r}")
    return int(token)


def parse_optional_number(value: str | None, *, field: str, context: str) -> float | None:
    if value is None:
        return None
    token = str(value).strip()
    if token == "":
        return None
    token = token.replace("$", "").replace(",", "")
    try:
        result = float(token)
    except ValueError as exc:
        raise PipelineError(f"숫자 형식 오류: {context} {field}={value!r}") from exc
    if not math.isfinite(result):
        raise PipelineError(f"유한하지 않은 숫자: {context} {field}={value!r}")
    return result


def salary_log1p(value: str, *, context: str) -> str | float:
    if value == "":
        return ""
    amount = parse_int_strict(value, field="salary", context=context)
    if amount < 0:
        raise PipelineError(f"음수 연봉: {context} {amount}")
    return round(math.log1p(amount), 10)


def print_stage(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def print_summary(summary: dict[str, Any]) -> None:
    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")
        else:
            print(f"{key}: {value}")


# ---------------------------------------------------------------------------
# Stage 1: combine raw payroll files
# ---------------------------------------------------------------------------


def expected_payroll_keys(config: PipelineConfig) -> set[tuple[str, str]]:
    return {
        (team, str(season))
        for season in config.payroll_seasons
        for team in config.teams
    }


def discover_payroll_files(config: PipelineConfig) -> dict[tuple[str, str], Path]:
    if not config.payroll_dir.is_dir():
        raise PipelineError(f"Payroll 입력 폴더를 찾을 수 없습니다: {config.payroll_dir}")

    found: dict[tuple[str, str], Path] = {}
    unexpected: list[str] = []
    for path in sorted(config.payroll_dir.glob("*.csv")):
        match = PAYROLL_FILE_PATTERN.fullmatch(path.name)
        if not match:
            unexpected.append(path.name)
            continue
        team, season = match.groups()
        key = (team, season)
        if key in found:
            raise PipelineError(f"중복 팀·시즌 Payroll 파일: {path.name}")
        found[key] = path

    expected = expected_payroll_keys(config)
    missing = sorted(expected - set(found), key=lambda item: (item[1], item[0]))
    extra = sorted(set(found) - expected, key=lambda item: (item[1], item[0]))
    if unexpected or missing or extra:
        messages: list[str] = []
        if unexpected:
            messages.append(f"예상하지 않은 CSV: {unexpected}")
        if missing:
            messages.append(f"누락 팀·시즌: {missing}")
        if extra:
            messages.append(f"설정 범위 밖 팀·시즌: {extra}")
        raise PipelineError("\n".join(messages))
    return found


def stage1_combine_payroll(config: PipelineConfig) -> dict[str, Any]:
    files = discover_payroll_files(config)
    reference_fields: list[str] | None = None
    combined_rows: list[dict[str, str]] = []
    season_counts: Counter[str] = Counter()

    for (team, season), path in sorted(files.items(), key=lambda item: (item[0][1], item[0][0])):
        fields, rows = read_unique_header_csv(path, required=RAW_PAYROLL_REQUIRED)
        if "_source_file" in fields:
            raise PipelineError(f"원본 Payroll에 예약 열 _source_file이 이미 존재함: {path}")
        if reference_fields is None:
            reference_fields = fields
        elif fields != reference_fields:
            raise PipelineError(f"열 구조 또는 열 순서가 다른 Payroll 파일: {path.name}")

        for line_offset, row in enumerate(rows, start=2):
            if row["Team"] != team or row["Season"] != season:
                raise PipelineError(
                    f"파일명과 행의 팀·시즌 불일치: {path.name}:{line_offset} "
                    f"row=({row['Team']}, {row['Season']})"
                )
            if not row["playerid"].strip():
                raise PipelineError(f"playerid 누락: {path.name}:{line_offset}")
            output_row = dict(row)
            output_row["_source_file"] = path.name
            combined_rows.append(output_row)
            season_counts[season] += 1

    if reference_fields is None:
        raise PipelineError("Payroll 원본 CSV가 없습니다.")
    output_fields = [*reference_fields, "_source_file"]
    atomic_write_csv(config.combined_payroll_csv, output_fields, combined_rows)
    return {
        "file_count": len(files),
        "row_count": len(combined_rows),
        "column_count": len(output_fields),
        "season_counts": dict(sorted(season_counts.items())),
        "output": str(config.combined_payroll_csv.resolve()),
    }


# ---------------------------------------------------------------------------
# Stage 2: aggregate player-season contractual salary
# ---------------------------------------------------------------------------


def is_normalized_integer(value: str | None) -> bool:
    return bool(value) and str(value).isdigit()


def joined(values: Iterable[str]) -> str:
    return "|".join(sorted(value for value in set(values) if value))


def choose_player_name(rows: Sequence[dict[str, str]]) -> str:
    counts = Counter(row["Player"] for row in rows if row.get("Player"))
    if not counts:
        return ""
    highest = max(counts.values())
    return sorted(name for name, count in counts.items() if count == highest)[0]


def aggregate_salary_group(
    season: str,
    playerid: str,
    rows: Sequence[dict[str, str]],
) -> dict[str, Any]:
    # Preserve the original repository's rule: identical team+amount entries
    # appearing in multiple contract groups are treated as one payment.
    numeric_components: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    non_numeric_components: set[tuple[str, str]] = set()
    salary_statuses: set[str] = set()

    for row in rows:
        salary = row.get("Salary", "").strip()
        team = row.get("Team", "").strip()
        if not team:
            raise PipelineError(f"Team 누락: {(season, playerid)}")
        if is_normalized_integer(salary):
            numeric_components[(team, int(salary))].append(row)
        else:
            # A numeric-looking but non-normalized Salary is safer to reject than
            # silently reinterpret as a status such as TBD/Pre-ARB/MiLB.
            if salary and any(char.isdigit() for char in salary):
                raise PipelineError(
                    f"Salary 열은 정수 문자열이어야 함: {(season, playerid)} {salary!r}"
                )
            status = row.get("Salary_raw", "").strip()
            if not status:
                raise PipelineError(
                    f"Salary와 Salary_raw가 모두 비어 있음: {(season, playerid)} team={team}"
                )
            non_numeric_components.add((team, status))
            salary_statuses.add(status)

    salary_total = sum(amount for _, amount in numeric_components)
    retained_total = sum(
        amount
        for (_, amount), component_rows in numeric_components.items()
        if any(row.get("contract_group") == GROUP_RETAINED for row in component_rows)
    )
    paying_teams = {team for team, _ in numeric_components}
    listed_teams = {row.get("Team", "").strip() for row in rows}
    contract_groups = {row.get("contract_group", "").strip() for row in rows}
    unique_component_count = len(numeric_components) + len(non_numeric_components)
    duplicate_rows_removed = len(rows) - unique_component_count
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


def stage2_build_player_salary(config: PipelineConfig) -> dict[str, int | str]:
    _, rows = read_unique_header_csv(
        config.combined_payroll_csv,
        required=RAW_PAYROLL_REQUIRED | {"_source_file"},
    )

    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for line_number, row in enumerate(rows, start=2):
        season = row["Season"].strip()
        playerid = row["playerid"].strip()
        if not season or not playerid:
            raise PipelineError(
                f"Season 또는 playerid 누락: {config.combined_payroll_csv.name}:{line_number}"
            )
        groups[(season, playerid)].append(row)

    output_rows = [
        aggregate_salary_group(season, playerid, group_rows)
        for (season, playerid), group_rows in sorted(
            groups.items(),
            key=lambda item: (int(item[0][0]), item[1][0].get("Player", ""), item[0][1]),
        )
    ]
    fields = list(output_rows[0])
    atomic_write_csv(config.salary_csv, fields, output_rows)

    complete = sum(int(row["salary_is_complete"]) for row in output_rows)
    return {
        "raw_rows": len(rows),
        "player_seasons": len(output_rows),
        "complete_salary": complete,
        "incomplete_salary": len(output_rows) - complete,
        "multi_team": sum(int(row["paying_team_count"]) > 1 for row in output_rows),
        "retained_salary": sum(int(row["has_retained_salary"]) for row in output_rows),
        "duplicates_removed": sum(
            int(row["duplicate_salary_rows_removed"]) for row in output_rows
        ),
        "output": str(config.salary_csv.resolve()),
    }


# ---------------------------------------------------------------------------
# Stage 3: join Standard/Value and create hitter sample flags
# ---------------------------------------------------------------------------


def read_value_csv(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    """Read Value CSV while allowing unrelated duplicate display headers.

    The current FanGraphs export contains duplicate '-- Line Break --' columns.
    Required analytical columns must still occur exactly once.
    """
    ensure_file(path, "FanGraphs Value")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            headers = next(reader)
        except StopIteration as exc:
            raise PipelineError(f"빈 Value CSV: {path}") from exc
        ensure_required_columns(headers, VALUE_REQUIRED, path, exact_once=True)
        counts = Counter(headers)
        duplicated_required = sorted(name for name in VALUE_REQUIRED if counts[name] != 1)
        if duplicated_required:
            raise PipelineError(f"Value 필수 열 중복/누락: {duplicated_required}")

        indices = {name: headers.index(name) for name in VALUE_REQUIRED}
        metric_indices = {
            output_name: headers.index(source_name)
            for source_name, output_name in VALUE_COLUMNS.items()
        }
        output: dict[tuple[str, str], dict[str, str]] = {}
        for line_number, row in enumerate(reader, start=2):
            if len(row) != len(headers):
                raise PipelineError(f"열 개수 불일치: {path.name}:{line_number}")
            key = (row[indices["Season"]], row[indices["playerid"]])
            if key in output:
                raise PipelineError(f"Value 중복 Season+playerid: {path.name}:{line_number} {key}")
            values = {name: row[index] for name, index in metric_indices.items()}
            values.update(
                {
                    "value_name": row[indices["Name"]],
                    "value_team": row[indices["Team"]],
                    "value_position": row[indices["player_position"]],
                    "value_source_url": row[indices["_source_url"]],
                    "value_collected_at": row[indices["_collected_at"]],
                }
            )
            output[key] = values

    if not output:
        raise PipelineError(f"Value CSV에 데이터 행이 없습니다: {path}")
    return output


def pa_band(is_hitter: bool, is_ohtani: bool, pa: int, *, include_ohtani: bool) -> str:
    if is_ohtani and not include_ohtani:
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


def stage3_prepare_batting_features(config: PipelineConfig) -> dict[str, int | str]:
    standard_fields, standard_rows = read_unique_header_csv(
        config.standard_csv,
        required=STANDARD_REQUIRED,
    )
    value_by_key = read_value_csv(config.value_csv)

    standard_by_key: dict[tuple[str, str], dict[str, str]] = {}
    for line_number, row in enumerate(standard_rows, start=2):
        key = (row["Season"], row["playerid"])
        if key in standard_by_key:
            raise PipelineError(
                f"Standard 중복 Season+playerid: {config.standard_csv.name}:{line_number} {key}"
            )
        standard_by_key[key] = row

    missing_value = sorted(set(standard_by_key) - set(value_by_key))
    missing_standard = sorted(set(value_by_key) - set(standard_by_key))
    if missing_value or missing_standard:
        raise PipelineError(
            "Standard/Value 키 불일치: "
            f"Value 누락={len(missing_value)}, Standard 누락={len(missing_standard)}; "
            f"예시 Value 누락={missing_value[:5]}, Standard 누락={missing_standard[:5]}"
        )

    ignored_standard = {"#", "_source_url", "_collected_at"}
    kept_standard_fields = [field for field in standard_fields if field not in ignored_standard]
    partial_snapshot_field = f"is_partial_{config.snapshot_season}_snapshot"
    output_rows: list[dict[str, Any]] = []

    for key in sorted(
        standard_by_key,
        key=lambda item: (int(item[0]), standard_by_key[item]["Name"], item[1]),
    ):
        standard = standard_by_key[key]
        value = value_by_key[key]
        if standard["Name"] != value["value_name"]:
            raise PipelineError(
                f"선수명 불일치: {key} {standard['Name']!r} / {value['value_name']!r}"
            )
        if standard["Team"] != value["value_team"]:
            raise PipelineError(
                f"팀 불일치: {key} {standard['Team']!r} / {value['value_team']!r}"
            )
        if standard["player_position"] != value["value_position"]:
            raise PipelineError(f"포지션 불일치: {key}")

        pa = parse_int_strict(standard["PA"], field="PA", context=str(key))
        if pa < 0:
            raise PipelineError(f"음수 PA: {key} {pa}")
        is_ohtani = standard["playerid"] == OHTANI_PLAYERID
        if config.include_ohtani:
            is_pitcher = standard["player_position"] == "P" and not is_ohtani
            is_hitter = is_ohtani or not is_pitcher
        else:
            is_pitcher = standard["player_position"] == "P"
            is_hitter = not is_pitcher and not is_ohtani

        row: dict[str, Any] = {field: standard[field] for field in kept_standard_fields}
        row.update({column: value[column] for column in VALUE_COLUMNS.values()})
        row.update(
            {
                "is_pitcher": int(is_pitcher),
                "is_shohei_ohtani": int(is_ohtani),
                "is_hitter": int(is_hitter),
                "pa_band": pa_band(
                    is_hitter,
                    is_ohtani,
                    pa,
                    include_ohtani=config.include_ohtani,
                ),
                "eligible_all_hitters_pa1": int(is_hitter and pa >= 1),
                "eligible_main_pa50": int(is_hitter and pa >= 50),
                "eligible_stable_pa100": int(is_hitter and pa >= 100),
                "eligible_regular_pa200": int(is_hitter and pa >= 200),
                partial_snapshot_field: int(standard["Season"] == str(config.snapshot_season)),
                "standard_source_url": standard["_source_url"],
                "standard_collected_at": standard["_collected_at"],
                "value_source_url": value["value_source_url"],
                "value_collected_at": value["value_collected_at"],
            }
        )
        output_rows.append(row)

    fields = list(output_rows[0])
    atomic_write_csv(config.batting_features_csv, fields, output_rows)
    return {
        "rows": len(output_rows),
        "pitcher_rows": sum(int(row["is_pitcher"]) for row in output_rows),
        "ohtani_rows": sum(int(row["is_shohei_ohtani"]) for row in output_rows),
        "hitter_rows": sum(int(row["is_hitter"]) for row in output_rows),
        "all_hitters_pa1": sum(int(row["eligible_all_hitters_pa1"]) for row in output_rows),
        "main_pa50": sum(int(row["eligible_main_pa50"]) for row in output_rows),
        "stable_pa100": sum(int(row["eligible_stable_pa100"]) for row in output_rows),
        "regular_pa200": sum(int(row["eligible_regular_pa200"]) for row in output_rows),
        "output": str(config.batting_features_csv.resolve()),
    }


# ---------------------------------------------------------------------------
# Stage 4: build current-season player/payroll features
# ---------------------------------------------------------------------------


def one_value(
    rows: Sequence[dict[str, str]], field: str, key: tuple[str, str]
) -> str:
    values = {row.get(field, "").strip() for row in rows if row.get(field, "").strip()}
    if len(values) > 1:
        raise PipelineError(f"동일 선수·시즌의 {field} 값 충돌: {key} {sorted(values)}")
    return next(iter(values), "")


def parse_age(raw: str, key: tuple[str, str]) -> float | None:
    token = raw.strip()
    if token in {"", "#N/A", "N/A", "NA", "null", "None"}:
        return None
    try:
        age = float(token)
    except ValueError as exc:
        raise PipelineError(f"Age 형식 오류: {key} {raw!r}") from exc
    if not math.isfinite(age) or not (15 <= age <= 60):
        raise PipelineError(f"Age 범위 오류: {key} {raw!r}")
    return age


def parse_service_time(
    raw: str, key: tuple[str, str]
) -> tuple[int, int, int] | None:
    token = raw.strip()
    if token in {"", "#N/A", "N/A", "NA", "null", "None"}:
        return None
    parts = token.split(".")
    if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
        raise PipelineError(f"Service Time 형식 오류: {key} {raw!r}")
    years, days = int(parts[0]), int(parts[1])
    if days >= SERVICE_DAYS_PER_YEAR:
        raise PipelineError(f"Service Time 일수 범위 오류: {key} {raw!r}")
    return years, days, years * SERVICE_DAYS_PER_YEAR + days


def contract_stage(groups: set[str], key: tuple[str, str]) -> str:
    if not groups:
        raise PipelineError(f"계약 그룹 누락: {key}")
    if len(groups) != 1:
        return "multiple"
    group = next(iter(groups))
    mapping = {
        GROUP_GUARANTEED: "guaranteed",
        GROUP_ARBITRATION: "arbitration",
        GROUP_PRE_ARB: "pre_arb",
        GROUP_RETAINED: "no_longer_40man",
    }
    try:
        return mapping[group]
    except KeyError as exc:
        raise PipelineError(f"알 수 없는 계약 그룹: {key} {group!r}") from exc


def stage4_prepare_payroll_features(config: PipelineConfig) -> dict[str, int | str]:
    _, raw_rows = read_unique_header_csv(
        config.combined_payroll_csv,
        required=RAW_PAYROLL_REQUIRED | {"_source_file"},
    )
    _, salary_rows = read_unique_header_csv(
        config.salary_csv,
        required={
            "Season", "Player", "playerid", "season_salary_total", "salary_is_complete",
            "salary_status", "paying_teams", "paying_team_count", "listed_teams",
            "listed_team_count", "salary_component_count", "retained_salary_total",
            "has_retained_salary", "contract_groups",
        },
    )

    raw_groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in raw_rows:
        raw_groups[(row["Season"], row["playerid"])].append(row)
    salary_by_key = unique_index(salary_rows, ("Season", "playerid"), "salary")

    if set(raw_groups) != set(salary_by_key):
        only_raw = sorted(set(raw_groups) - set(salary_by_key))[:10]
        only_salary = sorted(set(salary_by_key) - set(raw_groups))[:10]
        raise PipelineError(
            "원본/연봉 선수·시즌 키 불일치: "
            f"raw={len(raw_groups)}, salary={len(salary_by_key)}, "
            f"raw_only={only_raw}, salary_only={only_salary}"
        )

    output_rows: list[dict[str, Any]] = []
    age_missing_count = 0
    service_missing_count = 0

    for key in sorted(
        raw_groups,
        key=lambda item: (int(item[0]), salary_by_key[item]["Player"], item[1]),
    ):
        rows = raw_groups[key]
        salary = salary_by_key[key]
        age_raw = one_value(rows, "Age", key)
        service_raw = one_value(rows, "Service_Time", key)
        age = parse_age(age_raw, key)
        service = parse_service_time(service_raw, key)
        groups = {row["contract_group"].strip() for row in rows if row.get("contract_group", "").strip()}
        unknown_groups = groups - EXPECTED_CONTRACT_GROUPS
        if unknown_groups:
            raise PipelineError(f"알 수 없는 계약 그룹: {key} {sorted(unknown_groups)}")
        if joined(groups) != salary["contract_groups"]:
            raise PipelineError(
                f"연봉 테이블과 계약 그룹 불일치: {key} "
                f"raw={joined(groups)!r}, salary={salary['contract_groups']!r}"
            )

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
                "service_3_years_plus": "" if service is None else int(
                    int(service_total_days) >= 3 * SERVICE_DAYS_PER_YEAR
                ),
                "service_6_years_plus": "" if service is None else int(
                    int(service_total_days) >= 6 * SERVICE_DAYS_PER_YEAR
                ),
                "current_contract_stage": contract_stage(groups, key),
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

    fields = list(output_rows[0])
    atomic_write_csv(config.payroll_features_csv, fields, output_rows)
    return {
        "rows": len(output_rows),
        "age_missing": age_missing_count,
        "service_time_missing": service_missing_count,
        "guaranteed": sum(int(row["current_is_guaranteed"]) for row in output_rows),
        "arbitration": sum(int(row["current_is_arbitration"]) for row in output_rows),
        "pre_arb": sum(int(row["current_is_pre_arb"]) for row in output_rows),
        "no_longer_40man": sum(
            int(row["current_is_no_longer_40man"]) for row in output_rows
        ),
        "multiple_contract_groups": sum(
            int(row["current_has_multiple_contract_groups"]) for row in output_rows
        ),
        "output": str(config.payroll_features_csv.resolve()),
    }


# ---------------------------------------------------------------------------
# Stage 5: align feature season with next-season salary
# ---------------------------------------------------------------------------


def stage5_build_modeling_datasets(config: PipelineConfig) -> dict[str, Any]:
    batting_fields, batting_rows = read_unique_header_csv(
        config.batting_features_csv,
        required={
            "Season", "Name", "Team", "playerid", "player_position", "is_hitter",
            "eligible_all_hitters_pa1", "eligible_main_pa50",
            "eligible_stable_pa100", "eligible_regular_pa200",
        },
    )
    payroll_fields, payroll_rows = read_unique_header_csv(
        config.payroll_features_csv,
        required={
            "Season", "Player", "playerid", "current_salary_total",
            "current_salary_is_complete", "current_contract_stage",
        },
    )
    _, salary_rows = read_unique_header_csv(
        config.salary_csv,
        required={
            "Season", "Player", "playerid", "season_salary_total",
            "salary_is_complete", "salary_status",
        },
    )

    unique_index(batting_rows, ("Season", "playerid"), "batting features")
    payroll_by_key = unique_index(payroll_rows, ("Season", "playerid"), "payroll features")
    salary_by_key = unique_index(salary_rows, ("Season", "playerid"), "salary")

    payroll_names_by_playerid: dict[str, set[str]] = {}
    for payroll in payroll_rows:
        payroll_names_by_playerid.setdefault(payroll["playerid"], set()).add(payroll["Player"])
    canonical_payroll_name = {
        playerid: sorted(
            names,
            key=lambda name: (name.endswith("..."), -len(name), name),
        )[0]
        for playerid, names in payroll_names_by_playerid.items()
    }

    payroll_copy_fields = [
        field for field in payroll_fields if field not in {"Season", "Player", "playerid"}
    ]
    batting_copy_fields = [field for field in batting_fields if field != "Name"]
    master_rows: list[dict[str, Any]] = []

    for batting in batting_rows:
        feature_season = parse_int_strict(
            batting["Season"], field="Season", context=batting.get("Name", "batting")
        )
        if (
            feature_season < config.payroll_start_season
            or feature_season > config.snapshot_season
            or batting["is_hitter"] != "1"
        ):
            continue

        key = (batting["Season"], batting["playerid"])
        current = payroll_by_key.get(key)
        target_year = feature_season + 1
        target = (
            salary_by_key.get((str(target_year), batting["playerid"]))
            if target_year <= config.snapshot_season
            else None
        )
        has_current = current is not None
        current_complete = bool(current) and current["current_salary_is_complete"] == "1"
        has_target = target is not None
        target_complete = bool(target) and target["salary_is_complete"] == "1"
        training_period = config.payroll_start_season <= feature_season < config.snapshot_season
        prediction_period = feature_season == config.snapshot_season
        train_base = training_period and current_complete and target_complete
        predict_base = prediction_period and current_complete

        row: dict[str, Any] = {
            "Season": batting["Season"],
            "target_salary_season": target_year,
            "Player": (
                current["Player"]
                if current
                else canonical_payroll_name.get(batting["playerid"], batting["Name"])
            ),
            "batting_name": batting["Name"],
        }
        row.update({field: batting[field] for field in batting_copy_fields if field != "Season"})
        for field in payroll_copy_fields:
            row[field] = current[field] if current else ""

        current_salary = current["current_salary_total"] if current else ""
        target_salary = target["season_salary_total"] if target else ""
        context = f"Season={feature_season}, playerid={batting['playerid']}"
        row.update(
            {
                "log1p_current_salary": salary_log1p(current_salary, context=context),
                "target_salary_total": target_salary,
                "log1p_target_salary": salary_log1p(target_salary, context=context),
                "target_salary_is_complete": target["salary_is_complete"] if target else "",
                "target_salary_status": target["salary_status"] if target else "",
                "has_current_payroll": int(has_current),
                "has_target_payroll_record": int(has_target),
                "is_training_period": int(training_period),
                f"is_{config.snapshot_season + 1}_prediction_period": int(prediction_period),
                "feature_is_covid_2020": int(feature_season == 2020),
                "target_is_covid_2020": int(target_year == 2020),
                "train_base_eligible": int(train_base),
                "predict_base_eligible": int(predict_base),
                "train_eligible_all_hitters_pa1": int(
                    train_base and batting["eligible_all_hitters_pa1"] == "1"
                ),
                "train_eligible_main_pa50": int(
                    train_base and batting["eligible_main_pa50"] == "1"
                ),
                "train_eligible_stable_pa100": int(
                    train_base and batting["eligible_stable_pa100"] == "1"
                ),
                "train_eligible_regular_pa200": int(
                    train_base and batting["eligible_regular_pa200"] == "1"
                ),
                "predict_eligible_all_hitters_pa1": int(
                    predict_base and batting["eligible_all_hitters_pa1"] == "1"
                ),
                "predict_eligible_main_pa50": int(
                    predict_base and batting["eligible_main_pa50"] == "1"
                ),
                "predict_eligible_stable_pa100": int(
                    predict_base and batting["eligible_stable_pa100"] == "1"
                ),
                "predict_eligible_regular_pa200": int(
                    predict_base and batting["eligible_regular_pa200"] == "1"
                ),
            }
        )
        master_rows.append(row)

    if not master_rows:
        raise PipelineError("모델링 마스터 데이터가 비어 있습니다.")
    fields = list(master_rows[0])
    for row_number, row in enumerate(master_rows, start=1):
        if list(row) != fields:
            raise PipelineError(f"행별 열 순서 불일치: master row {row_number}")

    training_rows = [row for row in master_rows if row["train_base_eligible"] == 1]
    prediction_rows = [row for row in master_rows if row["predict_base_eligible"] == 1]
    if not training_rows:
        raise PipelineError("학습 데이터가 비어 있습니다.")
    if not prediction_rows:
        raise PipelineError("예측 데이터가 비어 있습니다.")

    atomic_write_csv(config.modeling_master_csv, fields, master_rows)
    atomic_write_csv(config.training_csv, fields, training_rows)
    atomic_write_csv(config.prediction_csv, fields, prediction_rows)

    return {
        "master_rows": len(master_rows),
        "training_rows": len(training_rows),
        "prediction_rows": len(prediction_rows),
        "master_missing_current_payroll": sum(
            row["has_current_payroll"] == 0 for row in master_rows
        ),
        "training_missing_target_record": sum(
            row["is_training_period"] == 1 and row["has_target_payroll_record"] == 0
            for row in master_rows
        ),
        "train_all_pa1": sum(row["train_eligible_all_hitters_pa1"] for row in master_rows),
        "train_main_pa50": sum(row["train_eligible_main_pa50"] for row in master_rows),
        "train_stable_pa100": sum(row["train_eligible_stable_pa100"] for row in master_rows),
        "train_regular_pa200": sum(row["train_eligible_regular_pa200"] for row in master_rows),
        "predict_all_pa1": sum(
            row["predict_eligible_all_hitters_pa1"] for row in master_rows
        ),
        "predict_main_pa50": sum(row["predict_eligible_main_pa50"] for row in master_rows),
        "predict_stable_pa100": sum(
            row["predict_eligible_stable_pa100"] for row in master_rows
        ),
        "predict_regular_pa200": sum(
            row["predict_eligible_regular_pa200"] for row in master_rows
        ),
        "output_dir": str(config.model_input_dir.resolve()),
    }


# ---------------------------------------------------------------------------
# Stage 6: engineer rates and write model-ready tables/manifest
# ---------------------------------------------------------------------------


def safe_ratio(
    numerator: float | None,
    denominator: float | None,
    multiplier: float = 1.0,
) -> str | float:
    if numerator is None or denominator is None or denominator <= 0:
        return ""
    return round(numerator / denominator * multiplier, 10)


def row_number(
    row: dict[str, str], field: str, *, missing_policy: str
) -> float | None:
    context = f"Season={row.get('Season', '')}, playerid={row.get('playerid', '')}"
    value = parse_optional_number(row.get(field), field=field, context=context)
    if value is None and missing_policy == "legacy-zero":
        return 0.0
    return value


def engineer_model_features(
    row: dict[str, str], *, missing_policy: str
) -> dict[str, Any]:
    number = lambda field: row_number(row, field, missing_policy=missing_policy)
    pa = number("PA")
    games = number("G")
    sb = number("SB")
    cs = number("CS")
    retained = number("current_retained_salary_total")

    stolen_attempts = None if sb is None or cs is None else sb + cs
    if retained is not None and retained < 0:
        raise PipelineError(
            f"음수 retained salary: Season={row.get('Season')}, playerid={row.get('playerid')}"
        )
    output: dict[str, Any] = {
        "pa_per_game": safe_ratio(pa, games),
        "hit_rate": safe_ratio(number("H"), pa),
        "single_rate": safe_ratio(number("1B"), pa),
        "double_rate": safe_ratio(number("2B"), pa),
        "triple_rate": safe_ratio(number("3B"), pa),
        "home_run_rate": safe_ratio(number("HR"), pa),
        "run_rate": safe_ratio(number("R"), pa),
        "rbi_rate": safe_ratio(number("RBI"), pa),
        "walk_rate": safe_ratio(number("BB"), pa),
        "intentional_walk_rate": safe_ratio(number("IBB"), pa),
        "strikeout_rate": safe_ratio(number("SO"), pa),
        "hbp_rate": safe_ratio(number("HBP"), pa),
        "sac_fly_rate": safe_ratio(number("SF"), pa),
        "sac_hit_rate": safe_ratio(number("SH"), pa),
        "gidp_rate": safe_ratio(number("GDP"), pa),
        "stolen_base_rate": safe_ratio(sb, pa),
        "caught_stealing_rate": safe_ratio(cs, pa),
        "stolen_base_success_rate": safe_ratio(sb, stolen_attempts),
        "value_batting_per_600_pa": safe_ratio(number("value_batting"), pa, 600),
        "value_base_running_per_600_pa": safe_ratio(
            number("value_base_running"), pa, 600
        ),
        "value_fielding_per_600_pa": safe_ratio(number("value_fielding"), pa, 600),
        "value_positional_per_600_pa": safe_ratio(
            number("value_positional"), pa, 600
        ),
        "value_offense_per_600_pa": safe_ratio(number("value_offense"), pa, 600),
        "value_defense_per_600_pa": safe_ratio(number("value_defense"), pa, 600),
        "value_rar_per_600_pa": safe_ratio(number("value_rar"), pa, 600),
        "value_war_per_600_pa": safe_ratio(number("value_war"), pa, 600),
        "value_dollars_per_600_pa": safe_ratio(number("value_dollars"), pa, 600),
        "log1p_current_retained_salary": (
            "" if retained is None else round(math.log1p(retained), 10)
        ),
    }
    return output


def select_model_row(
    row: dict[str, str], *, missing_policy: str
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for field in (
        ID_COLUMNS
        + CATEGORICAL_COLUMNS
        + RAW_BATTING_COLUMNS
        + RAW_VALUE_COLUMNS
        + PLAYER_COLUMNS
    ):
        if field == "value_dollars":
            value = row_number(row, field, missing_policy=missing_policy)
            output[field] = "" if value is None else value
        else:
            output[field] = row.get(field, "")
    output.update(engineer_model_features(row, missing_policy=missing_policy))
    for field in SAMPLE_FLAG_COLUMNS + TARGET_COLUMNS:
        output[field] = row.get(field, "")
    return output


def feature_manifest(snapshot_season: int = 2026) -> dict[str, Any]:
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
    expanded_numeric = list(
        dict.fromkeys(
            [
                "target_salary_season", *RAW_BATTING_COLUMNS, *RAW_VALUE_COLUMNS,
                *RATE_COLUMNS, *VALUE_RATE_COLUMNS, *PLAYER_COLUMNS,
                *ENGINEERED_PAYROLL_COLUMNS,
            ]
        )
    )
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
            f"full_main_rate_safe avoids raw cumulative batting/value totals because {snapshot_season} is a partial-season snapshot",
            "full_expanded_sensitivity retains raw cumulative features for comparison only",
            "identifiers, source metadata, eligibility flags, and future payroll attributes are not model inputs",
        ],
    }


def validate_model_ready_rows(
    rows: Sequence[dict[str, Any]],
    manifest: dict[str, Any],
    *,
    is_training: bool,
) -> None:
    if not rows:
        raise PipelineError("학습/예측 model-ready 데이터가 비어 있습니다.")
    fields = set(rows[0])
    forbidden = {"Contract", "AAV", "target_contract_groups", "target_age", "target_service_time"}
    if fields & forbidden:
        raise PipelineError(f"누수 위험 열 포함: {sorted(fields & forbidden)}")
    for name, spec in manifest["feature_sets"].items():
        missing = (set(spec["numeric"]) | set(spec["categorical"])) - fields
        if missing:
            raise PipelineError(f"Feature manifest 열 누락 ({name}): {sorted(missing)}")

    if is_training:
        missing_targets = sum(row["log1p_target_salary"] in {"", None} for row in rows)
        if missing_targets:
            raise PipelineError(f"학습 데이터 목표 연봉 결측: {missing_targets}행")
    else:
        leaked_targets = sum(row["log1p_target_salary"] not in {"", None} for row in rows)
        if leaked_targets:
            raise PipelineError(f"예측 데이터에 목표 연봉 포함: {leaked_targets}행")


def stage6_prepare_model_features(config: PipelineConfig) -> dict[str, Any]:
    training_fields, training_raw = read_unique_header_csv(config.training_csv)
    prediction_fields, prediction_raw = read_unique_header_csv(config.prediction_csv)
    if training_fields != prediction_fields:
        raise PipelineError("학습/예측 입력 CSV의 열 구조 또는 순서가 다릅니다.")

    training = [
        select_model_row(row, missing_policy=config.numeric_missing_policy)
        for row in training_raw
    ]
    prediction = [
        select_model_row(row, missing_policy=config.numeric_missing_policy)
        for row in prediction_raw
    ]
    manifest = feature_manifest(config.snapshot_season)
    validate_model_ready_rows(training, manifest, is_training=True)
    validate_model_ready_rows(prediction, manifest, is_training=False)

    training_output_fields = list(training[0])
    if list(prediction[0]) != training_output_fields:
        raise PipelineError("학습/예측 model-ready 열 구조가 다릅니다.")
    atomic_write_csv(config.training_model_ready_csv, training_output_fields, training)
    atomic_write_csv(config.prediction_model_ready_csv, training_output_fields, prediction)
    atomic_write_json(config.feature_manifest_json, manifest)

    return {
        "training_rows": len(training),
        "prediction_rows": len(prediction),
        "columns": len(training_output_fields),
        "output_dir": str(config.model_ready_dir.resolve()),
    }


# ---------------------------------------------------------------------------
# Integrated audits
# ---------------------------------------------------------------------------


def audit_interim(config: PipelineConfig) -> dict[str, Any]:
    _, combined = read_unique_header_csv(config.combined_payroll_csv)
    _, salary = read_unique_header_csv(config.salary_csv)
    batting_fields, batting = read_unique_header_csv(config.batting_features_csv)
    payroll_fields, payroll = read_unique_header_csv(config.payroll_features_csv)

    salary_by_key = unique_index(salary, ("Season", "playerid"), "salary")
    batting_by_key = unique_index(batting, ("Season", "playerid"), "batting")
    payroll_by_key = unique_index(payroll, ("Season", "playerid"), "payroll features")
    if set(salary_by_key) != set(payroll_by_key):
        raise PipelineError("audit: salary와 payroll features의 선수·시즌 키가 다름")

    raw_groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in combined:
        raw_groups[(row["Season"], row["playerid"])].append(row)
    if set(raw_groups) != set(salary_by_key):
        raise PipelineError("audit: combined와 salary의 선수·시즌 키가 다름")

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
            raise PipelineError(
                f"audit: 총연봉 합계 불일치 {key}: {expected_total} != {actual_total}"
            )
        expected_removed = len(rows) - len(numeric_components) - len(non_numeric_components)
        actual_removed = int(salary_by_key[key]["duplicate_salary_rows_removed"])
        if expected_removed != actual_removed:
            raise PipelineError(
                f"audit: 중복 제거 건수 불일치 {key}: {expected_removed} != {actual_removed}"
            )
        duplicate_components_removed += expected_removed

    if "Contract" in payroll_fields or any("AAV" in field.upper() for field in payroll_fields):
        raise PipelineError("audit: Payroll feature에 Contract 또는 AAV가 포함됨")

    flag_errors = 0
    for row in batting:
        is_pitcher = int(row["is_pitcher"])
        is_ohtani = int(row["is_shohei_ohtani"])
        is_hitter = int(row["is_hitter"])
        if config.include_ohtani:
            classification_ok = (
                (is_ohtani == 1 and is_pitcher == 0 and is_hitter == 1)
                or (is_ohtani == 0 and is_pitcher + is_hitter == 1)
            )
        else:
            classification_ok = is_pitcher + is_ohtani + is_hitter == 1
        nested = (
            int(row["eligible_regular_pa200"])
            <= int(row["eligible_stable_pa100"])
            <= int(row["eligible_main_pa50"])
            <= int(row["eligible_all_hitters_pa1"])
            <= is_hitter
        )
        if not classification_ok or not nested:
            flag_errors += 1
    if flag_errors:
        raise PipelineError(f"audit: 타자/PA 플래그 논리 오류 {flag_errors}행")

    raw_file_check: str | int = "skipped (raw directory unavailable or incomplete)"
    try:
        raw_files = discover_payroll_files(config)
    except PipelineError:
        raw_files = {}
    if raw_files:
        raw_row_counts: Counter[str] = Counter()
        for path in raw_files.values():
            _, rows = read_unique_header_csv(path)
            raw_row_counts[path.name] = len(rows)
        combined_row_counts = Counter(row["_source_file"] for row in combined)
        if raw_row_counts != combined_row_counts:
            raise PipelineError("audit: 원본 파일별 행 수와 통합 파일의 행 수가 다름")
        raw_file_check = len(raw_files)

    shared_keys = set(batting_by_key) & set(payroll_by_key)
    name_mismatches = sum(
        batting_by_key[key]["Name"] != payroll_by_key[key]["Player"]
        for key in shared_keys
    )
    return {
        "status": "PASSED",
        "raw_files_checked": raw_file_check,
        "combined_rows": len(combined),
        "salary_player_seasons": len(salary_by_key),
        "batting_player_seasons": len(batting_by_key),
        "payroll_feature_player_seasons": len(payroll_by_key),
        "duplicate_salary_components_removed": duplicate_components_removed,
        "incomplete_salary_rows": sum(row["salary_is_complete"] == "0" for row in salary),
        "age_missing": sum(row["age_missing"] == "1" for row in payroll),
        "service_time_missing": sum(row["service_time_missing"] == "1" for row in payroll),
        "batting_payroll_shared_keys": len(shared_keys),
        "name_mismatches_on_shared_playerid": name_mismatches,
        "batting_columns": len(batting_fields),
    }


def audit_final(config: PipelineConfig) -> dict[str, Any]:
    _, training = read_unique_header_csv(config.training_csv)
    _, prediction = read_unique_header_csv(config.prediction_csv)
    ready_training_fields, ready_training = read_unique_header_csv(
        config.training_model_ready_csv
    )
    ready_prediction_fields, ready_prediction = read_unique_header_csv(
        config.prediction_model_ready_csv
    )

    if len(training) != len(ready_training):
        raise PipelineError("audit: training 입력/model-ready 행 수 불일치")
    if len(prediction) != len(ready_prediction):
        raise PipelineError("audit: prediction 입력/model-ready 행 수 불일치")
    if ready_training_fields != ready_prediction_fields:
        raise PipelineError("audit: training/prediction model-ready 열 구조 불일치")

    training_keys = unique_index(training, ("Season", "playerid"), "training")
    prediction_keys = unique_index(prediction, ("Season", "playerid"), "prediction")
    if set(training_keys) & set(prediction_keys):
        raise PipelineError("audit: training과 prediction에 같은 Season+playerid 존재")

    for row in training:
        if int(row["target_salary_season"]) != int(row["Season"]) + 1:
            raise PipelineError(
                f"audit: 학습 목표 시즌 정렬 오류 {row['Season']} {row['playerid']}"
            )
        if row["log1p_target_salary"] == "":
            raise PipelineError("audit: 학습 목표값 결측")
    for row in prediction:
        if int(row["Season"]) != config.snapshot_season:
            raise PipelineError("audit: 예측 입력에 snapshot 이외 시즌 포함")
        if int(row["target_salary_season"]) != config.snapshot_season + 1:
            raise PipelineError("audit: 예측 목표 시즌 오류")
        if row["log1p_target_salary"] != "":
            raise PipelineError("audit: 예측 입력에 목표값 누수")

    ensure_file(config.feature_manifest_json, "feature manifest")
    with config.feature_manifest_json.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    validate_model_ready_rows(ready_training, manifest, is_training=True)
    validate_model_ready_rows(ready_prediction, manifest, is_training=False)

    return {
        "status": "PASSED",
        "training_rows": len(training),
        "prediction_rows": len(prediction),
        "model_ready_columns": len(ready_training_fields),
        "training_pa50": sum(row["train_eligible_main_pa50"] == "1" for row in training),
        "prediction_pa50": sum(
            row["predict_eligible_main_pa50"] == "1" for row in prediction
        ),
    }


# ---------------------------------------------------------------------------
# CLI orchestration
# ---------------------------------------------------------------------------


def infer_project_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    if script_dir.name == "02_data_preprocessing":
        return script_dir.parent
    return Path.cwd().resolve()


def parse_teams(raw: str) -> tuple[str, ...]:
    teams = tuple(part.strip().upper() for part in raw.split(",") if part.strip())
    if not teams:
        raise argparse.ArgumentTypeError("teams 목록이 비어 있습니다.")
    invalid = [team for team in teams if not re.fullmatch(r"[A-Z]+", team)]
    if invalid:
        raise argparse.ArgumentTypeError(f"잘못된 팀 코드: {invalid}")
    if len(teams) != len(set(teams)):
        raise argparse.ArgumentTypeError("teams 목록에 중복이 있습니다.")
    return teams


def resolve_override(path: Path | None, default: Path, project_root: Path) -> Path:
    selected = default if path is None else path
    if not selected.is_absolute():
        selected = project_root / selected
    return selected.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--project-root", type=Path, default=infer_project_root())
    parser.add_argument(
        "--preprocessing-dir",
        type=Path,
        default=None,
        help="interim/model_input/model_ready 폴더를 둘 경로",
    )
    parser.add_argument("--payroll-dir", type=Path, default=None)
    parser.add_argument("--standard", type=Path, default=None)
    parser.add_argument("--value", type=Path, default=None)
    parser.add_argument(
        "--payroll-start-season", type=int, default=DEFAULT_PAYROLL_START_SEASON
    )
    parser.add_argument(
        "--batting-start-season", type=int, default=DEFAULT_BATTING_START_SEASON
    )
    parser.add_argument("--snapshot-season", type=int, default=DEFAULT_SNAPSHOT_SEASON)
    parser.add_argument(
        "--teams",
        type=parse_teams,
        default=DEFAULT_TEAMS,
        help="쉼표로 구분한 팀 코드",
    )
    parser.add_argument(
        "--include-ohtani",
        action="store_true",
        help="오타니를 타자 표본에 포함. 기본값은 원본 03 코드와 동일하게 제외",
    )
    parser.add_argument(
        "--numeric-missing-policy",
        choices=("preserve", "legacy-zero"),
        default="preserve",
        help=(
            "Stage 6 수치 결측 처리. preserve는 결측을 유지하고, legacy-zero는 "
            "원본 06 코드처럼 결측을 0으로 바꿔 기존 파일을 정확히 재현"
        ),
    )
    parser.add_argument("--start-stage", type=int, choices=range(1, 7), default=1)
    parser.add_argument("--end-stage", type=int, choices=range(1, 7), default=6)
    parser.add_argument("--skip-audit", action="store_true")
    parser.add_argument(
        "--check-inputs-only",
        action="store_true",
        help="원본 입력 경로와 설정된 Payroll 파일만 검사하고 종료",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="기존 산출물을 다시 만들지 않고 감사만 실행",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="실행 요약을 JSON으로 추가 저장",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PipelineConfig:
    project_root = args.project_root.resolve()
    preprocessing_dir = resolve_override(
        args.preprocessing_dir,
        project_root / "02_data_preprocessing",
        project_root,
    )
    payroll_dir = resolve_override(
        args.payroll_dir,
        project_root / "01_data_collection" / "downloads" / "payroll",
        project_root,
    )
    standard_csv = resolve_override(
        args.standard,
        project_root
        / "01_data_collection"
        / "downloads"
        / "standard"
        / f"fangraphs_standard_{args.batting_start_season}_{args.snapshot_season}.csv",
        project_root,
    )
    value_csv = resolve_override(
        args.value,
        project_root
        / "01_data_collection"
        / "downloads"
        / "value"
        / f"fangraphs_value_{args.batting_start_season}_{args.snapshot_season}.csv",
        project_root,
    )
    config = PipelineConfig(
        project_root=project_root,
        preprocessing_dir=preprocessing_dir,
        payroll_dir=payroll_dir,
        standard_csv=standard_csv,
        value_csv=value_csv,
        payroll_start_season=args.payroll_start_season,
        batting_start_season=args.batting_start_season,
        snapshot_season=args.snapshot_season,
        teams=tuple(args.teams),
        include_ohtani=args.include_ohtani,
        numeric_missing_policy=args.numeric_missing_policy,
    )
    if config.batting_start_season > config.payroll_start_season:
        raise PipelineError("batting-start-season은 payroll-start-season보다 늦을 수 없습니다.")
    if config.snapshot_season <= config.payroll_start_season:
        raise PipelineError("snapshot-season은 payroll-start-season보다 커야 합니다.")
    return config


def check_initial_inputs(config: PipelineConfig) -> dict[str, Any]:
    files = discover_payroll_files(config)
    ensure_file(config.standard_csv, "FanGraphs Standard")
    ensure_file(config.value_csv, "FanGraphs Value")
    standard_fields, standard_rows = read_unique_header_csv(
        config.standard_csv, required=STANDARD_REQUIRED
    )
    value_by_key = read_value_csv(config.value_csv)
    return {
        "payroll_dir": str(config.payroll_dir),
        "payroll_files": len(files),
        "expected_payroll_files": len(config.teams) * len(config.payroll_seasons),
        "standard_csv": str(config.standard_csv),
        "standard_rows": len(standard_rows),
        "standard_columns": len(standard_fields),
        "value_csv": str(config.value_csv),
        "value_rows": len(value_by_key),
        "status": "OK",
    }


def run_pipeline(config: PipelineConfig, args: argparse.Namespace) -> dict[str, Any]:
    if args.start_stage > args.end_stage:
        raise PipelineError("start-stage는 end-stage보다 클 수 없습니다.")

    results: dict[str, Any] = {
        "config": {
            "project_root": str(config.project_root),
            "preprocessing_dir": str(config.preprocessing_dir),
            "payroll_start_season": config.payroll_start_season,
            "batting_start_season": config.batting_start_season,
            "snapshot_season": config.snapshot_season,
            "team_count": len(config.teams),
            "include_ohtani": config.include_ohtani,
            "numeric_missing_policy": config.numeric_missing_policy,
        }
    }

    if args.check_inputs_only:
        print_stage("입력 파일 검사")
        results["input_check"] = check_initial_inputs(config)
        print_summary(results["input_check"])
        return results

    if args.audit_only:
        print_stage("중간 산출물 감사")
        results["interim_audit"] = audit_interim(config)
        print_summary(results["interim_audit"])
        print_stage("최종 산출물 감사")
        results["final_audit"] = audit_final(config)
        print_summary(results["final_audit"])
        return results

    stage_functions = {
        1: ("Stage 1/6 - Payroll 원본 통합", stage1_combine_payroll),
        2: ("Stage 2/6 - 선수·시즌 총연봉 집계", stage2_build_player_salary),
        3: ("Stage 3/6 - Standard/Value 결합 및 타자 플래그", stage3_prepare_batting_features),
        4: ("Stage 4/6 - 현재 시즌 Payroll 특성 생성", stage4_prepare_payroll_features),
        5: ("Stage 5/6 - 다음 시즌 연봉과 시점 정렬", stage5_build_modeling_datasets),
        6: ("Stage 6/6 - 모델용 변수 및 manifest 생성", stage6_prepare_model_features),
    }
    for stage in range(args.start_stage, args.end_stage + 1):
        title, function = stage_functions[stage]
        print_stage(title)
        stage_result = function(config)
        results[f"stage_{stage}"] = stage_result
        print_summary(stage_result)

    if not args.skip_audit and args.start_stage <= 4 <= args.end_stage:
        print_stage("중간 산출물 감사")
        results["interim_audit"] = audit_interim(config)
        print_summary(results["interim_audit"])
    if not args.skip_audit and args.end_stage >= 6:
        print_stage("최종 산출물 감사")
        results["final_audit"] = audit_final(config)
        print_summary(results["final_audit"])

    return results


def main() -> None:
    args = parse_args()
    config = build_config(args)
    results = run_pipeline(config, args)
    if args.summary_json is not None:
        summary_path = args.summary_json
        if not summary_path.is_absolute():
            summary_path = config.project_root / summary_path
        atomic_write_json(summary_path.resolve(), results)
        print(f"\nsummary_json: {summary_path.resolve()}")
    print("\nPipeline complete")


if __name__ == "__main__":
    try:
        main()
    except (PipelineError, OSError, csv.Error, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
