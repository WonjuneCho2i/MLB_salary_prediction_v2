import csv

from validate_exports import STANDARD_REQUIRED, VALUE_REQUIRED, read_csv, validate_report


def _write_csv(path, headers, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def test_valid_standard_and_value_shapes(tmp_path):
    standard_path = tmp_path / "standard.csv"
    value_path = tmp_path / "value.csv"
    standard_rows = []
    value_rows = []
    for index in range(550):
        season = 2018 + (index % 9)
        player_id = str(index + 1)
        shared = {"Season": season, "Name": f"Player {index}", "Team": "TST", "playerid": player_id}
        standard_rows.append({**shared, "PA": 400, "HR": 20})
        value_rows.append({**shared, "WAR": 2.0, "Dollars": "$16,000,000"})

    _write_csv(standard_path, sorted(STANDARD_REQUIRED), standard_rows)
    _write_csv(value_path, sorted(VALUE_REQUIRED), value_rows)

    standard = read_csv(standard_path)
    value = read_csv(value_path)
    assert validate_report(standard, "Standard", STANDARD_REQUIRED) == []
    assert validate_report(value, "Value", VALUE_REQUIRED) == []
    assert len(set(standard.keys) & set(value.keys)) == 550


def test_duplicate_player_season_is_rejected(tmp_path):
    path = tmp_path / "duplicate.csv"
    row = {
        "Season": 2025,
        "Name": "Player A",
        "Team": "TST",
        "PA": 100,
        "HR": 5,
        "playerid": "123",
    }
    _write_csv(path, sorted(STANDARD_REQUIRED), [row, row])
    summary = read_csv(path)
    errors = validate_report(summary, "Standard", STANDARD_REQUIRED)
    assert any("duplicate" in error for error in errors)
