"""Modeling 4-1: build and validate expanding-window temporal splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


VALIDATION_YEARS = [2023, 2024, 2025, 2026]
INITIAL_TRAIN_END_YEAR = 2022
KEY_COLUMNS = ["Season", "target_salary_season", "Player", "playerid"]


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-input",
        type=Path,
        default=project_root / "02_data_preprocessing" / "model_ready" / "training_model_ready.csv",
    )
    parser.add_argument(
        "--prediction-input",
        type=Path,
        default=project_root / "02_data_preprocessing" / "model_ready" / "prediction_2027_model_ready.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    return parser.parse_args()


def load_main_samples(training_path: Path, prediction_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    training = pd.read_csv(training_path, encoding="utf-8-sig")
    prediction = pd.read_csv(prediction_path, encoding="utf-8-sig")
    training = training.loc[training["train_eligible_main_pa50"].eq(1)].copy()
    prediction = prediction.loc[prediction["predict_eligible_main_pa50"].eq(1)].copy()

    numeric = ["Season", "target_salary_season", "playerid"]
    training[numeric] = training[numeric].apply(pd.to_numeric, errors="raise")
    prediction[numeric] = prediction[numeric].apply(pd.to_numeric, errors="raise")
    training[["Season", "target_salary_season", "playerid"]] = training[
        ["Season", "target_salary_season", "playerid"]
    ].astype(int)
    prediction[["Season", "target_salary_season", "playerid"]] = prediction[
        ["Season", "target_salary_season", "playerid"]
    ].astype(int)
    return training, prediction


def validate_source_data(training: pd.DataFrame, prediction: pd.DataFrame) -> None:
    if training.empty or prediction.empty:
        raise ValueError("The main PA>=50 training or prediction sample is empty.")
    if training.duplicated(["Season", "playerid"]).any():
        raise ValueError("Duplicate player-season rows exist in the training sample.")
    if prediction.duplicated(["Season", "playerid"]).any():
        raise ValueError("Duplicate player-season rows exist in the prediction sample.")
    if not training["target_salary_season"].sub(training["Season"]).eq(1).all():
        raise ValueError("Training feature season and target salary season are not exactly one year apart.")
    if not prediction["target_salary_season"].sub(prediction["Season"]).eq(1).all():
        raise ValueError("Prediction feature season and target salary season are not exactly one year apart.")
    if training["target_salary_total"].isna().any() or training["log1p_target_salary"].isna().any():
        raise ValueError("Training targets contain missing values.")
    if set(prediction["target_salary_season"].unique()) != {2027}:
        raise ValueError("The prediction sample is not restricted to target salary year 2027.")
    if prediction["Season"].max() >= prediction["target_salary_season"].min():
        raise ValueError("Prediction features are not strictly earlier than their target year.")
    overlap = training.merge(prediction, on=["Season", "playerid"], how="inner")
    if not overlap.empty:
        raise ValueError("Training and 2027 prediction samples contain overlapping player-season rows.")


def build_fold_assignments(training: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    assignment_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for fold_number, validation_year in enumerate(VALIDATION_YEARS, start=1):
        train_end_year = validation_year - 1
        train_mask = training["target_salary_season"].between(
            training["target_salary_season"].min(), train_end_year
        )
        validation_mask = training["target_salary_season"].eq(validation_year)
        train_rows = training.loc[train_mask, KEY_COLUMNS].copy()
        validation_rows = training.loc[validation_mask, KEY_COLUMNS].copy()

        if train_rows.empty or validation_rows.empty:
            raise ValueError(f"Fold {fold_number} has an empty train or validation partition.")
        if train_rows["target_salary_season"].max() >= validation_rows["target_salary_season"].min():
            raise ValueError(f"Fold {fold_number} violates temporal ordering.")
        if train_rows.merge(validation_rows, on=["Season", "playerid"], how="inner").shape[0] != 0:
            raise ValueError(f"Fold {fold_number} contains overlapping player-season rows.")

        train_rows.insert(0, "role", "train")
        validation_rows.insert(0, "role", "validation")
        fold_rows = pd.concat([train_rows, validation_rows], ignore_index=True)
        fold_rows.insert(0, "fold", fold_number)
        assignment_frames.append(fold_rows)

        train_players = set(train_rows["playerid"])
        validation_players = set(validation_rows["playerid"])
        summary_rows.append(
            {
                "fold": fold_number,
                "train_target_year_start": int(train_rows["target_salary_season"].min()),
                "train_target_year_end": int(train_rows["target_salary_season"].max()),
                "validation_target_year": validation_year,
                "train_feature_year_start": int(train_rows["Season"].min()),
                "train_feature_year_end": int(train_rows["Season"].max()),
                "validation_feature_year": int(validation_rows["Season"].unique()[0]),
                "train_rows": len(train_rows),
                "validation_rows": len(validation_rows),
                "train_unique_players": train_rows["playerid"].nunique(),
                "validation_unique_players": validation_rows["playerid"].nunique(),
                "validation_players_seen_in_train": len(train_players & validation_players),
                "validation_players_new": len(validation_players - train_players),
            }
        )

    assignments = pd.concat(assignment_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    return assignments, summary


def build_config(training: pd.DataFrame, prediction: pd.DataFrame, summary: pd.DataFrame) -> dict[str, object]:
    folds = []
    for row in summary.itertuples(index=False):
        folds.append(
            {
                "fold": int(row.fold),
                "train_target_years": [int(row.train_target_year_start), int(row.train_target_year_end)],
                "validation_target_year": int(row.validation_target_year),
                "train_feature_years": [int(row.train_feature_year_start), int(row.train_feature_year_end)],
                "validation_feature_year": int(row.validation_feature_year),
                "train_rows": int(row.train_rows),
                "validation_rows": int(row.validation_rows),
            }
        )
    return {
        "strategy": "expanding_window_by_target_salary_year",
        "sample": "main_hitters_pa50",
        "feature_to_target_horizon_years": 1,
        "initial_training_target_years": [
            int(training["target_salary_season"].min()),
            INITIAL_TRAIN_END_YEAR,
        ],
        "folds": folds,
        "final_fit": {
            "target_years": [
                int(training["target_salary_season"].min()),
                int(training["target_salary_season"].max()),
            ],
            "feature_years": [int(training["Season"].min()), int(training["Season"].max())],
            "rows": len(training),
        },
        "final_prediction": {
            "feature_year": int(prediction["Season"].unique()[0]),
            "target_salary_year": int(prediction["target_salary_season"].unique()[0]),
            "rows": len(prediction),
            "partial_season_snapshot": True,
        },
        "primary_selection_metric": "MAE_dollars",
        "secondary_metrics": [
            "RMSE_dollars",
            "median_absolute_error_dollars",
            "RMSE_log1p_salary",
            "R2_log1p_salary",
        ],
        "aggregation": {
            "primary": "mean_across_validation_folds",
            "also_report": ["per_fold", "standard_deviation_across_folds"],
        },
        "leakage_rules": [
            "All preprocessing is fitted on each fold's training partition only.",
            "No target-year payroll attributes are model inputs.",
            "Hyperparameters are selected using 2023-2026 validation folds only.",
            "The 2026 feature snapshot is held out until final 2027 prediction.",
            "Repeated players across years are allowed because this matches the forecasting use case.",
        ],
    }


def write_notes(path: Path, summary: pd.DataFrame, training_rows: int, prediction_rows: int) -> None:
    lines = [
        "# 4-1 temporal validation design",
        "",
        "Random splitting is not used. Every validation season occurs strictly after all training seasons in its fold.",
        "",
        "| Fold | Training salary years | Validation salary year | Train rows | Validation rows |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.fold} | {row.train_target_year_start}-{row.train_target_year_end} | "
            f"{row.validation_target_year} | {row.train_rows:,} | {row.validation_rows:,} |"
        )
    lines.extend(
        [
            "",
            f"After model selection, all {training_rows:,} labeled rows from salary years 2020-2026 are refitted.",
            f"The final model then predicts 2027 salary for {prediction_rows:,} PA>=50 hitters using 2026 features.",
            "",
            "Repeated players may appear in both earlier training years and a later validation year. This is intentional: the real task predicts a player's future salary from information already available at prediction time.",
            "",
            "All imputers, scalers, and encoders must be fitted inside each training fold to prevent leakage.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    split_dir = output_dir / "splits"
    config_dir = output_dir / "config"
    split_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    training, prediction = load_main_samples(args.training_input.resolve(), args.prediction_input.resolve())
    validate_source_data(training, prediction)
    assignments, summary = build_fold_assignments(training)
    config = build_config(training, prediction, summary)

    assignments.to_csv(split_dir / "temporal_fold_assignments.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(split_dir / "temporal_split_summary.csv", index=False, encoding="utf-8-sig")
    training[KEY_COLUMNS].to_csv(split_dir / "final_training_rows.csv", index=False, encoding="utf-8-sig")
    prediction[KEY_COLUMNS].to_csv(split_dir / "prediction_2027_rows.csv", index=False, encoding="utf-8-sig")
    (config_dir / "temporal_validation.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_notes(output_dir / "01_temporal_validation_notes.md", summary, len(training), len(prediction))

    print("Temporal validation setup complete")
    print(f"training_rows: {len(training)}")
    print(f"prediction_2027_rows: {len(prediction)}")
    for row in summary.itertuples(index=False):
        print(
            f"  fold {row.fold}: train target {row.train_target_year_start}-{row.train_target_year_end} "
            f"({row.train_rows}), validate {row.validation_target_year} ({row.validation_rows})"
        )
    print(f"output_dir: {output_dir}")


if __name__ == "__main__":
    main()
