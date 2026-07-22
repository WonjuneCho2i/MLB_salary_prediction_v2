"""Modeling 4-2: evaluate salary baselines on the temporal validation folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MODEL_ORDER = ["naive_current_salary", "dummy_median_log", "linear_previous_salary"]
TARGET_LOG = "log1p_target_salary"
TARGET_RAW = "target_salary_total"
ID_COLUMNS = ["Season", "target_salary_season", "Player", "playerid"]


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-input",
        type=Path,
        default=project_root / "02_data_preprocessing" / "model_ready" / "training_model_ready.csv",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=project_root / "02_data_preprocessing" / "model_ready" / "feature_manifest_final.json",
    )
    parser.add_argument(
        "--validation-config",
        type=Path,
        default=project_root / "04_modeling" / "config" / "temporal_validation.json",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    return parser.parse_args()


def dollars_from_log(prediction: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, np.expm1(np.asarray(prediction, dtype=float)))


def metric_row(
    model: str,
    fold: int,
    validation_year: int,
    actual_log: np.ndarray,
    actual_dollars: np.ndarray,
    predicted_log: np.ndarray,
) -> dict[str, object]:
    predicted_log = np.asarray(predicted_log, dtype=float)
    predicted_dollars = dollars_from_log(predicted_log)
    return {
        "model": model,
        "fold": fold,
        "validation_target_year": validation_year,
        "n": len(actual_log),
        "mae_dollars": mean_absolute_error(actual_dollars, predicted_dollars),
        "rmse_dollars": mean_squared_error(actual_dollars, predicted_dollars) ** 0.5,
        "median_absolute_error_dollars": median_absolute_error(actual_dollars, predicted_dollars),
        "rmse_log1p_salary": mean_squared_error(actual_log, predicted_log) ** 0.5,
        "r2_log1p_salary": r2_score(actual_log, predicted_log),
    }


def build_linear_pipeline(features: list[str]) -> Pipeline:
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                features,
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessing),
            ("model", LinearRegression()),
        ]
    )


def validate_inputs(data: pd.DataFrame, features: list[str], config: dict[str, object]) -> None:
    required = set(ID_COLUMNS + features + [TARGET_LOG, TARGET_RAW, "train_eligible_main_pa50"])
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if len(config["folds"]) == 0:
        raise ValueError("Temporal validation config contains no folds.")
    if data.loc[data["train_eligible_main_pa50"].eq(1), [TARGET_LOG, TARGET_RAW]].isna().any().any():
        raise ValueError("Main-sample training targets contain missing values.")


def evaluate(
    data: pd.DataFrame,
    features: list[str],
    config: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample = data.loc[data["train_eligible_main_pa50"].eq(1)].copy()
    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

    for fold_config in config["folds"]:
        fold = int(fold_config["fold"])
        train_start, train_end = fold_config["train_target_years"]
        validation_year = int(fold_config["validation_target_year"])
        train = sample.loc[sample["target_salary_season"].between(train_start, train_end)].copy()
        validation = sample.loc[sample["target_salary_season"].eq(validation_year)].copy()

        if len(train) != int(fold_config["train_rows"]) or len(validation) != int(fold_config["validation_rows"]):
            raise ValueError(f"Fold {fold} row counts do not match the 4-1 configuration.")
        if train["target_salary_season"].max() >= validation["target_salary_season"].min():
            raise ValueError(f"Fold {fold} violates temporal ordering.")

        actual_log = validation[TARGET_LOG].to_numpy(dtype=float)
        actual_dollars = validation[TARGET_RAW].to_numpy(dtype=float)
        predictions: dict[str, np.ndarray] = {}

        predictions["naive_current_salary"] = validation["log1p_current_salary"].to_numpy(dtype=float)

        dummy = DummyRegressor(strategy="median")
        dummy.fit(train[features], train[TARGET_LOG])
        predictions["dummy_median_log"] = dummy.predict(validation[features])

        linear = build_linear_pipeline(features)
        linear.fit(train[features], train[TARGET_LOG])
        predictions["linear_previous_salary"] = linear.predict(validation[features])

        for model_name in MODEL_ORDER:
            predicted_log = predictions[model_name]
            predicted_dollars = dollars_from_log(predicted_log)
            metric_rows.append(
                metric_row(
                    model_name,
                    fold,
                    validation_year,
                    actual_log,
                    actual_dollars,
                    predicted_log,
                )
            )
            frame = validation[ID_COLUMNS + ["current_contract_stage", TARGET_RAW, TARGET_LOG]].copy()
            frame.insert(0, "model", model_name)
            frame.insert(1, "fold", fold)
            frame["predicted_log1p_salary"] = predicted_log
            frame["predicted_salary_dollars"] = predicted_dollars
            frame["absolute_error_dollars"] = np.abs(actual_dollars - predicted_dollars)
            prediction_frames.append(frame)

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    return metrics, predictions


def summarize_metrics(metrics: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "mae_dollars",
        "rmse_dollars",
        "median_absolute_error_dollars",
        "rmse_log1p_salary",
        "r2_log1p_salary",
    ]
    rows: list[dict[str, object]] = []
    for model_name in MODEL_ORDER:
        model_metrics = metrics.loc[metrics["model"].eq(model_name)]
        model_predictions = predictions.loc[predictions["model"].eq(model_name)]
        actual_log = model_predictions[TARGET_LOG].to_numpy(dtype=float)
        actual_dollars = model_predictions[TARGET_RAW].to_numpy(dtype=float)
        predicted_log = model_predictions["predicted_log1p_salary"].to_numpy(dtype=float)
        oof_metrics = metric_row(
            model_name,
            fold=0,
            validation_year=0,
            actual_log=actual_log,
            actual_dollars=actual_dollars,
            predicted_log=predicted_log,
        )
        row: dict[str, object] = {
            "model": model_name,
            "fold_count": len(model_metrics),
            "oof_rows": len(model_predictions),
        }
        for metric in metric_columns:
            row[f"fold_mean_{metric}"] = model_metrics[metric].mean()
            row[f"fold_std_{metric}"] = model_metrics[metric].std(ddof=1)
            row[f"combined_oof_{metric}"] = oof_metrics[metric]
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary["rank_by_fold_mean_mae"] = summary["fold_mean_mae_dollars"].rank(method="min").astype(int)
    return summary.sort_values("rank_by_fold_mean_mae").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    result_dir = output_dir / "results"
    prediction_dir = output_dir / "predictions"
    result_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.training_input.resolve(), encoding="utf-8-sig")
    manifest = json.loads(args.manifest.resolve().read_text(encoding="utf-8"))
    config = json.loads(args.validation_config.resolve().read_text(encoding="utf-8"))
    features = manifest["feature_sets"]["baseline_previous_salary"]["numeric"]
    validate_inputs(data, features, config)

    metrics, predictions = evaluate(data, features, config)
    summary = summarize_metrics(metrics, predictions)
    metrics.to_csv(result_dir / "baseline_fold_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(result_dir / "baseline_summary_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(
        prediction_dir / "baseline_out_of_fold_predictions.csv", index=False, encoding="utf-8-sig"
    )

    print("Baseline temporal validation complete")
    print(f"features: {features}")
    for row in summary.itertuples(index=False):
        print(
            f"  rank {row.rank_by_fold_mean_mae}: {row.model}, "
            f"fold mean MAE=${row.fold_mean_mae_dollars:,.0f}, "
            f"OOF RMSE=${row.combined_oof_rmse_dollars:,.0f}, "
            f"OOF log R2={row.combined_oof_r2_log1p_salary:.4f}"
        )
    print(f"output_dir: {output_dir}")


if __name__ == "__main__":
    main()
