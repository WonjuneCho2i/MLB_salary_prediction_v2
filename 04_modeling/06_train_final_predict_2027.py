"""Modeling 4-6: refit the selected model and predict 2027 hitter salaries."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGET_LOG = "log1p_target_salary"
TARGET_RAW = "target_salary_total"
CURRENT_LOG = "log1p_current_salary"
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    modeling = project_root / "04_modeling"
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
    parser.add_argument(
        "--manifest",
        type=Path,
        default=project_root / "02_data_preprocessing" / "model_ready" / "feature_manifest_final.json",
    )
    parser.add_argument(
        "--selection-config",
        type=Path,
        default=modeling / "config" / "final_model_selection.json",
    )
    parser.add_argument(
        "--temporal-config",
        type=Path,
        default=modeling / "config" / "temporal_validation.json",
    )
    parser.add_argument(
        "--oof-predictions",
        type=Path,
        default=modeling / "predictions" / "tree_out_of_fold_predictions.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=modeling)
    return parser.parse_args()


def make_pipeline(
    numeric: list[str],
    categorical: list[str],
    hyperparameters: dict[str, object],
) -> Pipeline:
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    model = GradientBoostingRegressor(
        n_estimators=int(hyperparameters["n_estimators"]),
        learning_rate=float(hyperparameters["learning_rate"]),
        max_depth=int(hyperparameters["max_depth"]),
        min_samples_leaf=int(hyperparameters["min_samples_leaf"]),
        loss=str(hyperparameters["loss"]),
        random_state=RANDOM_STATE,
    )
    return Pipeline(steps=[("preprocess", preprocessing), ("model", model)])


def validate_inputs(
    training: pd.DataFrame,
    prediction: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    selection: dict[str, object],
    temporal: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = set(numeric + categorical + [CURRENT_LOG, "Season", "target_salary_season", "Player", "playerid"])
    missing_training = sorted((required | {TARGET_LOG, TARGET_RAW, "train_eligible_main_pa50"}) - set(training.columns))
    missing_prediction = sorted((required | {"predict_eligible_main_pa50"}) - set(prediction.columns))
    if missing_training or missing_prediction:
        raise ValueError(f"Missing columns: training={missing_training}, prediction={missing_prediction}")
    train_sample = training.loc[training["train_eligible_main_pa50"].eq(1)].copy()
    prediction_sample = prediction.loc[prediction["predict_eligible_main_pa50"].eq(1)].copy()
    if len(train_sample) != int(temporal["final_fit"]["rows"]):
        raise ValueError("Final training row count does not match the temporal-validation configuration.")
    if len(prediction_sample) != int(temporal["final_prediction"]["rows"]):
        raise ValueError("Prediction row count does not match the temporal-validation configuration.")
    if train_sample[[TARGET_LOG, TARGET_RAW, CURRENT_LOG]].isna().any().any():
        raise ValueError("Final training salary fields contain missing values.")
    if prediction_sample[CURRENT_LOG].isna().any():
        raise ValueError("Prediction current salary contains missing values.")
    if set(train_sample["target_salary_season"].astype(int).unique()) != set(range(2020, 2027)):
        raise ValueError("Final training target years are not 2020-2026.")
    if set(prediction_sample["Season"].astype(int).unique()) != {2026}:
        raise ValueError("Final prediction features are not restricted to 2026.")
    if set(prediction_sample["target_salary_season"].astype(int).unique()) != {2027}:
        raise ValueError("Final prediction target year is not 2027.")
    if selection["selected_model"] != "gradient_boosting__log_change_from_current":
        raise ValueError("Selection config does not contain the expected final model.")
    if train_sample.duplicated(["Season", "playerid"]).any() or prediction_sample.duplicated(["Season", "playerid"]).any():
        raise ValueError("Duplicate player-season rows exist in the final training or prediction sample.")
    return train_sample, prediction_sample


def empirical_log_interval(oof_path: Path, selected_model: str) -> tuple[float, int]:
    oof = pd.read_csv(oof_path, encoding="utf-8-sig")
    selected = oof.loc[oof["model"].eq(selected_model)].copy()
    if selected.empty:
        raise ValueError("Selected-model OOF predictions are missing.")
    absolute_log_error = np.abs(selected[TARGET_LOG] - selected["predicted_log1p_salary"])
    return float(absolute_log_error.quantile(0.80)), len(selected)


def build_predictions(
    prediction_sample: pd.DataFrame,
    predicted_log_change: np.ndarray,
    interval_radius: float,
) -> pd.DataFrame:
    predicted_log_salary = prediction_sample[CURRENT_LOG].to_numpy(dtype=float) + predicted_log_change
    predicted_salary = np.maximum(0.0, np.expm1(predicted_log_salary))
    current_salary = np.maximum(0.0, np.expm1(prediction_sample[CURRENT_LOG].to_numpy(dtype=float)))
    lower = np.maximum(0.0, np.expm1(predicted_log_salary - interval_radius))
    upper = np.maximum(0.0, np.expm1(predicted_log_salary + interval_radius))
    change = predicted_salary - current_salary
    change_percent = np.where(current_salary > 0, change / current_salary * 100, np.nan)

    desired_columns = [
        "Player", "playerid", "Team", "player_position", "current_contract_stage",
        "current_age", "current_service_total_days", "PA", "AVG", "value_war_per_600_pa",
    ]
    output = prediction_sample[[column for column in desired_columns if column in prediction_sample.columns]].copy()
    output.insert(0, "feature_season", prediction_sample["Season"].astype(int).to_numpy())
    output.insert(1, "predicted_salary_season", prediction_sample["target_salary_season"].astype(int).to_numpy())
    output["current_salary_2026"] = current_salary
    output["predicted_salary_2027"] = predicted_salary
    output["predicted_change_dollars"] = change
    output["predicted_change_percent"] = change_percent
    output["empirical_80pct_lower"] = lower
    output["empirical_80pct_upper"] = upper
    output["predicted_log1p_salary_2027"] = predicted_log_salary
    output["predicted_log_change"] = predicted_log_change
    output["prediction_rank"] = output["predicted_salary_2027"].rank(method="min", ascending=False).astype(int)
    output["prediction_snapshot_note"] = "2026 partial-season rate-safe features"
    return output.sort_values(["predicted_salary_2027", "Player"], ascending=[False, True]).reset_index(drop=True)


def summary_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    salary = predictions["predicted_salary_2027"]
    overall = pd.DataFrame(
        [
            {
                "prediction_year": 2027,
                "n": len(predictions),
                "mean": salary.mean(),
                "median": salary.median(),
                "p25": salary.quantile(0.25),
                "p75": salary.quantile(0.75),
                "p90": salary.quantile(0.90),
                "maximum": salary.max(),
                "minimum": salary.min(),
                "predicted_increase_count": predictions["predicted_change_dollars"].gt(0).sum(),
                "predicted_decrease_count": predictions["predicted_change_dollars"].lt(0).sum(),
            }
        ]
    )
    by_stage = (
        predictions.groupby("current_contract_stage", observed=False)["predicted_salary_2027"]
        .agg(n="count", mean="mean", median="median", minimum="min", maximum="max")
        .reset_index()
    )
    return overall, by_stage


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    model_dir = output_dir / "models"
    prediction_dir = output_dir / "predictions"
    result_dir = output_dir / "results"
    config_dir = output_dir / "config"
    model_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    training = pd.read_csv(args.training_input.resolve(), encoding="utf-8-sig")
    prediction = pd.read_csv(args.prediction_input.resolve(), encoding="utf-8-sig")
    manifest = json.loads(args.manifest.resolve().read_text(encoding="utf-8"))
    selection = json.loads(args.selection_config.resolve().read_text(encoding="utf-8"))
    temporal = json.loads(args.temporal_config.resolve().read_text(encoding="utf-8"))
    feature_set = manifest["feature_sets"][selection["final_fit_feature_set"]]
    numeric = feature_set["numeric"]
    categorical = feature_set["categorical"]
    feature_columns = numeric + categorical
    train_sample, prediction_sample = validate_inputs(
        training, prediction, numeric, categorical, selection, temporal
    )

    pipeline = make_pipeline(numeric, categorical, selection["hyperparameters"])
    train_target = (train_sample[TARGET_LOG] - train_sample[CURRENT_LOG]).to_numpy(dtype=float)
    pipeline.fit(train_sample[feature_columns], train_target)
    predicted_change = pipeline.predict(prediction_sample[feature_columns])
    interval_radius, interval_oof_n = empirical_log_interval(
        args.oof_predictions.resolve(), str(selection["selected_model"])
    )
    predictions = build_predictions(prediction_sample, predicted_change, interval_radius)

    if not np.isfinite(predictions["predicted_salary_2027"]).all():
        raise ValueError("Non-finite 2027 salary predictions were produced.")
    if predictions["predicted_salary_2027"].lt(0).any():
        raise ValueError("Negative 2027 salary predictions were produced.")
    if len(predictions) != prediction_sample["playerid"].nunique():
        raise ValueError("Final predictions are not one row per player.")

    transformed_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    importance = pd.DataFrame(
        {
            "feature": transformed_names,
            "importance": pipeline.named_steps["model"].feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    overall, by_stage = summary_tables(predictions)

    predictions.to_csv(
        prediction_dir / "mlb_hitters_salary_prediction_2027.csv", index=False, encoding="utf-8-sig"
    )
    predictions.head(50).to_csv(
        prediction_dir / "top_50_salary_predictions_2027.csv", index=False, encoding="utf-8-sig"
    )
    predictions.nlargest(25, "predicted_change_dollars").to_csv(
        prediction_dir / "top_25_predicted_salary_increases_2027.csv", index=False, encoding="utf-8-sig"
    )
    predictions.nsmallest(25, "predicted_change_dollars").to_csv(
        prediction_dir / "top_25_predicted_salary_decreases_2027.csv", index=False, encoding="utf-8-sig"
    )
    overall.to_csv(result_dir / "prediction_2027_summary.csv", index=False, encoding="utf-8-sig")
    by_stage.to_csv(result_dir / "prediction_2027_by_contract_stage.csv", index=False, encoding="utf-8-sig")
    importance.to_csv(result_dir / "final_refit_feature_importance.csv", index=False, encoding="utf-8-sig")

    artifact = {
        "pipeline": pipeline,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "selected_model": selection,
        "empirical_80pct_log_interval_radius": interval_radius,
    }
    joblib.dump(artifact, model_dir / "final_gradient_boosting_2027.joblib")
    metadata = {
        "selected_model": selection["selected_model"],
        "training_rows": len(train_sample),
        "training_feature_years": [2019, 2025],
        "training_target_salary_years": [2020, 2026],
        "prediction_rows": len(predictions),
        "prediction_feature_year": 2026,
        "prediction_target_salary_year": 2027,
        "partial_season_snapshot": True,
        "feature_set": selection["final_fit_feature_set"],
        "empirical_interval": {
            "nominal_coverage": 0.80,
            "method": "symmetric absolute selected-model OOF log-error quantile",
            "oof_rows": interval_oof_n,
            "log_radius": interval_radius,
            "note": "Diagnostic empirical range, not a guaranteed contractual or conformal interval.",
        },
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    (config_dir / "final_model_2027_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    row = overall.iloc[0]
    print("Final 2027 salary prediction complete")
    print(f"training_rows: {len(train_sample)}")
    print(f"prediction_rows: {len(predictions)}")
    print(f"predicted_mean: ${row['mean']:,.0f}")
    print(f"predicted_median: ${row['median']:,.0f}")
    print(f"predicted_minimum: ${row['minimum']:,.0f}")
    print(f"predicted_maximum: ${row['maximum']:,.0f}")
    print(f"empirical_80pct_log_radius: {interval_radius:.4f}")
    print("top_10_predictions:")
    for result in predictions.head(10).itertuples(index=False):
        print(
            f"  {result.prediction_rank}. {result.Player}: "
            f"${result.predicted_salary_2027:,.0f} "
            f"(change ${result.predicted_change_dollars:+,.0f})"
        )
    print(f"output_dir: {output_dir}")


if __name__ == "__main__":
    main()
