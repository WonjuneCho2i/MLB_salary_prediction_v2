"""Modeling 4-3: tune and evaluate Ridge and Elastic Net on temporal folds."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
ELASTIC_ALPHAS = [0.001, 0.01, 0.1, 1.0]
ELASTIC_L1_RATIOS = [0.1, 0.5, 0.9]
LASSO_ALPHAS = [0.001, 0.01, 0.1, 1.0]
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
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        default=project_root / "04_modeling" / "results" / "baseline_summary_metrics.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    return parser.parse_args()


def dollars_from_log(values: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, np.expm1(np.asarray(values, dtype=float)))


def make_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_model(model_family: str, alpha: float, l1_ratio: float | None) -> object:
    if model_family == "ridge":
        return Ridge(alpha=alpha)
    if model_family == "lasso":
        return Lasso(alpha=alpha, max_iter=50_000, tol=1e-5)
    if model_family == "elastic_net":
        return ElasticNet(alpha=alpha, l1_ratio=float(l1_ratio), max_iter=50_000, tol=1e-5)
    raise ValueError(f"Unsupported model family: {model_family}")


def make_pipeline(
    model_family: str,
    alpha: float,
    l1_ratio: float | None,
    numeric: list[str],
    categorical: list[str],
) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", make_preprocessor(numeric, categorical)),
            ("model", make_model(model_family, alpha, l1_ratio)),
        ]
    )


def metric_values(actual_log: np.ndarray, actual_raw: np.ndarray, predicted_log: np.ndarray) -> dict[str, float]:
    predicted_raw = dollars_from_log(predicted_log)
    return {
        "mae_dollars": mean_absolute_error(actual_raw, predicted_raw),
        "rmse_dollars": mean_squared_error(actual_raw, predicted_raw) ** 0.5,
        "median_absolute_error_dollars": median_absolute_error(actual_raw, predicted_raw),
        "rmse_log1p_salary": mean_squared_error(actual_log, predicted_log) ** 0.5,
        "r2_log1p_salary": r2_score(actual_log, predicted_log),
    }


def parameter_grid() -> list[dict[str, object]]:
    candidates = [
        {"model_family": "ridge", "alpha": alpha, "l1_ratio": None}
        for alpha in RIDGE_ALPHAS
    ]
    candidates.extend(
        {"model_family": "lasso", "alpha": alpha, "l1_ratio": 1.0}
        for alpha in LASSO_ALPHAS
    )
    candidates.extend(
        {
            "model_family": "elastic_net",
            "alpha": alpha,
            "l1_ratio": l1_ratio,
        }
        for alpha, l1_ratio in product(ELASTIC_ALPHAS, ELASTIC_L1_RATIOS)
    )
    return candidates


def validate_inputs(
    data: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    config: dict[str, object],
) -> pd.DataFrame:
    required = set(ID_COLUMNS + numeric + categorical + [TARGET_LOG, TARGET_RAW, "train_eligible_main_pa50"])
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    sample = data.loc[data["train_eligible_main_pa50"].eq(1)].copy()
    if sample[[TARGET_LOG, TARGET_RAW]].isna().any().any():
        raise ValueError("Main-sample training targets contain missing values.")
    configured_rows = int(config["final_fit"]["rows"])
    if len(sample) != configured_rows:
        raise ValueError(f"Training sample has {len(sample)} rows, expected {configured_rows} from 4-1.")
    return sample


def tune_candidates(
    sample: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    config: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows: list[dict[str, object]] = []
    feature_columns = numeric + categorical

    for candidate_id, parameters in enumerate(parameter_grid(), start=1):
        for fold_config in config["folds"]:
            fold = int(fold_config["fold"])
            train_start, train_end = fold_config["train_target_years"]
            validation_year = int(fold_config["validation_target_year"])
            train = sample.loc[sample["target_salary_season"].between(train_start, train_end)]
            validation = sample.loc[sample["target_salary_season"].eq(validation_year)]
            if len(train) != int(fold_config["train_rows"]) or len(validation) != int(fold_config["validation_rows"]):
                raise ValueError(f"Fold {fold} row counts do not match the 4-1 configuration.")

            pipeline = make_pipeline(
                str(parameters["model_family"]),
                float(parameters["alpha"]),
                parameters["l1_ratio"],
                numeric,
                categorical,
            )
            pipeline.fit(train[feature_columns], train[TARGET_LOG])
            predicted_log = pipeline.predict(validation[feature_columns])
            metrics = metric_values(
                validation[TARGET_LOG].to_numpy(dtype=float),
                validation[TARGET_RAW].to_numpy(dtype=float),
                predicted_log,
            )
            fold_rows.append(
                {
                    "candidate_id": candidate_id,
                    **parameters,
                    "fold": fold,
                    "validation_target_year": validation_year,
                    "n": len(validation),
                    **metrics,
                }
            )

    fold_metrics = pd.DataFrame(fold_rows)
    summary = (
        fold_metrics.groupby(["candidate_id", "model_family", "alpha", "l1_ratio"], dropna=False)
        .agg(
            fold_count=("fold", "count"),
            fold_mean_mae_dollars=("mae_dollars", "mean"),
            fold_std_mae_dollars=("mae_dollars", "std"),
            fold_mean_rmse_dollars=("rmse_dollars", "mean"),
            fold_mean_median_absolute_error_dollars=("median_absolute_error_dollars", "mean"),
            fold_mean_rmse_log1p_salary=("rmse_log1p_salary", "mean"),
            fold_mean_r2_log1p_salary=("r2_log1p_salary", "mean"),
        )
        .reset_index()
    )
    summary["family_rank_by_mae"] = (
        summary.groupby("model_family")["fold_mean_mae_dollars"].rank(method="min").astype(int)
    )
    summary["overall_rank_by_mae"] = summary["fold_mean_mae_dollars"].rank(method="min").astype(int)
    return fold_metrics, summary.sort_values("overall_rank_by_mae").reset_index(drop=True)


def evaluate_selected(
    sample: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    config: dict[str, object],
    tuning_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected_parameters = tuning_summary.loc[tuning_summary["family_rank_by_mae"].eq(1)].copy()
    selected_parameters = selected_parameters.sort_values("model_family")
    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    coefficient_frames: list[pd.DataFrame] = []
    feature_columns = numeric + categorical

    for parameter_row in selected_parameters.itertuples(index=False):
        model_family = parameter_row.model_family
        l1_ratio = None if pd.isna(parameter_row.l1_ratio) else float(parameter_row.l1_ratio)
        for fold_config in config["folds"]:
            fold = int(fold_config["fold"])
            train_start, train_end = fold_config["train_target_years"]
            validation_year = int(fold_config["validation_target_year"])
            train = sample.loc[sample["target_salary_season"].between(train_start, train_end)]
            validation = sample.loc[sample["target_salary_season"].eq(validation_year)]
            pipeline = make_pipeline(
                model_family,
                float(parameter_row.alpha),
                l1_ratio,
                numeric,
                categorical,
            )
            pipeline.fit(train[feature_columns], train[TARGET_LOG])
            predicted_log = pipeline.predict(validation[feature_columns])
            predicted_raw = dollars_from_log(predicted_log)
            metrics = metric_values(
                validation[TARGET_LOG].to_numpy(dtype=float),
                validation[TARGET_RAW].to_numpy(dtype=float),
                predicted_log,
            )
            metric_rows.append(
                {
                    "model": model_family,
                    "alpha": float(parameter_row.alpha),
                    "l1_ratio": l1_ratio,
                    "fold": fold,
                    "validation_target_year": validation_year,
                    "n": len(validation),
                    **metrics,
                }
            )

            frame = validation[ID_COLUMNS + ["current_contract_stage", TARGET_RAW, TARGET_LOG]].copy()
            frame.insert(0, "model", model_family)
            frame.insert(1, "fold", fold)
            frame["predicted_log1p_salary"] = predicted_log
            frame["predicted_salary_dollars"] = predicted_raw
            frame["absolute_error_dollars"] = np.abs(validation[TARGET_RAW].to_numpy(dtype=float) - predicted_raw)
            prediction_frames.append(frame)

            transformed_names = pipeline.named_steps["preprocess"].get_feature_names_out()
            coefficients = pipeline.named_steps["model"].coef_
            coefficient_frames.append(
                pd.DataFrame(
                    {
                        "model": model_family,
                        "fold": fold,
                        "feature": transformed_names,
                        "coefficient": coefficients,
                    }
                )
            )

    selected_metrics = pd.DataFrame(metric_rows)
    selected_predictions = pd.concat(prediction_frames, ignore_index=True)
    coefficients = pd.concat(coefficient_frames, ignore_index=True)
    return selected_metrics, selected_predictions, coefficients


def summarize_selected(metrics: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_name, model_metrics in metrics.groupby("model"):
        model_predictions = predictions.loc[predictions["model"].eq(model_name)]
        combined = metric_values(
            model_predictions[TARGET_LOG].to_numpy(dtype=float),
            model_predictions[TARGET_RAW].to_numpy(dtype=float),
            model_predictions["predicted_log1p_salary"].to_numpy(dtype=float),
        )
        first = model_metrics.iloc[0]
        rows.append(
            {
                "model": model_name,
                "alpha": first["alpha"],
                "l1_ratio": first["l1_ratio"],
                "fold_count": len(model_metrics),
                "oof_rows": len(model_predictions),
                "fold_mean_mae_dollars": model_metrics["mae_dollars"].mean(),
                "fold_std_mae_dollars": model_metrics["mae_dollars"].std(ddof=1),
                "combined_oof_mae_dollars": combined["mae_dollars"],
                "combined_oof_rmse_dollars": combined["rmse_dollars"],
                "combined_oof_median_absolute_error_dollars": combined["median_absolute_error_dollars"],
                "combined_oof_rmse_log1p_salary": combined["rmse_log1p_salary"],
                "combined_oof_r2_log1p_salary": combined["r2_log1p_salary"],
            }
        )
    result = pd.DataFrame(rows)
    result["rank_by_fold_mean_mae"] = result["fold_mean_mae_dollars"].rank(method="min").astype(int)
    return result.sort_values("rank_by_fold_mean_mae").reset_index(drop=True)


def compare_with_baseline(selected: pd.DataFrame, baseline_path: Path) -> pd.DataFrame:
    baseline = pd.read_csv(baseline_path, encoding="utf-8-sig")
    baseline = baseline[["model", "fold_mean_mae_dollars", "combined_oof_rmse_dollars", "combined_oof_r2_log1p_salary"]]
    selected_comparison = selected[
        ["model", "fold_mean_mae_dollars", "combined_oof_rmse_dollars", "combined_oof_r2_log1p_salary"]
    ]
    comparison = pd.concat([baseline, selected_comparison], ignore_index=True)
    naive_mae = float(
        comparison.loc[comparison["model"].eq("naive_current_salary"), "fold_mean_mae_dollars"].iloc[0]
    )
    comparison["mae_improvement_vs_naive_dollars"] = naive_mae - comparison["fold_mean_mae_dollars"]
    comparison["mae_improvement_vs_naive_percent"] = (
        comparison["mae_improvement_vs_naive_dollars"] / naive_mae * 100
    )
    comparison["rank_by_fold_mean_mae"] = comparison["fold_mean_mae_dollars"].rank(method="min").astype(int)
    return comparison.sort_values("rank_by_fold_mean_mae").reset_index(drop=True)


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
    feature_set = manifest["feature_sets"]["linear_reduced_main"]
    numeric = feature_set["numeric"]
    categorical = feature_set["categorical"]
    sample = validate_inputs(data, numeric, categorical, config)

    candidate_metrics, tuning_summary = tune_candidates(sample, numeric, categorical, config)
    selected_metrics, selected_predictions, coefficients = evaluate_selected(
        sample, numeric, categorical, config, tuning_summary
    )
    selected_summary = summarize_selected(selected_metrics, selected_predictions)
    comparison = compare_with_baseline(selected_summary, args.baseline_summary.resolve())

    candidate_metrics.to_csv(result_dir / "regularized_candidate_fold_metrics.csv", index=False, encoding="utf-8-sig")
    tuning_summary.to_csv(result_dir / "regularized_hyperparameter_summary.csv", index=False, encoding="utf-8-sig")
    selected_metrics.to_csv(result_dir / "regularized_selected_fold_metrics.csv", index=False, encoding="utf-8-sig")
    selected_summary.to_csv(result_dir / "regularized_selected_summary.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(result_dir / "linear_vs_baseline_comparison.csv", index=False, encoding="utf-8-sig")
    coefficients.to_csv(result_dir / "regularized_coefficients_by_fold.csv", index=False, encoding="utf-8-sig")
    selected_predictions.to_csv(
        prediction_dir / "regularized_linear_out_of_fold_predictions.csv", index=False, encoding="utf-8-sig"
    )

    print("Regularized linear temporal validation complete")
    print(f"numeric_features: {len(numeric)}")
    print(f"categorical_features: {len(categorical)}")
    print(f"candidate_count: {tuning_summary.shape[0]}")
    for row in selected_summary.itertuples(index=False):
        ratio = "NA" if pd.isna(row.l1_ratio) else f"{row.l1_ratio:g}"
        print(
            f"  {row.model}: alpha={row.alpha:g}, l1_ratio={ratio}, "
            f"fold mean MAE=${row.fold_mean_mae_dollars:,.0f}, "
            f"OOF log R2={row.combined_oof_r2_log1p_salary:.4f}"
        )
    print("comparison ranking:")
    for row in comparison.itertuples(index=False):
        print(
            f"  rank {row.rank_by_fold_mean_mae}: {row.model}, "
            f"MAE=${row.fold_mean_mae_dollars:,.0f}, "
            f"vs naive={row.mae_improvement_vs_naive_percent:+.2f}%"
        )
    print(f"output_dir: {output_dir}")


if __name__ == "__main__":
    main()
