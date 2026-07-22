"""Modeling 4-4: tune tree ensembles with direct and residual salary targets."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor


TARGET_MODES = ["direct_log_salary", "log_change_from_current"]
TARGET_LOG = "log1p_target_salary"
TARGET_RAW = "target_salary_total"
CURRENT_LOG = "log1p_current_salary"
ID_COLUMNS = ["Season", "target_salary_season", "Player", "playerid"]
RANDOM_STATE = 42


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
        "--previous-comparison",
        type=Path,
        default=project_root / "04_modeling" / "results" / "linear_vs_baseline_comparison.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    return parser.parse_args()


def parameter_grid() -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for family in ["random_forest", "extra_trees"]:
        for target_mode, max_depth, min_samples_leaf in product(
            TARGET_MODES, [None, 10], [1, 5]
        ):
            candidates.append(
                {
                    "model_family": family,
                    "target_mode": target_mode,
                    "n_estimators": 350,
                    "max_depth": max_depth,
                    "min_samples_leaf": min_samples_leaf,
                    "max_features": 0.8,
                    "learning_rate": None,
                    "loss": None,
                }
            )
    for target_mode, n_estimators, learning_rate, max_depth in product(
        TARGET_MODES, [150, 300], [0.03, 0.07], [2, 3]
    ):
        candidates.append(
            {
                "model_family": "gradient_boosting",
                "target_mode": target_mode,
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_leaf": 5,
                "max_features": None,
                "learning_rate": learning_rate,
                "loss": "huber",
            }
        )
    for target_mode, n_estimators, learning_rate, max_depth in product(
        TARGET_MODES, [250, 500], [0.03, 0.07], [2, 3]
    ):
        candidates.append(
            {
                "model_family": "xgboost",
                "target_mode": target_mode,
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_leaf": 5,
                "max_features": 0.8,
                "learning_rate": learning_rate,
                "loss": "squared_error",
                "num_leaves": None,
            }
        )
    for target_mode, n_estimators, learning_rate, num_leaves in product(
        TARGET_MODES, [250, 500], [0.03, 0.07], [7, 15]
    ):
        candidates.append(
            {
                "model_family": "lightgbm",
                "target_mode": target_mode,
                "n_estimators": n_estimators,
                "max_depth": -1,
                "min_samples_leaf": 10,
                "max_features": 0.8,
                "learning_rate": learning_rate,
                "loss": "squared_error",
                "num_leaves": num_leaves,
            }
        )
    for candidate_id, candidate in enumerate(candidates, start=1):
        candidate["candidate_id"] = candidate_id
    return candidates


def make_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
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


def make_estimator(parameters: dict[str, object]) -> object:
    family = parameters["model_family"]
    if family == "random_forest":
        return RandomForestRegressor(
            n_estimators=int(parameters["n_estimators"]),
            max_depth=parameters["max_depth"],
            min_samples_leaf=int(parameters["min_samples_leaf"]),
            max_features=float(parameters["max_features"]),
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    if family == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=int(parameters["n_estimators"]),
            max_depth=parameters["max_depth"],
            min_samples_leaf=int(parameters["min_samples_leaf"]),
            max_features=float(parameters["max_features"]),
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    if family == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=int(parameters["n_estimators"]),
            learning_rate=float(parameters["learning_rate"]),
            max_depth=int(parameters["max_depth"]),
            min_samples_leaf=int(parameters["min_samples_leaf"]),
            loss=str(parameters["loss"]),
            random_state=RANDOM_STATE,
        )
    if family == "xgboost":
        return XGBRegressor(
            n_estimators=int(parameters["n_estimators"]),
            learning_rate=float(parameters["learning_rate"]),
            max_depth=int(parameters["max_depth"]),
            min_child_weight=float(parameters["min_samples_leaf"]),
            subsample=0.8,
            colsample_bytree=float(parameters["max_features"]),
            reg_lambda=1.0,
            objective="reg:squarederror",
            eval_metric="rmse",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )
    if family == "lightgbm":
        return LGBMRegressor(
            n_estimators=int(parameters["n_estimators"]),
            learning_rate=float(parameters["learning_rate"]),
            num_leaves=int(parameters["num_leaves"]),
            max_depth=int(parameters["max_depth"]),
            min_child_samples=int(parameters["min_samples_leaf"]),
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=float(parameters["max_features"]),
            reg_lambda=1.0,
            objective="regression",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        )
    raise ValueError(f"Unsupported model family: {family}")


def make_pipeline(parameters: dict[str, object], numeric: list[str], categorical: list[str]) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", make_preprocessor(numeric, categorical)),
            ("model", make_estimator(parameters)),
        ]
    )


def training_target(data: pd.DataFrame, target_mode: str) -> np.ndarray:
    if target_mode == "direct_log_salary":
        return data[TARGET_LOG].to_numpy(dtype=float)
    if target_mode == "log_change_from_current":
        return (data[TARGET_LOG] - data[CURRENT_LOG]).to_numpy(dtype=float)
    raise ValueError(f"Unsupported target mode: {target_mode}")


def reconstruct_log_prediction(raw_prediction: np.ndarray, data: pd.DataFrame, target_mode: str) -> np.ndarray:
    if target_mode == "direct_log_salary":
        return np.asarray(raw_prediction, dtype=float)
    if target_mode == "log_change_from_current":
        return data[CURRENT_LOG].to_numpy(dtype=float) + np.asarray(raw_prediction, dtype=float)
    raise ValueError(f"Unsupported target mode: {target_mode}")


def dollars_from_log(values: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, np.expm1(np.asarray(values, dtype=float)))


def metric_values(actual_log: np.ndarray, actual_raw: np.ndarray, predicted_log: np.ndarray) -> dict[str, float]:
    predicted_raw = dollars_from_log(predicted_log)
    return {
        "mae_dollars": mean_absolute_error(actual_raw, predicted_raw),
        "rmse_dollars": mean_squared_error(actual_raw, predicted_raw) ** 0.5,
        "median_absolute_error_dollars": median_absolute_error(actual_raw, predicted_raw),
        "rmse_log1p_salary": mean_squared_error(actual_log, predicted_log) ** 0.5,
        "r2_log1p_salary": r2_score(actual_log, predicted_log),
    }


def validate_inputs(
    data: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    config: dict[str, object],
) -> pd.DataFrame:
    required = set(ID_COLUMNS + numeric + categorical + [TARGET_LOG, TARGET_RAW, CURRENT_LOG, "train_eligible_main_pa50"])
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    sample = data.loc[data["train_eligible_main_pa50"].eq(1)].copy()
    if sample[[TARGET_LOG, TARGET_RAW, CURRENT_LOG]].isna().any().any():
        raise ValueError("Main-sample salary fields contain missing values.")
    if len(sample) != int(config["final_fit"]["rows"]):
        raise ValueError("Training row count does not match the 4-1 temporal configuration.")
    return sample


def tune_candidates(
    sample: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    config: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    candidates = parameter_grid()
    feature_columns = numeric + categorical
    rows: list[dict[str, object]] = []

    for candidate in candidates:
        for fold_config in config["folds"]:
            fold = int(fold_config["fold"])
            train_start, train_end = fold_config["train_target_years"]
            validation_year = int(fold_config["validation_target_year"])
            train = sample.loc[sample["target_salary_season"].between(train_start, train_end)]
            validation = sample.loc[sample["target_salary_season"].eq(validation_year)]
            if len(train) != int(fold_config["train_rows"]) or len(validation) != int(fold_config["validation_rows"]):
                raise ValueError(f"Fold {fold} row counts do not match the 4-1 configuration.")

            pipeline = make_pipeline(candidate, numeric, categorical)
            pipeline.fit(train[feature_columns], training_target(train, str(candidate["target_mode"])))
            raw_prediction = pipeline.predict(validation[feature_columns])
            predicted_log = reconstruct_log_prediction(raw_prediction, validation, str(candidate["target_mode"]))
            metrics = metric_values(
                validation[TARGET_LOG].to_numpy(dtype=float),
                validation[TARGET_RAW].to_numpy(dtype=float),
                predicted_log,
            )
            rows.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "model_family": candidate["model_family"],
                    "target_mode": candidate["target_mode"],
                    "fold": fold,
                    "validation_target_year": validation_year,
                    "n": len(validation),
                    **metrics,
                }
            )

    fold_metrics = pd.DataFrame(rows)
    summary = (
        fold_metrics.groupby(["candidate_id", "model_family", "target_mode"])
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
    parameters = pd.DataFrame(candidates)
    summary = summary.merge(parameters, on=["candidate_id", "model_family", "target_mode"], how="left")
    summary["family_rank_by_mae"] = (
        summary.groupby("model_family")["fold_mean_mae_dollars"].rank(method="min").astype(int)
    )
    summary["overall_rank_by_mae"] = summary["fold_mean_mae_dollars"].rank(method="min").astype(int)
    return fold_metrics, summary.sort_values("overall_rank_by_mae").reset_index(drop=True), candidates


def evaluate_selected(
    sample: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    config: dict[str, object],
    tuning_summary: pd.DataFrame,
    candidates: list[dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidate_lookup = {int(candidate["candidate_id"]): candidate for candidate in candidates}
    selected = tuning_summary.loc[tuning_summary["family_rank_by_mae"].eq(1)].copy()
    feature_columns = numeric + categorical
    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []

    for selected_row in selected.itertuples(index=False):
        candidate = candidate_lookup[int(selected_row.candidate_id)]
        model_label = f"{candidate['model_family']}__{candidate['target_mode']}"
        for fold_config in config["folds"]:
            fold = int(fold_config["fold"])
            train_start, train_end = fold_config["train_target_years"]
            validation_year = int(fold_config["validation_target_year"])
            train = sample.loc[sample["target_salary_season"].between(train_start, train_end)]
            validation = sample.loc[sample["target_salary_season"].eq(validation_year)]

            pipeline = make_pipeline(candidate, numeric, categorical)
            pipeline.fit(train[feature_columns], training_target(train, str(candidate["target_mode"])))
            raw_prediction = pipeline.predict(validation[feature_columns])
            predicted_log = reconstruct_log_prediction(raw_prediction, validation, str(candidate["target_mode"]))
            predicted_raw = dollars_from_log(predicted_log)
            metrics = metric_values(
                validation[TARGET_LOG].to_numpy(dtype=float),
                validation[TARGET_RAW].to_numpy(dtype=float),
                predicted_log,
            )
            metric_rows.append(
                {
                    "model": model_label,
                    "model_family": candidate["model_family"],
                    "target_mode": candidate["target_mode"],
                    "candidate_id": candidate["candidate_id"],
                    "fold": fold,
                    "validation_target_year": validation_year,
                    "n": len(validation),
                    **metrics,
                }
            )

            frame = validation[ID_COLUMNS + ["current_contract_stage", TARGET_RAW, TARGET_LOG]].copy()
            frame.insert(0, "model", model_label)
            frame.insert(1, "fold", fold)
            frame["predicted_log1p_salary"] = predicted_log
            frame["predicted_salary_dollars"] = predicted_raw
            frame["absolute_error_dollars"] = np.abs(validation[TARGET_RAW].to_numpy(dtype=float) - predicted_raw)
            prediction_frames.append(frame)

            transformed_names = pipeline.named_steps["preprocess"].get_feature_names_out()
            importances = pipeline.named_steps["model"].feature_importances_
            importance_frames.append(
                pd.DataFrame(
                    {
                        "model": model_label,
                        "fold": fold,
                        "feature": transformed_names,
                        "importance": importances,
                    }
                )
            )

    return (
        pd.DataFrame(metric_rows),
        pd.concat(prediction_frames, ignore_index=True),
        pd.concat(importance_frames, ignore_index=True),
    )


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
                "model_family": first["model_family"],
                "target_mode": first["target_mode"],
                "candidate_id": int(first["candidate_id"]),
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


def compare_all(selected: pd.DataFrame, previous_path: Path) -> pd.DataFrame:
    previous = pd.read_csv(previous_path, encoding="utf-8-sig")
    previous = previous[["model", "fold_mean_mae_dollars", "combined_oof_rmse_dollars", "combined_oof_r2_log1p_salary"]]
    tree = selected[["model", "fold_mean_mae_dollars", "combined_oof_rmse_dollars", "combined_oof_r2_log1p_salary"]]
    comparison = pd.concat([previous, tree], ignore_index=True)
    naive_mae = float(
        comparison.loc[comparison["model"].eq("naive_current_salary"), "fold_mean_mae_dollars"].iloc[0]
    )
    comparison["mae_improvement_vs_naive_dollars"] = naive_mae - comparison["fold_mean_mae_dollars"]
    comparison["mae_improvement_vs_naive_percent"] = comparison["mae_improvement_vs_naive_dollars"] / naive_mae * 100
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
    feature_set = manifest["feature_sets"]["tree_reduced_main"]
    numeric = feature_set["numeric"]
    categorical = feature_set["categorical"]
    sample = validate_inputs(data, numeric, categorical, config)

    candidate_metrics, tuning_summary, candidates = tune_candidates(sample, numeric, categorical, config)
    selected_metrics, selected_predictions, importances = evaluate_selected(
        sample, numeric, categorical, config, tuning_summary, candidates
    )
    selected_summary = summarize_selected(selected_metrics, selected_predictions)
    comparison = compare_all(selected_summary, args.previous_comparison.resolve())

    candidate_metrics.to_csv(result_dir / "tree_candidate_fold_metrics.csv", index=False, encoding="utf-8-sig")
    tuning_summary.to_csv(result_dir / "tree_hyperparameter_summary.csv", index=False, encoding="utf-8-sig")
    selected_metrics.to_csv(result_dir / "tree_selected_fold_metrics.csv", index=False, encoding="utf-8-sig")
    selected_summary.to_csv(result_dir / "tree_selected_summary.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(result_dir / "all_model_comparison_through_4_4.csv", index=False, encoding="utf-8-sig")
    importances.to_csv(result_dir / "tree_feature_importance_by_fold.csv", index=False, encoding="utf-8-sig")
    selected_predictions.to_csv(
        prediction_dir / "tree_out_of_fold_predictions.csv", index=False, encoding="utf-8-sig"
    )

    print("Tree-model temporal validation complete")
    print(f"numeric_features: {len(numeric)}")
    print(f"categorical_features: {len(categorical)}")
    print(f"candidate_count: {len(candidates)}")
    for row in selected_summary.itertuples(index=False):
        print(
            f"  {row.model}: fold mean MAE=${row.fold_mean_mae_dollars:,.0f}, "
            f"OOF RMSE=${row.combined_oof_rmse_dollars:,.0f}, "
            f"OOF log R2={row.combined_oof_r2_log1p_salary:.4f}"
        )
    print("top overall comparison:")
    for row in comparison.head(6).itertuples(index=False):
        print(
            f"  rank {row.rank_by_fold_mean_mae}: {row.model}, "
            f"MAE=${row.fold_mean_mae_dollars:,.0f}, "
            f"vs naive={row.mae_improvement_vs_naive_percent:+.2f}%"
        )
    print(f"output_dir: {output_dir}")


if __name__ == "__main__":
    main()
