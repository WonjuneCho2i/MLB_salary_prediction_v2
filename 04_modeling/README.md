# 4단계: 모델 학습 및 검증

## 4-1. 시간순 학습·검증 분할

```powershell
& .\prototype\.venv\Scripts\python.exe .\04_modeling\01_build_temporal_splits.py
```

목표 연도 기준 expanding-window 검증을 사용합니다.

```text
Fold 1: 2020-2022 연봉 학습 -> 2023 연봉 검증
Fold 2: 2020-2023 연봉 학습 -> 2024 연봉 검증
Fold 3: 2020-2024 연봉 학습 -> 2025 연봉 검증
Fold 4: 2020-2025 연봉 학습 -> 2026 연봉 검증
```

모델 선택이 끝나면 2020-2026 연봉 자료 전체로 다시 학습하고, 2026년 특성으로 2027년 연봉을 예측합니다.

산출물:

```text
04_modeling/config/temporal_validation.json
04_modeling/splits/temporal_fold_assignments.csv
04_modeling/splits/temporal_split_summary.csv
04_modeling/splits/final_training_rows.csv
04_modeling/splits/prediction_2027_rows.csv
04_modeling/01_temporal_validation_notes.md
```

- 랜덤 분할은 사용하지 않습니다.
- 전처리기는 각 fold의 학습 데이터에서만 적합합니다.
- 같은 선수의 과거 시즌이 학습 데이터에 존재하는 것은 실제 예측 상황과 일치하므로 허용합니다.
- 2026년 특성 데이터는 최종 모델을 선택하기 전까지 2027년 예측 전용으로 보존합니다.

## 4-2. 기준 모델 평가

```powershell
& .\prototype\.venv\Scripts\python.exe .\04_modeling\02_evaluate_baselines.py
```

동일한 시간순 Fold에서 다음 세 가지 기준을 비교합니다.

- `naive_current_salary`: 다음 시즌 연봉이 현재 연봉과 같다고 예측
- `dummy_median_log`: 해당 Fold 학습 자료의 로그 연봉 중앙값으로 예측
- `linear_previous_salary`: 목표 연도와 현재 로그 연봉을 이용한 선형회귀

산출물:

```text
04_modeling/results/baseline_fold_metrics.csv
04_modeling/results/baseline_summary_metrics.csv
04_modeling/predictions/baseline_out_of_fold_predictions.csv
```

- 모델 순위는 네 Fold의 실제 달러 MAE 평균을 기준으로 정합니다.
- 로그 예측값은 `expm1`으로 달러로 환산하고 음수 예측은 0으로 제한합니다.
- 아직 2027년 예측 데이터는 사용하지 않습니다.

## 4-6. 전체 재학습 및 2027년 연봉 예측

```powershell
& .\prototype\.venv\Scripts\python.exe .\04_modeling\06_train_final_predict_2027.py
```

최종 Gradient Boosting 변화량 모델을 2020-2026 연봉 전체 자료로 재학습하고, 2026년 특성으로 2027년 연봉을 예측합니다.

산출물:

```text
04_modeling/models/final_gradient_boosting_2027.joblib
04_modeling/config/final_model_2027_metadata.json
04_modeling/predictions/mlb_hitters_salary_prediction_2027.csv
04_modeling/predictions/top_50_salary_predictions_2027.csv
04_modeling/predictions/top_25_predicted_salary_increases_2027.csv
04_modeling/predictions/top_25_predicted_salary_decreases_2027.csv
04_modeling/results/prediction_2027_summary.csv
04_modeling/results/prediction_2027_by_contract_stage.csv
04_modeling/results/final_refit_feature_importance.csv
```

- 예측치는 현재 연봉 대비 로그 변화량을 현재 로그 연봉에 다시 더해 달러로 복원합니다.
- OOF 로그 오차의 80% 분위수를 이용한 경험적 범위를 함께 제공합니다.
- 경험적 범위는 계약상 보장 구간이나 엄밀한 보장 확률 구간이 아닙니다.
- 2026년 자료가 부분 시즌이라는 점을 모든 결과에 표시합니다.

## 4-5. 최종 모델 진단 및 선정

```powershell
& .\prototype\.venv\Scripts\python.exe .\04_modeling\05_select_final_model.py
```

모든 모델의 시간순 OOF 성능을 비교하고, 1위 모델의 연도·계약 단계·실제 연봉 구간별 오차와 변수 중요도를 진단합니다.

산출물:

```text
04_modeling/config/final_model_selection.json
04_modeling/figures/10_all_model_mae_comparison.png
04_modeling/figures/11_final_model_diagnostics.png
04_modeling/figures/12_final_model_feature_importance.png
04_modeling/results/final_model_ranking.csv
04_modeling/results/final_model_error_by_year.csv
04_modeling/results/final_model_error_by_contract_stage.csv
04_modeling/results/final_model_error_by_salary_band.csv
04_modeling/results/final_vs_naive_by_year.csv
04_modeling/results/final_model_feature_importance.csv
04_modeling/results/final_model_largest_errors.csv
```

- 최종 선택 기준은 네 Fold 실제 달러 MAE 평균입니다.
- 기준 모델을 모든 검증 연도에서 이기는지도 확인합니다.
- 변수 중요도는 예측 기여도이며 인과효과로 해석하지 않습니다.
- 아직 2027년 예측 데이터는 사용하지 않습니다.

## 4-4. 트리 앙상블 모델

```powershell
& .\prototype\.venv\Scripts\python.exe .\04_modeling\04_evaluate_tree_models.py
```

`tree_reduced_main` 변수 세트로 다음 모델을 비교합니다.

- Random Forest
- Extra Trees
- Gradient Boosting
- XGBoost
- LightGBM

각 모델은 `log1p(next salary)` 직접 예측과 `log1p(next salary) - log1p(current salary)` 변화량 예측을 모두 시험합니다. 하이퍼파라미터는 네 시간 Fold의 실제 달러 MAE 평균으로 선택합니다.

산출물:

```text
04_modeling/results/tree_candidate_fold_metrics.csv
04_modeling/results/tree_hyperparameter_summary.csv
04_modeling/results/tree_selected_fold_metrics.csv
04_modeling/results/tree_selected_summary.csv
04_modeling/results/all_model_comparison_through_4_4.csv
04_modeling/results/tree_feature_importance_by_fold.csv
04_modeling/predictions/tree_out_of_fold_predictions.csv
```

- 모든 결측치 처리와 범주형 인코딩은 각 Fold 학습 데이터에서만 적합합니다.
- `random_state=42`를 고정해 재현성을 확보합니다.
- 아직 2027년 예측 데이터는 사용하지 않습니다.

## 4-3. 규제 선형 모델

```powershell
& .\prototype\.venv\Scripts\python.exe .\04_modeling\03_evaluate_regularized_linear.py
```

최종 `linear_reduced_main` 변수 세트로 Ridge, Lasso, Elastic Net을 비교합니다.

- Ridge alpha: `0.01, 0.1, 1, 10, 100, 1000`
- Lasso alpha: `0.001, 0.01, 0.1, 1`
- Elastic Net alpha: `0.001, 0.01, 0.1, 1`
- Elastic Net l1 ratio: `0.1, 0.5, 0.9`
- 선택 기준: 네 시간 Fold의 실제 달러 MAE 평균

모든 결측치 처리, 표준화, 원-핫 인코딩은 각 Fold 학습 데이터에서만 적합합니다.

산출물:

```text
04_modeling/results/regularized_candidate_fold_metrics.csv
04_modeling/results/regularized_hyperparameter_summary.csv
04_modeling/results/regularized_selected_fold_metrics.csv
04_modeling/results/regularized_selected_summary.csv
04_modeling/results/linear_vs_baseline_comparison.csv
04_modeling/results/regularized_coefficients_by_fold.csv
04_modeling/predictions/regularized_linear_out_of_fold_predictions.csv
```

- 최적 Ridge와 최적 Elastic Net을 각각 보존합니다.
- 기준 모델과의 개선 폭을 달러 및 백분율로 계산합니다.
- 아직 2027년 예측 데이터는 사용하지 않습니다.
