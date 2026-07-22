# 3단계: 시각화 및 탐색적 데이터 분석

모든 분석은 주 모델 표본인 `PA >= 50` 선수·시즌을 기준으로 합니다.

## 3-1. 연봉 분포

```powershell
& .\prototype\.venv\Scripts\python.exe .\03_eda\01_salary_distribution.py
```

산출물:

```text
03_eda/figures/01_salary_raw_vs_log.png
03_eda/figures/02_salary_by_year.png
03_eda/tables/salary_distribution_summary.csv
03_eda/tables/salary_by_year_summary.csv
03_eda/tables/zero_salary_cases.csv
```

- 실제 연봉과 `log1p` 연봉 분포를 비교합니다.
- 목표 연도별 연봉 분포와 중앙값을 실제 백만 달러 단위로 비교합니다.
- 연봉 0인 사례는 별도 표로 보존하고 양수 연봉 로그 분포에서는 제외합니다.

## 3-2. 계약 단계 및 현재·다음 연봉 관계

```powershell
& .\prototype\.venv\Scripts\python.exe .\03_eda\02_salary_relationships.py
```

산출물:

```text
03_eda/figures/03_salary_by_contract_stage.png
03_eda/figures/04_current_vs_next_salary.png
03_eda/tables/salary_by_contract_stage.csv
03_eda/tables/current_next_salary_correlations.csv
```

- 현재 계약 단계별 다음 시즌 연봉 분포를 백만 달러 단위로 비교합니다.
- 현재·다음 연봉 관계는 양수 연봉 표본에서 전체 및 계약 단계별로 계산합니다.

## 3-3. 타격 성적과 다음 시즌 연봉 관계

```powershell
& .\prototype\.venv\Scripts\python.exe .\03_eda\03_performance_relationships.py
```

산출물:

```text
03_eda/figures/05_salary_vs_performance.png
03_eda/tables/performance_salary_correlations.csv
```

- 출전량, 타율, 홈런 비율, 타격 기여도 및 WAR와 다음 시즌 연봉의 관계를 비교합니다.
- y축 계산에는 로그 연봉을 사용하지만 눈금은 백만 달러 단위로 표시합니다.
- 상관관계는 탐색적 관계이며 인과효과로 해석하지 않습니다.

## 3-4. 선수 나이·서비스 타임과 다음 시즌 연봉

```powershell
& .\prototype\.venv\Scripts\python.exe .\03_eda\04_player_characteristics.py
```

산출물:

```text
03_eda/figures/06_salary_vs_age_service.png
03_eda/figures/07_salary_by_age_service_group.png
03_eda/tables/salary_by_age_service_group.csv
03_eda/tables/age_service_salary_correlations.csv
03_eda/tables/missing_service_time_cases.csv
```

- 나이와 MLB 서비스 타임의 연속적인 관계를 로그 연봉 축에서 확인합니다.
- 나이 구간과 노사제도상 핵심 서비스 타임 구간별 실제 연봉 분포를 비교합니다.
- 서비스 타임 결측은 임의로 채우지 않고 별도 집단과 별도 표로 보존합니다.
- 계약 단계별 상관계수도 표로 저장하여 계약 상태의 교란 가능성을 확인합니다.

## 3-5. 변수 상관관계 및 다중공선성

```powershell
& .\prototype\.venv\Scripts\python.exe .\03_eda\05_feature_correlation.py
```

산출물:

```text
03_eda/figures/08_selected_feature_correlation.png
03_eda/figures/09_top_collinear_feature_pairs.png
03_eda/tables/selected_feature_spearman.csv
03_eda/tables/full_main_numeric_correlation_pairs.csv
03_eda/tables/high_correlation_pairs.csv
03_eda/tables/numeric_feature_vif.csv
03_eda/05_feature_correlation_notes.md
```

- 주 모델의 대표 변수는 읽을 수 있는 크기의 Spearman 상관 히트맵으로 확인합니다.
- 주 모델 숫자형 변수 전체에서 강한 변수쌍을 별도로 추출합니다.
- VIF는 진단 목적으로만 중앙값 대체 후 계산하며 실제 모델 대체기는 학습 fold 내부에서 적합합니다.
- 중복 변수를 곧바로 삭제하지 않고 축소형 선형 모델과 규제·트리 모델 비교 근거로 사용합니다.

## 3-6. 최종 변수 세트 확정

```powershell
& .\prototype\.venv\Scripts\python.exe .\03_eda\06_finalize_feature_sets.py
```

산출물:

```text
02_data_preprocessing/model_ready/feature_manifest_final.json
03_eda/tables/final_feature_set_summary.csv
03_eda/tables/feature_selection_decisions.csv
03_eda/tables/linear_reduced_vif.csv
03_eda/06_final_feature_selection_notes.md
```

- 원본 model-ready CSV의 열은 삭제하지 않고 최종 manifest에서 모델별 입력 열을 통제합니다.
- 선형 모델은 중복을 줄인 해석 가능한 변수 세트, 트리 모델은 더 넓은 snapshot-safe 변수 세트를 사용합니다.
- 전년도 연봉 기준 모델과 성적만 사용하는 ablation 모델을 별도로 유지합니다.
- 전체 변수와 누적 성적 변수는 민감도 분석용으로만 보존합니다.
