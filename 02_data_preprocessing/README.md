# 2단계: 데이터 통합 및 전처리

이 폴더에서는 `01_data_collection`의 원본 CSV를 직접 수정하지 않습니다.

## 2-1. Payroll 원본 통합

`01_combine_payroll.py`는 2019~2026년 30팀, 총 240개 Payroll CSV를 하나로 합칩니다.

이 단계에서 하는 작업:

- 240개 팀·시즌 파일의 존재 여부 검사
- 모든 파일의 열 구조 일치 여부 검사
- 파일명과 각 행의 `Team`, `Season` 일치 여부 검사
- 원본 파일 추적을 위한 `_source_file` 열 추가
- 하나의 통합 CSV 생성

아직 하지 않는 작업:

- 중복 선수 처리
- 여러 구단이 나눠 지급한 연봉 합산
- `TBD`, `Pre-ARB`, `MiLB` 처리
- 투수 제외 및 타자 필터링
- Standard·Value 데이터 결합

실행 방법:

```powershell
cd C:\Users\user\Documents\MLB_salary_prediction_v2
& .\prototype\.venv\Scripts\python.exe .\02_data_preprocessing\01_combine_payroll.py
```

현재 PC의 전역 `python` 명령이 연결되어 있지 않아, 기존 가상환경의 Python 실행 파일로 실행합니다. 정식 분석 환경은 다음 세부 단계에서 별도로 구성합니다.

출력 파일:

```text
02_data_preprocessing/interim/payroll_2019_2026_raw_combined.csv
```

## 2-2. 선수·시즌별 계약상 총연봉 정의

본 프로젝트의 연봉은 FanGraphs Payroll에 기록된 해당 시즌 선수별 계약상 급여입니다. 여러 구단이 급여를 분담한 경우 합산하며, AAV·구단 총 Payroll·복리후생·세후 실수령액은 포함하지 않습니다.

`02_build_player_salary.py`의 처리 원칙:

- 동일 선수·시즌·팀·금액이 중복 기재된 경우 한 번만 계산
- 서로 다른 팀이 지급한 숫자형 `Salary`는 합산
- `No Longer On 40-Man`의 숫자형 잔여 급여도 합산
- `TBD`, `Pre-ARB`, `MiLB`는 0으로 바꾸지 않고 `salary_status`에 보존
- 숫자 연봉이 있고 상태값이 없을 때만 `salary_is_complete=1`
- 선수·시즌당 한 행 생성

실행 방법:

```powershell
cd C:\Users\user\Documents\MLB_salary_prediction_v2
& .\prototype\.venv\Scripts\python.exe .\02_data_preprocessing\02_build_player_salary.py
```

출력 파일:

```text
02_data_preprocessing/interim/payroll_player_season_salary_2019_2026.csv
```

## 2-3. Standard·Value 결합 및 타자 표본 플래그

타자는 `player_position != P`이면서 Shohei Ohtani(`playerid=19755`)가 아닌 선수로 정의합니다. 이 단계에서는 행을 삭제하지 않고 모델별 사용 여부만 표시합니다.

- `eligible_all_hitters_pa1`: 전체 타자 모델, PA 1 이상
- `eligible_main_pa50`: 주 모델, PA 50 이상
- `eligible_stable_pa100`: 안정적 표본 모델, PA 100 이상
- `eligible_regular_pa200`: 주전급 타자 모델, PA 200 이상
- `is_partial_2026_snapshot`: 2026년 7월 부분 시즌 표시

실행 방법:

```powershell
cd C:\Users\user\Documents\MLB_salary_prediction_v2
& .\prototype\.venv\Scripts\python.exe .\02_data_preprocessing\03_prepare_batting_features.py
```

출력 파일:

```text
02_data_preprocessing/interim/batting_standard_value_2018_2026_flagged.csv
```

## 2-4. Payroll 현재 시즌 선수 특성

`04_prepare_payroll_features.py`는 Payroll의 Age, Service Time, 현재 계약 그룹과 2-2에서 계산한 현재 시즌 총연봉을 선수·시즌당 한 행으로 정리합니다.

주요 원칙:

- Age의 `#N/A`는 임의 대체하지 않고 결측으로 보존
- Service Time의 `연도.일수` 형식을 연도·일수·총 서비스 일수로 분리
- 계약 그룹을 Guaranteed·Arbitration·Pre-Arb·No Longer On 40-Man 플래그로 변환
- 여러 그룹에 중복 분류된 경우 별도 플래그 생성
- 현재 시즌 총연봉과 다중 구단·잔여 연봉 정보 포함
- `Contract` 문자열과 AAV는 미래 계약 금액 누수를 막기 위해 제외

실행 방법:

```powershell
cd C:\Users\user\Documents\MLB_salary_prediction_v2
& .\prototype\.venv\Scripts\python.exe .\02_data_preprocessing\04_prepare_payroll_features.py
```

출력 파일:

```text
02_data_preprocessing/interim/payroll_player_features_2019_2026.csv
```

## 중간 데이터 감사

`validate_interim.py`는 원본 240개 파일부터 2-4 산출물까지 행 수, 키 중복, 급여 합계 보존, Standard·Value 결합 및 누수 열 제외 여부를 검사합니다.

```powershell
& .\prototype\.venv\Scripts\python.exe .\02_data_preprocessing\validate_interim.py
```

## 2-5. 시점 정렬 및 모델 입력 데이터

`05_build_modeling_dataset.py`는 같은 시즌의 타격·가치·현재 Payroll 특성을 결합하고, 다음 시즌의 계약상 총연봉만 목표값으로 연결합니다.

```text
2019년 특성 → 2020년 연봉
...
2025년 특성 → 2026년 연봉
2026년 특성 → 2027년 예측 행
```

미래 시즌의 계약 그룹, Age, Service Time, AAV 또는 Contract 문자열은 입력 변수로 사용하지 않습니다.

```powershell
& .\prototype\.venv\Scripts\python.exe .\02_data_preprocessing\05_build_modeling_dataset.py
```

출력 파일:

```text
02_data_preprocessing/model_input/modeling_master_hitters_2019_2026.csv
02_data_preprocessing/model_input/training_hitters_2019_2025.csv
02_data_preprocessing/model_input/prediction_hitters_2026_for_2027.csv
```

## 2-6. 모델 입력 열 및 변환 규칙

`06_prepare_model_features.py`는 모델에 전달할 수 있는 열만 별도 파일로 복사하고 PA 기준 타격 비율 및 600 PA 기준 Value 변수를 생성합니다.

```powershell
& .\prototype\.venv\Scripts\python.exe .\02_data_preprocessing\06_prepare_model_features.py
```

출력:

```text
02_data_preprocessing/model_ready/training_model_ready.csv
02_data_preprocessing/model_ready/prediction_2027_model_ready.csv
02_data_preprocessing/model_ready/feature_manifest.json
```

`feature_manifest.json`에는 다음 비교용 입력 세트가 기록됩니다.

- 이전 연봉 기준선
- 타격·가치 비율 모델
- 주 모델: 이전 연봉 + 타격·가치 비율 + Age + Service Time + 현재 계약 상태
- 확장 민감도 모델: 누적 성적과 팀 범주까지 포함

Service Time 결측은 CSV 단계에서 채우지 않습니다. 학습·검증 분할 후 각 학습 fold의 중앙값으로만 대체하여 데이터 누수를 방지합니다.
