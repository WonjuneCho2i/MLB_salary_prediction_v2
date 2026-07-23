# 02_data_preprocessing 단계 정리

## 1. 이 단계의 목적

이 단계의 목적은 **MLB 선수의 현재 시즌 정보로 다음 시즌 연봉을 예측할 수 있도록, 원본 데이터를 모델 학습용 형태로 정리하는 것**이다.

구체적으로 다음 데이터를 하나의 선수·시즌 단위 데이터셋으로 결합한다.

- FanGraphs Payroll 데이터
- FanGraphs Standard 타격 데이터
- FanGraphs Value 데이터
- 선수의 현재 연봉, 계약 상태, 나이, 서비스 타임
- 다음 시즌 연봉 목표값

최종적으로는 다음과 같은 시간 구조를 가진 데이터를 만든다.

```text
현재 시즌의 성적·나이·계약·연봉
        ↓
다음 시즌 연봉 예측
```

예시는 다음과 같다.

```text
2019년 정보 → 2020년 연봉
2020년 정보 → 2021년 연봉
...
2025년 정보 → 2026년 연봉
2026년 정보 → 2027년 연봉 예측
```

따라서 이 단계는 모델을 직접 학습하는 단계가 아니라, **모델이 바로 사용할 수 있는 학습 데이터와 예측 데이터를 만드는 전처리 단계**이다.

---

## 2. 수행한 데이터 전처리

전체 전처리는 `01`부터 `06`까지 순서대로 진행된다.

### Code 1. `01_combine_payroll.py`

#### 역할

2019년부터 2026년까지의 팀별 FanGraphs Payroll CSV를 하나의 파일로 통합한다.

#### 입력 데이터

```text
01_data_collection/downloads/payroll/
```

예시 파일:

```text
fangraphs_payroll_ARI_2019.csv
fangraphs_payroll_NYY_2020.csv
fangraphs_payroll_LAD_2026.csv
```

총 30개 팀 × 8개 시즌으로, 240개 파일을 대상으로 한다.

#### 주요 처리

- 팀·시즌별 Payroll CSV 파일 탐색
- 총 240개 파일이 모두 존재하는지 확인
- 파일명에서 팀과 시즌 추출
- 파일명과 CSV 내부 `Team`, `Season` 값 일치 여부 확인
- CSV별 열 구조가 같은지 확인
- `playerid` 누락 여부 확인
- 모든 파일을 하나의 CSV로 통합
- 각 행이 어느 원본 파일에서 왔는지 `_source_file` 열 추가

#### 출력

```text
02_data_preprocessing/interim/
payroll_2019_2026_raw_combined.csv
```

---

### Code 2. `02_build_player_salary.py`

#### 역할

통합된 Payroll 데이터를 선수·시즌 단위로 집계하여, 한 선수의 해당 시즌 총연봉을 계산한다.

#### 입력 데이터

```text
02_data_preprocessing/interim/
payroll_2019_2026_raw_combined.csv
```

#### 주요 처리

- `Season + playerid` 기준으로 선수별 데이터 그룹화
- 여러 팀이 한 선수의 연봉을 나누어 지급한 경우 합산
- 같은 팀·같은 금액이 중복 기재된 경우 중복 제거
- 숫자 연봉과 `TBD`, `Pre-ARB`, `MiLB` 등의 비숫자 상태 구분
- retained salary 별도 계산
- 연봉이 완전히 확정된 데이터인지 판단
- 지급 팀 수와 Payroll에 표시된 팀 수 계산
- 선수별 연봉 구성요소 수 계산

#### 주요 생성 변수

```text
season_salary_total
salary_is_complete
salary_status
paying_teams
paying_team_count
retained_salary_total
has_retained_salary
contract_groups
```

#### 출력

```text
02_data_preprocessing/interim/
payroll_player_season_salary_2019_2026.csv
```

---

### Code 3. `03_prepare_batting_features.py`

#### 역할

FanGraphs Standard 타격 데이터와 Value 데이터를 결합하고, 타자 학습 표본을 구분하는 플래그를 생성한다.

#### 입력 데이터

```text
01_data_collection/downloads/standard/
fangraphs_standard_2018_2026.csv
```

```text
01_data_collection/downloads/value/
fangraphs_value_2018_2026.csv
```

#### 주요 처리

- Standard와 Value 데이터를 `Season + playerid` 기준으로 결합
- 선수명, 팀, 포지션이 두 파일에서 일치하는지 확인
- Value 관련 열 이름 정리
- 투수와 타자 구분
- 오타니 별도 식별 및 일반 타자 표본에서 제외
- PA 기준으로 모델 사용 가능 표본 플래그 생성
- 2026년 데이터가 부분 시즌 스냅샷임을 표시

#### 주요 표본 플래그

```text
eligible_all_hitters_pa1       : PA 1 이상
eligible_main_pa50             : PA 50 이상
eligible_stable_pa100          : PA 100 이상
eligible_regular_pa200         : PA 200 이상
```

#### 주요 Value 변수

```text
value_batting
value_base_running
value_fielding
value_positional
value_offense
value_defense
value_rar
value_war
value_dollars
```

#### 출력

```text
02_data_preprocessing/interim/
batting_standard_value_2018_2026_flagged.csv
```

---

### Code 4. `04_prepare_payroll_features.py`

#### 역할

Payroll 원본과 선수·시즌 연봉 데이터를 이용해, 현재 시즌 선수의 계약·나이·서비스 타임 관련 특성을 만든다.

#### 입력 데이터

```text
02_data_preprocessing/interim/
payroll_2019_2026_raw_combined.csv
```

```text
02_data_preprocessing/interim/
payroll_player_season_salary_2019_2026.csv
```

#### 주요 처리

- 선수·시즌별 나이 추출
- 나이 제곱값 생성
- 30세 이상, 35세 이상 여부 생성
- 서비스 타임을 연·일·총일수로 변환
- 서비스 타임 3년 이상, 6년 이상 여부 생성
- 현재 계약 단계를 범주형 변수로 변환
- 보장계약, 연봉조정, 프리 아비트레이션 여부 생성
- 40인 로스터 제외 여부 생성
- 여러 계약 그룹에 동시에 포함되는지 확인
- 현재 총연봉 및 retained salary 연결
- 현재 연봉 지급 팀 수와 연봉 구성요소 수 생성

#### 주요 생성 변수

```text
current_age
current_age_squared
age_30_plus
age_35_plus
current_service_years
current_service_days
current_service_total_days
service_3_years_plus
service_6_years_plus
current_contract_stage
current_salary_total
current_retained_salary_total
current_paying_team_count
current_salary_component_count
```

#### 출력

```text
02_data_preprocessing/interim/
payroll_player_features_2019_2026.csv
```

---

### Code 5. `05_build_modeling_dataset.py`

#### 역할

현재 시즌의 타격·계약·연봉 특성과 다음 시즌 연봉을 연결해, 학습용 데이터와 2027년 예측용 데이터를 만든다.

#### 입력 데이터

```text
02_data_preprocessing/interim/
batting_standard_value_2018_2026_flagged.csv
```

```text
02_data_preprocessing/interim/
payroll_player_features_2019_2026.csv
```

```text
02_data_preprocessing/interim/
payroll_player_season_salary_2019_2026.csv
```

#### 주요 처리

- 타격 데이터와 현재 시즌 Payroll 특성 결합
- 현재 시즌 `t`와 다음 시즌 `t+1` 연봉 연결
- 현재 연봉과 다음 시즌 연봉에 `log1p` 변환 적용
- 학습 가능 기간과 예측 기간 구분
- 현재 시즌 Payroll 존재 여부 확인
- 다음 시즌 Payroll 존재 여부 확인
- 현재 연봉과 목표 연봉의 완전성 확인
- PA 기준별 학습·예측 적격 플래그 생성

#### 시간 정렬

```text
2019년 특성 → 2020년 연봉
2020년 특성 → 2021년 연봉
...
2025년 특성 → 2026년 연봉
2026년 특성 → 2027년 예측 데이터
```

#### 출력

```text
02_data_preprocessing/model_input/
modeling_master_hitters_2019_2026.csv
```

```text
02_data_preprocessing/model_input/
training_hitters_2019_2025.csv
```

```text
02_data_preprocessing/model_input/
prediction_hitters_2026_for_2027.csv
```

---

### Code 6. `06_prepare_model_features.py`

#### 역할

Code 5에서 만든 데이터를 모델에 바로 입력할 수 있는 형태로 변환한다.

#### 입력 데이터

```text
02_data_preprocessing/model_input/
training_hitters_2019_2025.csv
```

```text
02_data_preprocessing/model_input/
prediction_hitters_2026_for_2027.csv
```

#### 주요 처리

- 모델에 필요한 식별자, 범주형 변수, 수치형 변수 선택
- 불필요하거나 데이터 누수 위험이 있는 열 제외
- 타격 누적값을 PA 기준 비율로 변환
- Value 지표를 600 PA 기준으로 환산
- 도루 성공률 계산
- retained salary에 `log1p` 변환 적용
- 학습 목표값 결측 여부 검사
- 예측 데이터에 미래 연봉 목표값이 포함되지 않았는지 검사
- 모델 특성 세트와 전처리 방법을 manifest 파일로 저장

#### 주요 생성 비율 변수

```text
pa_per_game
hit_rate
home_run_rate
walk_rate
strikeout_rate
stolen_base_rate
stolen_base_success_rate
```

#### 주요 600 PA 환산 변수

```text
value_batting_per_600_pa
value_base_running_per_600_pa
value_fielding_per_600_pa
value_offense_per_600_pa
value_defense_per_600_pa
value_rar_per_600_pa
value_war_per_600_pa
value_dollars_per_600_pa
```

#### 출력

```text
02_data_preprocessing/model_ready/
training_model_ready.csv
```

```text
02_data_preprocessing/model_ready/
prediction_2027_model_ready.csv
```

```text
02_data_preprocessing/model_ready/
feature_manifest.json
```

---

## 3. 최종적으로 만들어진 파일

### 3.1 모델 학습용 최종 파일

```text
02_data_preprocessing/model_ready/
training_model_ready.csv
```

#### 용도

- 2019~2025년의 선수 특성을 입력값으로 사용
- 2020~2026년의 다음 시즌 연봉을 목표값으로 사용
- 모델 학습과 검증에 사용

#### 주요 목표 변수

```text
log1p_target_salary
```

원래 달러 단위 연봉은 다음 열에 저장된다.

```text
target_salary_total
```

---

### 3.2 2027년 연봉 예측용 최종 파일

```text
02_data_preprocessing/model_ready/
prediction_2027_model_ready.csv
```

#### 용도

- 2026년 선수 성적과 계약 정보를 입력값으로 사용
- 2027년 연봉을 예측하기 위한 데이터
- 아직 실제 2027년 연봉이 없으므로 목표 변수는 비어 있음

---

### 3.3 모델 특성 정의 파일

```text
02_data_preprocessing/model_ready/
feature_manifest.json
```

#### 용도

- 모델에서 사용할 특성 목록 정의
- 수치형 변수와 범주형 변수 구분
- 목표 변수 정의
- 결측값 처리 방법 정의
- 범주형 인코딩 방식 정의
- 모델별 특성 세트 정의

포함된 주요 특성 세트는 다음과 같다.

```text
baseline_previous_salary
batting_value_rate
full_main_rate_safe
full_expanded_sensitivity
```

---

## 최종 요약

이 단계는 여러 개의 FanGraphs 원본 CSV를 정리하고 결합하여, **현재 시즌 선수 정보로 다음 시즌 연봉을 예측할 수 있는 모델용 데이터셋을 만드는 과정**이다.

최종 산출물은 다음 세 파일이다.

```text
training_model_ready.csv
prediction_2027_model_ready.csv
feature_manifest.json
```

이 중 실제 모델 학습에 사용하는 핵심 파일은 다음과 같다.

```text
training_model_ready.csv
```

2027년 연봉 예측에 사용하는 파일은 다음과 같다.

```text
prediction_2027_model_ready.csv
```
