# 1단계: FanGraphs 필수 데이터 수집

이 단계에서는 모델링이나 전처리를 하지 않습니다. FanGraphs에서 보이는 `Standard`와 `Value` 표를 원본 CSV로 저장하고, 필요한 컬럼과 기간이 제대로 수집됐는지만 확인합니다.

공식 `Data Export`는 FanGraphs 회원 전용일 수 있습니다. 여기에 포함된 사용자 스크립트는 숨겨진 API를 호출하거나 로그인 제한을 우회하지 않고, 브라우저 화면에 이미 로드된 표만 CSV로 저장합니다.

## 준비

Chrome에서 Tampermonkey 같은 사용자 스크립트 확장 프로그램을 설치합니다.

1. 확장 프로그램에서 새 사용자 스크립트를 생성합니다.
2. 기본 내용을 모두 지웁니다.
3. `fangraphs_table_export.user.js` 내용을 붙여 넣습니다.
4. 저장하고 활성화합니다.

## FanGraphs 필터

[FanGraphs Major League Leaders](https://www.fangraphs.com/leaders/major-league)에서 다음과 같이 설정합니다.

```text
Player Stats
Batting
League: MLB / All Leagues
Team: All Teams
Split Teams: 끔
Season Type: Regular Season
Multiple Seasons: 2018 to 2026
Split Seasons: 켬
Primary Position: Position Players 또는 Non-Pitchers가 있으면 선택
Min Playing Time (PA): 0
Page Size: Infinity
```

`Primary Position`에서 타자만 선택하기 어렵더라도 괜찮습니다. 사용자 스크립트가 선수 링크의 `position` 정보도 저장하므로 이후 투수 타격 행을 구분할 수 있습니다.

## Standard 저장

1. Stat Presets에서 `Standard`를 누릅니다.
2. 표의 로딩이 끝났는지 확인합니다.
3. 우측 하단의 `FG 현재 표 CSV 저장` 버튼을 누릅니다.
4. 내려받은 파일을 `downloads/standard/`에 넣습니다.

권장 파일명은 다음과 같습니다.

```text
fangraphs_standard_2018_2026.csv
```

## Value 저장

1. 필터를 유지한 상태에서 `Value`를 누릅니다.
2. 표의 로딩이 끝났는지 확인합니다.
3. 다시 `FG 현재 표 CSV 저장` 버튼을 누릅니다.
4. 내려받은 파일을 `downloads/value/`에 넣습니다.

권장 파일명은 다음과 같습니다.

```text
fangraphs_value_2018_2026.csv
```

## 검증

Python 표준 라이브러리만 사용하므로 별도 패키지 설치가 필요 없습니다.

```powershell
cd C:\Users\user\Documents\MLB_salary_prediction_v2\01_data_collection
python validate_exports.py `
  --standard downloads\standard\fangraphs_standard_2018_2026.csv `
  --value downloads\value\fangraphs_value_2018_2026.csv
```

정상적인 경우 다음 항목이 출력됩니다.

- 행 개수
- 수집된 시즌 범위
- 고유 선수 수
- `playerid + Season` 중복 수
- 필수 컬럼 누락 여부
- Standard와 Value의 공통 선수-시즌 개수

## 수집할 때 주의할 점

- `Page Size=Infinity`를 선택하기 전에 저장하면 첫 페이지 일부만 수집됩니다.
- `Split Seasons`를 켜지 않으면 2018-2026 합계 한 행이 되어 다음 시즌 연봉 학습에 사용할 수 없습니다.
- `Split Teams`는 끕니다. 켜면 이적 선수가 한 시즌에 여러 행으로 나옵니다.
- `Salary` 결측은 0으로 바꾸지 않습니다.
- 확장 프로그램이나 사용자 스크립트로 저장한 CSV는 수정하지 않고 Raw 데이터로 보관합니다.
- FanGraphs 이용약관과 합리적인 사용 범위를 지키고, 반복적인 자동 요청은 하지 않습니다.
