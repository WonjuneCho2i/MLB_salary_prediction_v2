# 2단계: FanGraphs 팀별 Payroll 수집

FanGraphs 비회원 공개 화면에 보이는 **현재 팀·현재 시즌 1개**만 CSV로 저장합니다. 자동으로 여러 페이지를 순회하지 않습니다.

## 1. Tampermonkey에 스크립트 설치

1. Tampermonkey 대시보드에서 `+` 또는 **새 사용자 스크립트 만들기**를 누릅니다.
2. 기본으로 들어 있는 내용을 전부 지웁니다.
3. `fangraphs_payroll_export.user.js`의 내용을 전부 복사해 붙여 넣습니다.
4. `Ctrl+S`로 저장하고 스크립트가 활성화됐는지 확인합니다.

기존 Leaderboard 저장 스크립트와는 별개의 스크립트입니다.

## 2. 우선 Yankees 2019 한 파일만 시험

다음 페이지를 엽니다.

<https://www.fangraphs.com/roster-resource/payroll/yankees?season=2019>

표가 모두 로딩되면 오른쪽 아래의 **FG 현재 팀·시즌 Payroll CSV 저장** 버튼을 누릅니다.

정상 파일명:

```text
fangraphs_payroll_NYY_2019.csv
```

CSV에는 선수별 `Season`, `Team`, `Player`, `playerid`, 계약 구분, 나이, 서비스 타임, 계약, AAV, 해당 시즌 연봉이 저장됩니다. 화면에 중복으로 존재하는 모바일용 표는 제외합니다.

## 3. 다운로드 파일 보관

다운로드한 CSV를 다음 폴더로 옮깁니다.

```text
01_data_collection/downloads/payroll/
```

지금은 Yankees 2019 한 파일만 저장한 뒤 내용이 정상인지 확인합니다. 확인 전에는 다른 239개 팀·시즌 페이지를 진행하지 않습니다.

## 연봉 열 사용 원칙

- 2019 페이지에서는 `2019` 열만 `Salary`로 저장합니다.
- 2020 페이지에서는 `2020` 열만 저장하는 식으로 현재 페이지 시즌과 같은 열만 사용합니다.
- `AAV`는 별도 열이며 실제 해당 시즌 연봉과 혼용하지 않습니다.
- `Salary_raw`에는 화면 원문, `Salary`에는 쉼표와 달러 기호를 제거한 숫자가 들어갑니다.
- 투수도 함께 저장됩니다. 나중에 Standard 타격 데이터와 `playerid`로 결합하면 타자만 남길 수 있습니다.
