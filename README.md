<div align="center">

# AI 기반 수요 예측과 차량 배치 최적화를 통한<br>마을버스 탄소 감축 및 교통약자 이동권 보장 방안

**팀 작다고 감편하지 마라** · 전예준, 김명관<br>
<sub>한겨레 × 숲과나눔 「AI와 함께하는 교통문제 해결을 위한 데이터 분석 공모전」</sub>

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-speed%20%C2%B7%20quantile-EB5B25)
![SciPy](https://img.shields.io/badge/SciPy-MILP%20%C2%B7%20Dijkstra-8CAAE6?logo=scipy&logoColor=white)
![GeoPandas](https://img.shields.io/badge/GeoPandas-spatial-139C5A)
![Data](https://img.shields.io/badge/public%20data-24%20sets%20%C2%B7%20110GB-2A78D6)
![Trips](https://img.shields.io/badge/smart%20card-846%2C653%20trips-2A78D6)
![Tests](https://img.shields.io/badge/tests-19%20passed-2EA44F)

**[📄 분석보고서 (제출본 PDF)](docs/report/작다고감편하지마라_분석보고서.pdf)** &nbsp;·&nbsp; [보고서 본문](docs/report/maeul_bus_integrated_report.md) &nbsp;·&nbsp; [재현 방법](#repro) &nbsp;·&nbsp; [데이터 명세](data/README.md)

</div>

---

수요가 적은 마을버스 노선은 재정 지표로만 보면 감편 대상이 되기 쉽다. 그러나 언덕·골목에 사는 고령자에게는 그 노선이 이동의 유일한 수단일 수 있고, 반대로 적은 승객을 큰 차로 실어 나르면 탄소 효율이 떨어진다. 이 프로젝트는 서울 마을버스 230개 노선을 대상으로 **어떤 서비스를 지켜야 하는가**와 **어떤 운행의 차량을 바꿀 수 있는가**를 데이터로 답한다.

> [!IMPORTANT]
> **노선은 지키고, 차량은 운행에 맞게.** 고령자가 의존하는 노선은 없앨 때의 피해가 크고, 평균 승객이 적은 시간에도 작은 차의 정원을 넘는 운행이 있다. 감편이 아니라 운행별로 맞는 차를 보내는 것이 탄소와 이동권을 함께 지키는 방법이다.

<table>
<tr>
<td align="center" width="25%">🌿 <b>탄소 효율</b><h3>223 / 230</h3>승객 1명·1km당 CO₂가<br>승용차보다 적은 노선<br><sub>중앙값 91.1g 대 308.0g</sub></td>
<td align="center" width="25%">👵 <b>이동권</b><h3>약 2배</h3>노선을 없앨 때 보행이 늘어나는<br>고령 인구(의존도 높은 노선)<br><sub>1,372~1,684명 대 641~749명</sub></td>
<td align="center" width="25%">🤖 <b>AI 피크 예측</b><h3>5.34% → 0.21%</h3>소형차 후보 운행의<br>정원(16명) 초과율<br><sub>과거 기록 기준 대비</sub></td>
<td align="center" width="25%">🚐 <b>실행 후보</b><h3>5대 · 3개 노선</h3>운영 제약을 통과한 시범 후보<br>(중랑01·서초05·서초01)<br><sub>하루 44.1kgCO₂eq, 20대로 늘려도 동일</sub></td>
</tr>
</table>

![분석 흐름](docs/report/images/integrated/00_full_flow.png)

## 데이터

7개 기관·10개 플랫폼의 공공데이터 24종(원자료 약 110GB)과 교통카드 운행기록 35일분(846,653개 운행)을 결합했다. 자료별 선정 이유·사용처·전처리는 [데이터 명세](data/README.md)에 있다.

| 묶음 | 데이터 | 쓰임 |
| --- | --- | --- |
| 도로 속도 | 표준노드링크, 5분 통행속도, DEM, 레이더 강수, 도로 시설·사고다발지역, 교통량(View-T), 생활인구 | AI 속도 모델의 학습 목표와 변수 |
| 수송·운행 | 노선별 정류소, 버스 실제 경로(API), 정류장별 승하차, 운행횟수 | 노선 맵매칭, 재차인원 |
| 고령자·대체수단 | 생활인구(250m 격자·행정동), 지하철 역사 | 의존도, 노선 제거 시 보행 접근성 |
| 차량 구성 | 연료별 자동차 등록, 마을버스 연료 현황 | 버스·승용차 배출 가중 |
| 운행별 혼잡 | 교통카드 이용 운행기록(T-DATA, 7/6~8/9), 8월 운행횟수 | 운행별 최대 탑승 예측, 차량 배치 |

## 분석 방법

**① 탄소 비교.** 버스 실제 경로를 방향성 도로망에 맵매칭하고, 관측속도와 AI 추정 속도, DEM 경사로 같은 길·같은 시간대·같은 경사에서 버스와 승용차의 에너지를 계산했다. 전기차는 VT-CPEM, 승용차는 VT-CPFM 계열·하이브리드 모형, 경유·CNG 버스는 서울 실측 배출 비율을 쓰고, 서울의 연료 구성으로 가중했다.

$$
\text{버스 승객·km당 배출} = \frac{b}{L},\qquad \text{승용차} = \frac{c}{o},\qquad \text{손익분기 재차인원 } L^{*} = \frac{b \cdot o}{c}
$$

<sub>b, c: 버스·승용차의 차량·km당 배출, L: 평균 재차인원(하루 승객·km ÷ 하루 차량·km), o: 승용차 탑승인원 1.2명</sub>

**② 의존도와 접근성.** 고령 비율·고령 생활인구·대체수단 없는 승차 비중의 순위 평균으로 의존도를 만들고, 의존도와 승객·km당 탄소의 중앙값으로 네 유형을 나눴다. 노선을 하나씩 없애고 Tobler 보행속도(고령자 0.75배, 경사 반영)로 250m 격자별 정류장 접근시간을 다시 계산했다.

**③ 운행 피크와 차량 배치.** 교통카드 태그를 정류장 순서로 누적해 운행별 최대 탑승 인원을 재구성하고, 피크를 분위수 회귀로 예측했다. 예측 상한이 정원 이하이고 배터리·차량 수·자료 일관성을 만족하는 차량별 하루 운행 묶음을 MILP로 골랐다.

## AI 활용

| 기술 | 역할 | 검증 |
| --- | --- | --- |
| **도로 속도 XGBoost** | 속도 관측이 없는 도로(서울 링크의 29%, 마을버스 노선 길이의 13.2%)의 시간대별 속도 추정 | 학습에 쓰지 않은 도로로 평가(도로 단위 3-fold GroupKFold): R² 0.72·MAE 4.6km/h(이웃 관측 있음), R² 0.31(이웃 관측 없음, 노선 길이의 2.5%) |
| **운행 피크 XGBoost 분위수 회귀** | 운행별 최대 탑승의 Q50·Q95·Q99 예측, 보정 주 잔차로 Q99 보정 | 날짜 분리(학습 7/6~19, 보정 7/20~26, 개발 7/27~8/2, 최종 검증 8/3~9). 최종 검증 128,202개 운행에서 예측 상한 포함률 99.29% |
| **MILP 차량 배치** | 확보 대수·배터리 안에서 교체할 운행 묶음 선택 | 보정 주로 배분하고 검증 주에 고정 적용. 한계편익 탐욕법과 최적값 일치 |

분석 코드 작성과 교차 검토에는 생성형 AI(Claude Code, OpenAI Codex)를 썼고, 채택 여부는 연구자가 판단했다. 이 과정에서 카운티 일렉트릭 주행거리(303km)를 복합 값으로 잘못 읽은 에너지 보정 오류를 찾아 모든 수치를 다시 계산했다.

## 결과

### 마을버스는 '조건부 친환경'이다

손익분기 재차인원은 1.53명, 실제 노선 중앙값은 5.38명이다. 경유·CNG 버스 1,229대를 모두 전기버스로 바꾸면 하루 **14.33톤**, 평균 재차 3명 미만 시간대(운행의 16.1%)를 소형 전기차로 바꾸면 하루 **5.58톤**(그중 차량 크기 효과 3.73톤)이 준다. 승용차 탑승인원·버스 배출·승용차 배출 가정을 바꾼 18조건에서도 버스 우위는 157~223개였다.

![탄소 효율과 감축 잠재량](docs/report/images/integrated/01_carbon_and_potential.png)

### 고령자 의존 노선은 줄일 대상이 아니다

증편·지원 우선 60, 소형 차량 전환 57, 현행 유지 55, 노선 조정 검토 58개 노선. 고의존·상대 저효율 57개 노선은 운행비 충당이 어렵지만 노선을 없앨 때의 보행 피해는 증편·지원 우선 유형에 가깝다. 서대문01·관악04는 노선 유지가 필요하지만 운행 피크(95백분위 31명·53명)가 커 작은 차로 바꿀 수 없다.

![정책 유형과 접근성 손실](docs/report/images/integrated/02_equity_and_access.png)

### 평균만 보면 피크를 놓친다

평균 기준 소형화 후보 28개 노선 중 24개는 검증 주 피크 상한의 99백분위가 16명을 넘었다. 과거 기록(노선·요일·시간대 99백분위)과 비교해 AI는 소형차 후보 비중을 24.24%에서 5.11%로 줄이는 대신 정원 초과율을 **5.34%에서 0.21%**로 낮췄다.

![운행 피크와 AI](docs/report/images/integrated/03_peak_and_ai.png)

## 정책 제안

노선 유지·지원 판단에는 접근성 손실을, 차량 전환 판단에는 운행별 피크와 에너지 조건을 적용한다.

| | 제안 | 연결되는 현행 제도 |
| :---: | --- | --- |
| **1** | **지원 심사와 서비스 평가에 탄소·접근성·수송 용량을 함께 반영**<br>승객·km당 탄소, 노선 제거 시 고령자 접근성 영향, 피크 수송 용량을 평가표에 넣는다. 접근성 손실이 큰 노선의 조정은 '피크 고정노선 + 한산 시간 호출형(DRT)'을 전제로, 2026년 증편은 손익분기 전환인원이 낮은 29개 후보부터 | 2025.12 서울시–조합 합의(재정지원 500억 원, 서비스 평가 인센티브 2027년 정착) |
| **2** | **저상버스 예외 노선은 소형 저상 전기차로 단계 전환**<br>은평구 예외 노선 중 분석 대상 4개 가운데 3개가 소형 차량 전환 유형. 피크 예측 상한이 정원 이하인 운행부터 바꾼다 | 2026.10 서울시 조례(2030년 버스 전량 저상화, 예외 노선 사유 해소 방안 의무) |
| **3** | **운행별 적합성을 확인하는 단계적 전환과 보조금 연계**<br>중랑01·서초05·서초01의 일부 운행 묶음에서 비교군을 둔 시범운행, 보조금에 '운행별 수요 검증 통과' 요건 | 친환경 마을버스 보조금(차종별 대당 정액) |

현행 제도·근거·성과지표는 [보고서 본문의 정책제안](docs/report/maeul_bus_integrated_report.md#정책제안-및-기대효과)에 있다.

## 한계

- 노선별 실제 차량 정보가 없어 공통 중형 전기버스와 서울 연료 구성을 적용했다. 대형 비중은 버스 배출 1.28·1.5배 민감도로 보았다.
- 정지·출발·체류 에너지는 가정이며 실차 충전 기록으로 검증하지 않았다.
- 교통카드는 태그하지 않은 승객을 관측하지 못하고, 배치 운행의 15.7%는 정원 검증이 불가능했다. 운행 자료는 여름 5주분이다.
- 소형차 제원(휠체어 배치 16명·105kWh)은 국내 인증 차종이 아니다.
- 접근성 손실은 가장 가까운 정류장까지의 보행만 반영한다.
- 시간대 잠재량(7월 집계)과 차량 배치 감축(검증 주 운행 묶음)은 기간·단위가 달라 합산하지 않는다.

## 저장소 구조

```text
seoul-maeul-bus-carbon/
├── src/
│   ├── pipeline/          01~05 도로 속도 데이터 구축(서울 링크, 5분 속도 집계, 레이더 강수, 링크 차원 표, 관측속도 팩트)
│   ├── models/            01~04 AI 속도 모델(A 조회표, B·C XGBoost), 학습된 모델 artifacts/
│   ├── energy/            차량 에너지·배출 모델(VT-CPEM, VT-CPFM 계열, HEV, 경유·CNG 환산, 공인값 보정)
│   └── transit/
│       ├── 00~10          노선 맵매칭, 재차인원, 탄소, 유형, 운영, 노선 제거 접근성
│       ├── upgrade/       교통카드 운행 재구성, 피크 AI, 차량 배치 MILP, 수정 에너지 통합 재계산
│       └── checks/        보고서 수치 재현, 가정 민감도, AI 속도 ±5km/h, 군집 점검
├── config/maeul_upgrade/  차량·에너지·배치 가정(scenario.json)
├── notebooks/             속도 모델 개발 실험 기록
├── tests/                 단위 테스트 19개
├── docs/                  방법 문서, report/(제출본·본문·그림·공개 지표)
└── data/                  데이터 명세(원자료·산출물은 저장소에 포함하지 않음)
```

<a id="repro"></a>

## 재현 방법

Windows, Python 3.13에서 실행했다. 원자료는 저장소에 없으므로 [데이터 명세](data/README.md)의 출처에서 받아 같은 경로에 둔다.

> [!NOTE]
> 원자료 없이 바로 실행되는 것은 단위 테스트다. API 키는 코드와 저장소에 넣지 않고 환경변수로만 읽는다.

```powershell
pip install -r requirements.txt
python -X utf8 -m unittest discover -s tests -v          # 19개
$env:SEOUL_BUS_API_KEY = "<서울시 버스운행정보 API 키>"    # 노선 경로를 다시 받을 때
$env:KMA_API_KEY = "<기상청 API허브 키>"                    # 레이더 강수를 다시 받을 때
```

<details>
<summary><b>1단계 · 도로 속도와 AI 속도 모델</b></summary>

학습된 모델과 조회표는 저장소에 있어, 링크 차원 표(`pipeline/04`)를 만든 뒤에는 재학습 없이 예측할 수 있다.

```powershell
python src/pipeline/01_extract_seoul_links.py
python src/pipeline/02_aggregate_5min_speed.py
python src/pipeline/03_fetch_kma_radar.py --from-5min-times
python src/pipeline/04_build_link_dimension.py
python src/pipeline/05_build_speed_fact.py
python src/models/04_batch_predict_all_link_speed.py --start-date 20260707 --end-date 20260720
# 재학습: models/01_build_model_a.py, 02_build_model_bc.py --mode validate / --mode final
```

</details>

<details>
<summary><b>2단계 · 마을버스 기본 분석(01~10)</b></summary>

```powershell
python src/transit/01_build_bus_route_links.py
python src/transit/02_build_route_demand.py
python src/transit/03_build_route_feasibility.py
python src/transit/04_build_route_energy.py
python src/transit/05_build_route_carbon_breakeven.py
python src/transit/06_classify_route_priority.py
python src/transit/07_build_route_operations.py
python src/transit/08_access_gap.py
python src/transit/09_validate_ai_speed.py
python src/transit/10_route_removal_access.py
```

</details>

<details>
<summary><b>3단계 · 운행 피크 AI·차량 배치와 보고서 수치</b></summary>

```powershell
pip install xgboost==3.4.0                                       # 피크 모델 결과를 그대로 재현하는 버전
python -X utf8 -m src.transit.upgrade.run --download --integrated
python src/transit/checks/report_numbers.py                      # 보고서 수치 출력
python src/transit/checks/assumption_sensitivity.py
python src/transit/checks/ai_speed_stress.py
```

`run --download`는 교통카드 공개 원본(35일)을 받아 운행 재구성부터 피크 예측·차량 배치·민감도까지 실행하고, `--integrated`는 수정한 버스 에너지로 탄소·접근성을 `data/processed/transit_full/`에 다시 계산한다. 이미 받은 자료가 있으면 `--download`를 뺀다.

</details>

**검증 기록.** 새로 복제한 저장소에서 위 순서로 기본 분석 산출물과 저장된 속도 모델의 전 링크 예측이 기존 결과와 일치함을 확인했다. 운행 피크 모델과 차량 배치 결과는 기본 환경에 xgboost 3.4.0을 설치한 상태에서 제출본 수치와 같게 재현된다. `checks/`의 스크립트는 `transit_full`을 읽어 보고서 수치와 강건성 결과를 출력한다(AI 추정 속도 ±5km/h에서 버스 우위 224·223개, 유형 유지 226개).

## 문서 안내

| 문서 | 내용 |
| --- | --- |
| [분석보고서 본문](docs/report/maeul_bus_integrated_report.md) | 제출본과 같은 본문, 그림 1~4, 참고문헌 |
| [데이터 명세](data/README.md) | 자료별 선정 이유, 사용처, 전처리, 결합 키, 기간, 개인정보 처리 |
| [AI 속도 모델](docs/SPEED_MODEL.md) | A·B·C 계층 구조, 변수, 검증, 한계 |
| [마을버스 분석 단계](docs/TRANSIT_MAEUL_BUS.md) | 01~10 단계, 차량 에너지 모델, 주요 결과 |
| [운행별 피크 예측과 차량 배치](docs/MAEUL_UPGRADE_IMPLEMENTATION.md) | 운행 재구성, 피크 모형, MILP 가정 |
| [운행 검증 상세](docs/report/maeul_bus_upgrade_validation.md) | 모형 비교, 제약 단계, 차량 규모, 민감도 |
| [통합 재계산](docs/MAEUL_INTEGRATED_REPORT.md) | 보고서 수치의 계산 경로와 이전 결과와의 차이 |
| [모델 개발 노트북](notebooks/README.md) | 속도 모델 설계 실험 기록 |

---

<div align="center">
<sub>작다고 감편하지 마라 · 전예준, 김명관 · 2026</sub>
</div>
