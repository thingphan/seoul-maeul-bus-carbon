# 마을버스 탄소·고령자 이동권 분석(`src/transit/`)

AI 속도 모델([SPEED_MODEL.md](SPEED_MODEL.md))과 차량별 에너지 모델(`src/energy/`)을 서울 마을버스 노선에 적용한 분석이다.

- 서울 마을버스 247개 노선을 실제 경로(서울시 버스 API)로 도로망에 올리고, AI 속도와 경사로 노선별 에너지를 계산한다.
- 같은 길, 같은 시간대, 같은 경사에서 **승객 1명·1km당 탄소**를 마을버스와 승용차로 비교한다. 버스와 승용차 모두 2026년 서울의 실제 연료 구성을 반영한다.
- 교통약자 의존도(고령 생활인구, 대체수단 없는 정류장)와 탄소 효율로 노선을 네 유형으로 나눈다.
- 시간대별 재차인원, 차량 크기·운행 시간 시나리오, 노선 제거 시 고령자 보행 접근성 손실을 계산한다.

보고서 본문은 [report/maeul_bus_report.md](report/maeul_bus_report.md)에 있다.

## 실행

입력 데이터는 Git에 포함하지 않는다. 받는 곳과 저장 경로는 [data/README.md](../data/README.md)에 있다.

```powershell
pip install -r requirements.txt
python -m unittest tests.test_link_energy          # 에너지 모델 테스트(10개)

# 마을버스 분석: 번호 순서대로
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

# 보고서 수치 재현과 강건성 점검
python src/transit/checks/report_numbers.py
python src/transit/checks/assumption_sensitivity.py
python src/transit/checks/ai_speed_stress.py
python src/transit/checks/cluster_check.py

# 보고서 그림(선택, docs/report/images에 저장)
python src/transit/report_figures.py --font-dir <Pretendard 폴더>
```

- 결과는 `data/processed/transit/`에 저장된다. 캐시가 있으면 01~10 전체가 약 2~3분 걸린다.
- **00(노선 실제 경로 수집)**은 `data/raw/traffic/bus_route_path/route_paths.csv`가 있으면 실행하지 않아도 된다. 다시 받으려면 서울시 버스운행정보 API 인증키를 환경변수로만 넣는다(코드·문서에 적지 않는다).

  ```powershell
  $env:SEOUL_BUS_API_KEY = "<인증키>"
  python src/transit/00_fetch_bus_route_paths.py   # 이미 받은 노선은 건너뜀(전부 다시: --refresh)
  ```
- 그림 글꼴은 [Pretendard](https://github.com/orioncactus/pretendard)(OFL)다. `--font-dir`이나 환경변수 `REPORT_FONT_DIR`로 .otf 폴더를 넘기고, 없으면 맑은 고딕을 쓴다. 노선 지도 그림에는 서울 행정경계 파일(`data/raw/boundary/seoul_admin_boundary.geojson`)이 필요하다.

### 캐시

원자료를 단계마다 다시 읽지 않도록 아래 캐시를 쓴다. 원자료가 캐시보다 새로우면 다시 만들고, 원자료가 없으면 있는 캐시를 그대로 쓴다.

| 캐시(`data/processed/transit/`) | 만드는 곳 | 없을 때 필요한 것 |
|---|---|---|
| `route_stops.parquet` | `route_data.py` | 정류소 엑셀(OA-1095) |
| `bus_trip_counts.parquet` | `route_data.py` | TOPIS 운행횟수 원자료(`bus_trips/`) |
| `grid_population_daytime.csv`, `dong_population_night.csv` | 02 | 생활인구 원자료와 격자 도형(`match/`). 강제 재생성 `--rebuild` |
| `link_speed_weekday_profile.parquet` | 03 | 전 링크 속도 예측 CSV 335개(`data/processed/all_link_speed_prediction_*.csv`). `python src/models/04_batch_predict_all_link_speed.py --start-date 20260707 --end-date 20260720`로 생성 |

## 단계별 역할

| 순서 | 파일 | 하는 일 | 주요 산출물 |
|---|---|---|---|
| 00 | `00_fetch_bus_route_paths.py` | 버스 API(getRoutePath)로 노선의 실제 경로 좌표 수집 | `data/raw/traffic/bus_route_path/route_paths.csv` |
| 01 | `01_build_bus_route_links.py` | 실제 경로를 20m 간격으로 나눠 진행 방향이 맞는 표준노드링크에 맵매칭. 노선·정류장 구간 길이는 실제 경로선에서 측정. 정류장별 지하철역·일반버스 접근성 | `routes.csv`, `route_links.csv`, `route_segments.csv`, `bus_stops.csv` |
| 02 | `02_build_route_demand.py` | 7월 승하차, 정류장 300m 안 평일 낮 고령 생활인구, 대체수단 없는 정류장의 승차 비중, 노선 주변 1,000명당 자가용 승용차 | `route_demand.csv`, `stop_ridership.csv` |
| 03 | `03_build_route_feasibility.py` | 평일 시간대별 링크 속도 프로필, 노선별 도로 조건(경사·1차로·AI 추정 구간 비중) | `link_speed_weekday_profile.parquet`, `route_feasibility.csv` |
| 04 | `04_build_route_energy.py` | 전기 중형 마을버스(카운티 일렉트릭 보정) 노선별 에너지 | `route_energy.csv` |
| 05 | `05_build_route_carbon_breakeven.py` | 구간 승객 누적 → 승객·km, 운행횟수 → 재차인원, 버스·승용차 승객 1명당 탄소, 손익분기, 증편 손익분기 | `route_carbon_breakeven.csv` |
| 06 | `06_classify_route_priority.py` | 탄소 효율 × 교통약자 의존도 네 유형, 15인승급 소형 전기차 시나리오 | `route_priority.csv` |
| 07 | `07_build_route_operations.py` | 시간대별 재차인원, 재차인원 3명 미만 시간대 소형화 감축, 운행비 충당 범위(승차당 실수입 600·900·1,200원), 증편 검토 후보 | `route_hourly_load.csv`, `route_operations.csv` |
| 08 | `08_access_gap.py` | 정류장까지 고령자 보행 시간(Tobler × 0.75, 도로망·직선 간이), 도로망 거리로 본 대체수단과 유형 민감도 | `grid_walk_access.csv`, `stop_network_access.csv`, `route_class_network.csv` |
| 09 | `09_validate_ai_speed.py` | 분석 속도와 TOPIS 마을버스 실측 시간대 속도 비교 | `ai_speed_validation.csv` |
| 10 | `10_route_removal_access.py` | 노선을 하나씩 제거할 때 고령자 접근시간 증가(다중 출발 Dijkstra, 도로망·직선 간이) | `route_removal_access.csv`, `route_removal_cells.csv` |

공통 모듈: `route_data.py`(캐시와 공통 입력), `route_classes.py`(네 유형 규칙), `road_network.py`(도로망·최단 거리·고령자 보행 시간), `route_kinds.py`(노선 종류 구분: 승하차 데이터의 공식 분류 '서울마을버스'로 선정, 큰·작은 순환으로 나뉜 노선은 정류장별 운행횟수 비율로 승하차 배분).

### 검증 스크립트(`src/transit/checks/`)

| 파일 | 확인하는 것 |
|---|---|
| `report_numbers.py` | 보고서 핵심 수치를 산출물에서 다시 계산 |
| `assumption_sensitivity.py` | 승용차 탑승인원(1.2·1.35·1.5·1.82명), 버스·승용차 배출 배율, 대체수단 거리(150·200·300m, 도로망) |
| `ai_speed_stress.py` | AI 추정 구간 속도 ±5km/h |
| `cluster_check.py` | K-평균·가우시안 혼합·HDBSCAN 군집의 분리도와 네 유형 일치도 |
| `validate_bc_tiers.py` | AI 속도 모델 B·C의 계층별 검증 성능(약 1시간) |

## 차량별 에너지 모델

모든 차량을 **같은 노선, 같은 평일 시간대 속도, 같은 경사**로 계산하고, 공인 시험값(또는 서울 실측)으로 보정해 정지·출발 소비를 같은 기준으로 반영한다.

| 차량 | 모델 | 보정 근거 | 코드 |
|---|---|---|---|
| 휘발유·경유·LPG 승용차 | VT-CPFM 계열 연료 모델(Rakha et al. 2011) | 쏘나타 2.0, 쏘렌토 2.2 디젤, 쏘나타 LPi 공인 연비 | `src/energy/ice_fuel.py` |
| 하이브리드 승용차 | Ahn & Rakha(2019) HEV 모델 | 쏘나타 하이브리드 공인 연비 | `src/energy/hev_fuel.py` |
| 전기 승용차 | VT-CPEM(Fiori et al. 2016) | 아이오닉5 공인 전비(공기저항·부가 전력 역산) | `src/energy/ev_calibration.py` |
| 전기 마을버스(중형) | VT-CPEM | 카운티 일렉트릭: 배터리 128kWh, 복합 303km, 질량 7t, 냉방 5kW | `src/energy/ev_calibration.py`, `04` |
| 경유·CNG 마을버스 | 전기버스 전력 × 서울 실측 비율 | 서울 동일 노선: 전기 1.43kWh/km 대비 경유 723g, CNG 721gCO₂/km(전기전자학회논문지 24(3), 2020) | `src/energy/bus_fuel.py` |
| 소형 전기차(15인승급) | VT-CPEM | 3.5t, 냉방 3kW, 정지·출발 부가 전력은 마을버스 기준을 질량 비례 적용 | `06`, `07` |

- 경유·CNG 버스에 전용 연료 모델을 쓰지 않은 이유: 그 계수는 초 단위 가감속으로 보정되어 있어 링크 정속 근사에 넣으면 정류장 정지·출발 연료가 빠져 과소 추정된다.
- 탄소 계수: 전력 0.4173kgCO₂/kWh(국가 전력배출계수 2023년도, 2025.12 공표), 휘발유 2.29·경유 2.66kg/L, LPG 1.68kg/L.
- 차량 구성: 승용차는 서울 2026년 7월 등록(휘발유 57.4%, 경유 20.1%, 하이브리드 12.7%, LPG 6.0%, 전기·수소 3.9%), 마을버스는 2026년 1월 현황(CNG 885, 경유 344, 전기 397대). 승용차 탑승인원 1.2명.
- 마을버스 크기는 중형(카운티)으로 고정했다. 실제 서울 마을버스는 대형이 76%이며, 버스 배출 1.28·1.5배 민감도로 검토했다.

## 주요 결과(`checks/report_numbers.py` 출력 기준)

| 항목 | 값 |
|---|---|
| 분석 노선 | 247개 중 230개(승객 누적 이상 17개 제외), 하루 승차 약 93만 건 |
| 차량·km당 CO₂(중앙) | 마을버스 506g, 승용차 369g |
| 승객·km당 CO₂(중앙) | 마을버스 98g, 승용차 307g. 버스 우위 218/230개 |
| 손익분기 재차인원 / 실제 재차인원(중앙) | 1.66명 / 5.38명 |
| 시간대 재차인원 | 8시 9.4명, 18시 10.0명, 20~22시 4.9명(이 시간대 운행의 26.3%가 3명 미만) |
| 증편 손익분기 전환인원(중앙) | 9.8명(사분위 7.7~12.7) |
| 전면 전기화 | 하루 15.54톤 감축(현재 113.25톤의 13.7%) |
| 재차인원 3명 미만 시간대 소형화 | 운행 15.9% 전환, 하루 7.53톤(전기화 효과의 48.5%), 차량 크기 효과 5.55톤 |
| 소형 차량 전환 유형 | 55개 중 29개 전환 가능, 하루 4.43톤 |
| 네 유형 | 증편·지원 우선 62 / 소형 차량 전환 55 / 현행 유지 53 / 노선 조정 검토 60 |
| 노선 제거 시 영향 고령 인구(중앙) | 의존 높은 유형 1,372~1,684명, 낮은 유형 641~749명 |
| AI 추정 구간 | 마을버스 노선 길이의 13%, 그중 95%가 1차로(관측 구간 30%). 소형 차량 전환 유형 20%, 다른 유형 8~11% |

## 한계

- 노선별 차량 정보가 없어 서울 전체 연료 구성과 중형 차량을 공통 적용했다.
- 승용차의 골목 저속 연료는 공인 시험주기 보정을 저속으로 확장한 값이라 높게 잡혔을 수 있다(승용차 배출 0.8배 민감도로 검토).
- 정류장 순서(9월)와 승하차(7월)의 시점 차이 등으로 17개 노선은 제외했다.
- 링크 안은 정속으로 달린다고 가정했고 회생제동은 반영하지 않았다.
- 시간대별 재차인원은 승객·km가 승차와 같은 시간 분포를 따른다고 근사했다.
- 운행비 충당은 재정지원 기준액을 1회 운행 원가로 환산한 근사이며, 노선별 실수입은 범위로만 냈다.
- 보행 도로망(표준노드링크)에는 인도·계단·좁은 골목 일부가 없어 도로망 거리와 직선 × 1.3을 범위로 냈다.
- AI 속도 점검의 실측은 10월 버스 속도, 분석 속도는 7월 일반 차량 속도라 시기·차종이 다르다.
