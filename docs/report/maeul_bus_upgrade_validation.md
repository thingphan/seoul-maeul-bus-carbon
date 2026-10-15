# 업그레이드 분석 상세 검증

## 데이터 품질
```json
{
  "demand": {
    "dates": [
      "20260706",
      "20260707",
      "20260708",
      "20260709",
      "20260710",
      "20260711",
      "20260712",
      "20260713",
      "20260714",
      "20260715",
      "20260716",
      "20260717",
      "20260718",
      "20260719",
      "20260720",
      "20260721",
      "20260722",
      "20260723",
      "20260724",
      "20260725",
      "20260726",
      "20260727",
      "20260728",
      "20260729",
      "20260730",
      "20260731",
      "20260801",
      "20260802",
      "20260803",
      "20260804",
      "20260805",
      "20260806",
      "20260807",
      "20260808",
      "20260809"
    ],
    "routes": 247,
    "trips": 846653,
    "departure_keys_outside_downloaded_window": 16,
    "boardings": 32095738.0,
    "usable_trips": 641330,
    "split_counts": {
      "calibration": 171463,
      "development": 170096,
      "test": 168696,
      "train": 336398
    },
    "interval_note": "Tagged intervals only; missing pairs and same-minute event order bounded; untagged passengers not observed."
  },
  "source_validation": {
    "route_days": 8628,
    "official_matched_days": 8383,
    "ratio_median": 0.991044776119403,
    "ratio_q10": 0.8205893608878684,
    "ratio_q90": 1.0841751437417073,
    "routes_outside_20pct": 24,
    "note": "Vehicle/departure card keys and median stop-pass counts have different measurement definitions. Ratios outside tolerance are quarantined, not treated as missing empty trips or repaired by multiplying loads."
  },
  "distance": {
    "matched_boarding_share": 0.7921422123908997,
    "routes_geometry_difference_over_10pct": 97,
    "note": "Passenger-km is directly matched only for unique stop occurrences and forward cumulative distances. Unmatched/circular/ambiguous pairs are excluded, not silently imputed. Dated distance is not a surveyed GPS distance."
  },
  "topology": {
    "routes": 247,
    "exact_same_order_all_7days": 170,
    "note": "Same ordered stops do not prove unchanged historical street path. Reuse of API path and road profile remains an explicit counterfactual assumption. Energy distance takes the larger of API and dated master distance."
  },
  "candidate_reassessment": {
    "test_usable_trips": 128202,
    "legacy_low_average_selected_trips": 18951,
    "legacy_low_average_upper_exceed16": 2030,
    "legacy_low_average_lower_exceed16": 1582,
    "legacy_low_average_upper_unsafe_rate": 0.10711835787029708,
    "legacy_low_average_lower_unsafe_rate": 0.083478444409266,
    "legacy_route_candidate_count": 72,
    "legacy_route_candidates_q99_over16": 63,
    "legacy_high_dependence_candidate_count": 29,
    "legacy_high_dependence_candidates_q99_over16": 25,
    "서대문01": {
      "n": 825,
      "median_upper": 15.0,
      "q95_upper": 31.0,
      "q99_upper": 36.0,
      "lower_exceed16_share": 0.3527272727272727
    },
    "관악04": {
      "n": 761,
      "median_upper": 29.0,
      "q95_upper": 53.0,
      "q99_upper": 58.0,
      "lower_exceed16_share": 0.8068331143232589
    }
  }
}
```

## AI 검증
| model | n_test | upper_coverage | mean_upper | pinball_99 | convertible_trip_share | unsafe_selected_rate | unsafe_selected_trips | selected_trips |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Historical Q99 | 128202 | 0.8792452535841875 | 26.972995039078956 | 0.5605642587479135 | 0.2423909143383098 | 0.0533547868061142 | 1658 | 31075 |
| XGBoost Q99 | 128202 | 0.9926210199528868 | 37.29840621260077 | 0.2327738698655906 | 0.0521754730815431 | 0.0022424876663178 | 15 | 6689 |
| Calibrated XGBoost Q99 | 128202 | 0.9929408277561972 | 37.5164253801072 | 0.2335519018744364 | 0.0511146471973916 | 0.0021364260643979 | 14 | 6553 |
| XGBoost without lagged tree features | 128202 | 0.9930188296594435 | 38.47317630316629 | 0.2430850675389279 | 0.0566761828988627 | 0.0024772914946325 | 18 | 7266 |
| Unscaled XGBoost Q99 | 128202 | 0.99517948237937 | 44.19987106323242 | 0.2903307676315307 | 0.0 | nan | 0 | 0 |

## 차량 배치 검증
| fleet_cap | installed_vehicles | routes | selected_duty_days | selected_trips | selected_vkm_per_day | saved_kg_per_day | electricity_saving_won_per_day | assessable_selected_trips | unsafe_assessable_trips | unsafe_rate | unassessable_selected_share | elder_boardings_per_day | annualized_electricity_only_break_even_premium_per_vehicle_won |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 1 | 7 | 180 | 109.68876575596389 | 11.926871917364366 | 5716.209881315297 | 157 | 2 | 0.0127388535031847 | 0.1277777777777777 | 4.714285850524902 | 2086416.6066800836 |
| 2 | 2 | 2 | 14 | 333 | 208.60709009546667 | 22.08589256004779 | 10585.139017516312 | 283 | 2 | 0.0070671378091872 | 0.1501501501501501 | 9.285714149475098 | 1931787.870696727 |
| 3 | 3 | 2 | 21 | 471 | 297.82753949972414 | 31.26145204658105 | 14982.723243029495 | 394 | 2 | 0.0050761421319796 | 0.1634819532908704 | 14.428571701049805 | 1822897.994568589 |
| 5 | 5 | 3 | 30 | 644 | 414.23181580425234 | 44.126497477309954 | 21148.57295821229 | 543 | 2 | 0.003683241252302 | 0.156832298136646 | 19.14285659790039 | 1543845.8259494973 |
| 10 | 5 | 3 | 30 | 644 | 414.23181580425234 | 44.126497477309954 | 21148.57295821229 | 543 | 2 | 0.003683241252302 | 0.156832298136646 | 19.14285659790039 | 1543845.8259494973 |
| 20 | 5 | 3 | 30 | 644 | 414.23181580425234 | 44.126497477309954 | 21148.57295821229 | 543 | 2 | 0.003683241252302 | 0.156832298136646 | 19.14285659790039 | 1543845.8259494973 |

## 물리·운영 조건 민감도
| scenario | capacity | installed | routes | saved_kg_per_day | electricity_saving_won_per_day | selected_trips | solver_gap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base | 16 | 5 | 3 | 44.126497477309954 | 21148.57295821229 | 644 | 0.0 |
| capacity_buffer_13 | 13 | 4 | 3 | 28.55140479867056 | 13683.874813645129 | 430 | 0.0 |
| M3_wheelchair_19 | 19 | 7 | 4 | 65.68921612759091 | 31482.969627409973 | 872 | 0.0 |
| strong_cooling | 16 | 5 | 3 | 55.07896159637837 | 26397.77694530476 | 615 | 0.0 |
| no_stop_regeneration | 16 | 5 | 3 | 46.157350561460326 | 22121.902976975955 | 644 | 0.0 |
| road_speed_minus5 | 16 | 5 | 3 | 57.03773967157849 | 27336.563465889525 | 644 | 0.0 |
| road_speed_plus5 | 16 | 5 | 3 | 37.79316275209238 | 18113.18607816553 | 644 | 0.0 |
| usable_battery_70pct | 16 | 5 | 3 | 44.126497477309954 | 21148.57295821229 | 644 | 0.0 |
| depot_8km_each_way | 16 | 5 | 3 | 49.37915304374015 | 23666.02110890972 | 644 | 0.0 |

## 구현·가정
# 마을버스 분석 업그레이드 실행·검증 안내

기존 코드와 비공개 로컬 원자료를 보존하고 `src/transit/upgrade/` 및 `data/processed/transit_v2/`에 새 분석을 구축했다. 새 보고서는 기존 7.53톤/일을 운영 가능한 확정 감축량으로 재사용하지 않는다.

## 재현

Windows / Python 3.13에서 프로젝트 루트를 작업 폴더로 사용한다.

```powershell
python -m pip install -r requirements-upgrade.txt
python -X utf8 -m src.transit.upgrade.run --download
python -X utf8 -m unittest discover -s tests -v
python -X utf8 -m src.transit.upgrade.report
```

이미 받은 카드 자료를 재사용할 때는 `--download`를 생략한다. 공공 ZIP 전체 버스 자료에서 마을버스만 추출하므로 다운로드·처리에 시간과 충분한 디스크 공간이 필요하다. 파일 이름이 아니라 내부 `STDR_DE`로 실제 자료 날짜를 확인한다. 원시 카드번호는 읽지 않으며 차량 식별자는 로컬 분석에만 남긴다. 원시 데이터와 내부 개별 운행 파일은 Git에 포함하지 않는다.

v2는 보존된 v1 파생 입력 `routes.csv`, `route_links.csv`, `route_priority.csv`, `route_hourly_load.csv`, `link_speed_weekday_profile.parquet`, `seoul_link_dimension.csv`를 사용한다. 없는 경우 기존 `src/transit/01`~`07` 파이프라인 및 `docs/TRANSIT_MAEUL_BUS.md`의 원자료 확보 절차가 선행되어야 한다. 이 입력을 제공하지 않은 코드만으로 모든 공간 분석이 즉시 재현된다고 주장하지 않는다.

## 분석 계약

- 학습: 7/6~7/19. 오차 보정: 7/20~7/26. 개발 비교: 7/27~8/2. 최종 검증: 8/3~8/9.
- 최초 원척도 Q99는 과도한 전역 상한을 생성했다. 개발 주간을 확인한 뒤 수요 규모 정규화를 추가했다. 따라서 개발 주간을 독립 검증으로 재포장하지 않고 다음 주간을 추가 확보했다.
- 정규화 척도는 이전 7일 노선·시간별 운행 피크 평균+2명이다. 같은 날 결과나 미래 날짜를 피처에 넣지 않는다. 과거 정보가 없는 경우 학습 전체 평균으로 정한 상수만 사용한다.
- XGBoost Q50/Q95/Q99를 학습한다. 분위수 교차는 단조화하고, 보정 주간의 정규화 잔차 상위 순서통계량으로 Q99를 보완한다. 시계열 의존성이 있어 무조건적인 99% 보장은 주장하지 않는다.
- 최종 검증은 날짜 순으로 수행한다. 이전 검증일의 관측 결과는 다음 날짜에 사용할 수 있는 순차 예측이다. 전체 검증 주간을 첫날 한 번에 예측한 결과가 아니다.
- 시간 단위 승하차 태그의 순서 불확실성은 같은 시각의 순변화와 승차 우선 계산의 차이로 표시한다. 누락 하차는 불확실한 승객을 상한에 더한다. 미태그 승객은 관측하지 못한다.
- 출발 시각 이전 승차, 높은 미완결 비중, 비정상 관측 운행시간, 자정을 넘어 관측이 끝나는 운행은 모델 검증 가능 표본에서 제외한다. 이전 날짜의 결과가 다음 날짜 시작 전에 확보되어 있어야 한다는 입력 가용성 경계를 적용했다. 제외 표본 비중을 함께 제시하며 전체 운행의 정원 초과율이라고 부르지 않는다.
- 트리 이력피처 제거 비교도 목표 정규화에는 과거 수요 평균을 사용한다. 따라서 모든 수요 이력을 제거한 모형이라고 해석하지 않는다.
- 인접 자료일에 같은 차량·출발시각이 나타날 수 있어 실제 출발일 기준으로 자료를 먼저 합쳐 운행 키를 중복 집계하지 않는다.
- 운행별 최대 재차와 정류장 일합계에서 얻은 평균 최대 재차를 구분한다. 기존 평균 3명 기준은 과거 분석의 후보 규칙을 재평가하는 비교 대상이다. 동일한 예측 실험의 학습 기준선으로 속이지 않는다.

## 정책 모형

소형차는 제조사 e-JEST M2 휠체어 배치의 운전자 제외 승객 정원 16명과 105kWh를 참조한다. 국내 인증·보조금 대상·노선 진입 가능 여부를 확인한 도입 확정 차종은 아니다. 운행 질량 5t, 냉방 3kW, Cd 0.5는 별도의 물리 가정이다. 제조사의 총중량을 공차중량으로 사용하지 않는다.

현재 차종을 조사한 배출량 대신 공통 7t 전기차 기준과 비교하여 순수 크기 효과를 계산한다. 카운티 303/338km를 복합/정속 두 주기로 오해했던 보정을 수정했다. 338km 정속 값에서 Cd만 맞추며, 도심 소비는 정지·출발·체류 항으로 별도 산출한다. 계량기·실차 검증은 아직 없다.

정수계획은 노선별 설치 소형차 정수와 노선·차량·날짜 운행 묶음별 교체 여부를 선택한다. 보정 기간을 이용해 노선별 차량 수를 사전 배분하고, 최종 검증 주간에는 고정 배분 안에서 하루 단위 차량을 선택한다. 기록된 차량 운행 묶음을 통째로 교체하므로 주행 중 교대와 자유로운 차고지 간 공유를 가정하지 않는다.

용량 상한, 배터리 사용 80% 중 추가 예비량 10%, 하루 충전 없는 운행, 왕복 차고지 거리 4km 가정, 보정 7일의 동일 정류장 순서, 과거 운행횟수 비교 비율 0.8~1.2, 단일 노선 운행 조건을 적용한다. 에너지 거리는 API 실제 경로와 같은 날 카드 노선 거리 중 큰 값을 쓴다. 거리 10% 차이를 노선 변경으로 해석한 최초 조건은 정류장 순서 대조 후 교체했다. 동일 정류장 순서도 실제 과거 도로 경로의 불변성을 증명하지는 않는다. 조건은 운영 가능성을 보장하는 인증이 아닌 사전 선별이다. 태그가 없는 공차·무승객 출발을 놓치므로 차량·에너지 요구량은 추가 검증이 필요하다.

동일 차량 단가 대신 차량 수 상한만 두고 일별 노선 이득이 분리되는 현재 특수 구조에서는 한계편익 탐욕법도 최적해를 얻는다. 이 강한 비교법과 MILP의 보정 기간 목적값이 일치하는지 검증한다. 노선 전체를 먼저 구매하는 단순 탐욕법과만 비교하여 MILP가 유일하게 우월하다고 주장하지 않는다. 차고지·시간 블록·충전·실제 비용 결합이 추가될 때 더 일반적인 제약 모형으로 확장할 수 있다.

구매가격을 확인하지 못한 상태에서 시장가격이나 감축비용을 만들어 넣지 않는다. 차량 수 상한을 가정한 포트폴리오와 전기요금 200원/kWh 가정의 전력비 절감, 그 절감만으로 상쇄 가능한 연간 차량당 순비용을 계산한다. 이는 운전기사·정비·차고지·보험·구매비를 포괄한 TCO가 아니다.

## 산출물 및 공개

- 검증표: `data/processed/transit_v2/model_metrics.csv`, `policy_metrics.csv`, `constraint_attrition.csv`.
- 모델: `data/processed/transit_v2/models/`.
- 그림: `output/maeul_upgrade/figures/` (PNG/SVG).
- 5쪽 PDF: `output/pdf/서울_마을버스_AI_차량정책_분석보고서_5p.pdf`.
- 편집 가능한 원문: `docs/report/maeul_bus_upgrade_report.md`.
- 상세 결과·한계: `docs/report/maeul_bus_upgrade_validation.md`.

코드·설정·실행 안내는 이 저장소에 반영되어 있다. 교통카드 원자료와 차량 식별자가 들어 있는 개별 운행 자료는 커밋하지 않으며, `python -X utf8 -m src.transit.upgrade.run --download`로 공개 원본에서 다시 받는다.


## 원자료와 기존 속도 산출물의 상세 출처


**7개 기관·9개 플랫폼의 공공데이터 23종(원자료 약 110GB)을 결합하였다.** 표의 첫 열은 각 자료의 선정 이유(용도)이다.

| 용도(선정 이유) | 활용 데이터(명) | 제공기관(명) | 출처 플랫폼(명) | URL |
| --- | --- | --- | --- | --- |
| 도로망과 관측 속도 | 국가표준노드링크, 교통소통정보 5분 단위 통행속도 | 국토교통부 | ITS 국가교통정보센터 | [its.go.kr/nodelink](https://www.its.go.kr/nodelink/nodelinkStatus?service=inquiryNodelink), [its.go.kr/opendata](https://www.its.go.kr/opendata/) |
| 속도 예측 변수(교통·시설) | 링크별 교통량·AADT, 혼잡빈도·혼잡시간, 시간대별 평균속도 | 한국교통연구원 | View-T | [viewt.ktdb.go.kr](https://viewt.ktdb.go.kr/cong/map/second_map.do) |
|  | 버스전용차로 위치, 어린이 보호구역, 험프 | 서울특별시 | 서울 열린데이터광장 | [OA-22396](https://data.seoul.go.kr/dataList/OA-22396/F/1/datasetView.do), [OA-2799](https://data.seoul.go.kr/dataList/OA-2799/F/1/datasetView.do), [OA-15541](https://data.seoul.go.kr/dataList/OA-15541/A/1/datasetView.do) |
|  | 보행자 교통사고 다발지역 | 한국도로교통공단 | 교통사고정보 개방시스템 | [opendata.koroad.or.kr](https://opendata.koroad.or.kr/api/selectPedstriansDataSet.do) |
| 속도 예측 변수(기상·지형) | HSR 1시간 누적 레이더 강수 | 기상청 | 기상청 API허브 | [apihub.kma.go.kr](https://apihub.kma.go.kr/apiList.do?seqApi=5) |
|  | 수치표고모델(DEM), 서울 행정경계 | 국토지리정보원, 국토교통부 | 국토정보플랫폼, 브이월드 | [map.ngii.go.kr](https://map.ngii.go.kr/), [vworld.kr](https://www.vworld.kr/) |
| 노선 경로·승객·운행 | 버스노선별 정류소, 정류장별 시간대별 승하차, 노선별 정류장별 운행횟수 | 서울특별시, TOPIS | 서울 열린데이터광장 | [OA-1095](https://data.seoul.go.kr/dataList/OA-1095/L/1/datasetView.do), [OA-12913](https://data.seoul.go.kr/dataList/OA-12913/F/1/datasetView.do), [OA-21220](https://data.seoul.go.kr/dataList/OA-21220/S/1/datasetView.do) |
|  | 버스운행정보 API 실제 경로(getRoutePath), 노선유형별 통행속도 | 서울특별시 | 서울 버스정보시스템 | [ws.bus.go.kr](http://ws.bus.go.kr/api/rest/busRouteInfo/getRoutePath), [bus.go.kr](https://bus.go.kr/) |
| 고령 인구·대체수단 | 생활인구(250m 격자, 행정동), 역사마스터 | 서울특별시·KT | 서울 열린데이터광장 | [생활인구](https://data.seoul.go.kr/dataVisual/seoul/seoulLivingPopulation.do), [OA-21232](https://data.seoul.go.kr/dataList/OA-21232/S/1/datasetView.do) |
| 차량 구성 | 자치구·행정동별 연료별 자동차 등록, 마을버스 운행·연료별 현황 | 서울특별시 | 서울 열린데이터광장, 서울시 교통 누리집 | [OA-15640](https://data.seoul.go.kr/dataList/OA-15640/F/1/datasetView.do), [OA-21236](https://data.seoul.go.kr/dataList/OA-21236/F/1/datasetView.do), [news.seoul.go.kr](https://news.seoul.go.kr/traffic/archives/300) |

자료 기간은 도로 속도·강수 2026년 7월 7~20일, 승하차·생활인구·차량 등록 7월, 운행횟수 7월 6~31일, 정류소 9월 2일, 버스 속도 점검 10월 6~8일이다.



## 연구·정책 출처
1. [서울시 (2025.12.21). 마을버스 서비스 개선 합의.](https://www.seoul.go.kr/news/news_report.do?nttNo=448993)
2. [김성아 외 (2023). 서울시 고령자의 대중교통 이용에 영향을 미치는 도시특성 분석. 국토지리학회지 57(4), 377-390.](https://doi.org/10.22905/kaopqj.2023.57.4.2)
3. [김명환·이재현 (2026). 교통약자 수요 기반 저상버스 배치 전략을 통한 이용 형평성 개선효과 분석. 대한지리학회지 61(3), 351-370.](https://doi.org/10.22776/kgs.2026.61.3.351)
4. [Romano et al. (2019). Conformalized Quantile Regression.](https://arxiv.org/abs/1905.03222)
5. [Nakao et al. (2026). Joint optimization of charging infrastructure and fleet mix for CO₂-constrained feeder services. Applied Energy 405, 127216.](https://doi.org/10.1016/j.apenergy.2025.127216)
6. [현대자동차. 카운티 일렉트릭 제조사 제원·정속 주행거리.](https://www.hyundai.com/kr/ko/c/products/bus/county-electric)
7. [Karsan. e-JEST 제원·휠체어 배치별 정원.](https://www.karsan.com/en/electric-bus/e-jest/e-jest-specs)
8. [기후에너지환경부 (2025.12.18). 2023년 전력배출계수 0.4173tCO₂eq/MWh 공표.](https://niwdc.mcee.go.kr/home/web/board/read.do?boardId=1829260&boardMasterId=939&menuId=10598)
9. [XGBoost. Quantile regression 공식 예제.](https://xgboost.readthedocs.io/en/stable/python/examples/quantile_regression.html)
10. [SciPy. milp 공식 문서.](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html)
11. [인사혁신처 (2026.4.29). 제헌절 공휴일 재지정.](https://www.mpm.go.kr/mpm/comm/newsPress/newsPressRelease/?boardId=bbs_0000000000000029&cntId=4250&mode=view)