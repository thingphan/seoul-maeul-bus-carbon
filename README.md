# 마을버스는 언제 친환경인가

**AI로 보완한 골목길 속도에 기반한 서울 마을버스 230개 노선의 탄소 효율과 고령자 이동권 분석**

AI와 함께하는 교통문제 해결을 위한 데이터 분석 공모전 · 팀 덜컹덜컹달려간다마을버스야(전예준, 김명관)

![분석 결과 요약](docs/report/images/fig00_key_numbers.png)

- **대부분 친환경이지만 조건부다.** 승객 1명이 1km를 이동할 때 CO₂는 마을버스 98g, 승용차 307g(중앙값)으로 230개 노선 중 218개에서 버스가 적었다. 다만 저녁 20~22시에는 운행 4회 중 1회의 재차인원이 3명 미만이었다.
- **연료보다 크기와 시간.** 재차인원 3명 미만인 운행(전체의 15.9%)만 소형 전기차로 바꾸어도 경유·CNG 1,229대 전면 전기화 효과의 48.5%(하루 7.53톤)를 얻는다.
- **고령자 노선은 줄이지 말고 작게.** 고령자 의존도가 높은 노선을 하나 폐지하면 정류장까지 더 걸어야 하는 고령 인구가 의존도가 낮은 노선의 약 2배다.
- **AI가 골목을 채웠다.** 속도 관측이 없는 마을버스 구간(노선 길이의 13%, 95%가 1차로 골목)을 XGBoost로 추정했고, 추정 속도를 ±5km/h 바꾸어도 결론은 유지됐다.

보고서 본문: [docs/report/maeul_bus_report.md](docs/report/maeul_bus_report.md)

## 분석 흐름

```text
공공데이터 23종(약 110GB)
  ├─ 도로·속도·기상·시설·교통·인구 ─→ src/pipeline   링크 차원 표 + 관측속도 팩트
  │                                  └→ src/models     AI 속도 모델(A 이력 조회 / B·C XGBoost)
  │                                                    → 서울 전 링크 63,548개의 시간대별 속도
  ├─ 노선·승하차·운행·인구·차량 ────→ src/transit    노선 맵매칭 → 재차인원 → 버스·승용차 탄소(src/energy)
  │                                                    → 네 유형 분류 → 시간대·차량 크기 시나리오
  │                                                    → 노선 제거 시 고령자 보행 접근성
  └─────────────────────────────────→ src/transit/checks  보고서 수치 재현과 강건성 점검
```

## 저장소 구조

| 경로 | 내용 |
| --- | --- |
| `src/pipeline/` | 01~05 도로 속도 데이터 구축(서울 링크 추출, 5분 속도 집계, 레이더 강수, 링크 차원 표, 관측속도 팩트) |
| `src/models/` | 01~04 AI 속도 모델 학습·검증·예측, `artifacts/`에 학습된 B·C 모델 |
| `src/energy/` | 차량별 에너지·배출 모델(VT-CPEM, VT-CPFM 계열, HEV, 경유·CNG 버스 환산) |
| `src/transit/` | 00~11 마을버스 분석, `checks/` 검증 스크립트, `report_figures.py` 보고서 그림 |
| `notebooks/` | 속도 모델 기준모형 비교, 변수 실험, Model A 시간 분할 검증, B/C 분리 평가 |
| `tests/` | 에너지 모델 단위 테스트 |
| `docs/` | [속도 모델](docs/SPEED_MODEL.md), [마을버스 분석](docs/TRANSIT_MAEUL_BUS.md), [보고서](docs/report/maeul_bus_report.md) |
| `data/` | 폴더 구조와 [데이터 안내](data/README.md)(원자료는 포함하지 않음) |

## 실행

Python 3.13에서 검증했다.

```powershell
pip install -r requirements.txt
python -m unittest tests.test_link_energy        # 에너지 모델 테스트(10개)
```

원자료는 Git에 없다. [data/README.md](data/README.md)의 출처에서 내려받아 같은 경로에 둔다.

### 1) 마을버스 분석(보고서 결과 재현)

전 링크 속도 예측 결과(`data/processed/all_link_speed_prediction_*.csv`)와 마을버스 원자료가 있으면 아래만 실행하면 된다(캐시가 있으면 약 2~3분).

```powershell
python src/transit/01_build_bus_route_links.py
python src/transit/02_build_route_demand.py
python src/transit/03_build_route_feasibility.py
python src/transit/04_build_route_energy.py
python src/transit/05_build_route_carbon_breakeven.py
python src/transit/06_classify_route_priority.py
python src/transit/07_build_route_operations.py
python src/transit/08_route_typology.py
python src/transit/09_access_gap.py
python src/transit/10_validate_ai_speed.py
python src/transit/11_route_removal_access.py

python src/transit/checks/report_numbers.py          # 보고서 수치 재현
python src/transit/checks/assumption_sensitivity.py  # 가정 민감도
python src/transit/checks/ai_speed_stress.py         # AI 속도 ±5km/h
python src/transit/checks/cluster_check.py           # 군집분석 점검
```

### 2) AI 속도 예측(학습된 모델 사용)

학습된 모델과 조회표가 저장소에 있으므로 재학습 없이 예측할 수 있다.

```powershell
python src/models/03_predict_all_link_speed.py --tm 202607160800                       # 한 시점
python src/models/04_batch_predict_all_link_speed.py --start-date 20260707 --end-date 20260720   # 기간 전체
```

### 3) 원자료부터 전체 재구축

```powershell
python src/pipeline/01_extract_seoul_links.py
python src/pipeline/02_aggregate_5min_speed.py
python src/pipeline/03_fetch_kma_radar.py --from-5min-times     # 환경변수 KMA_API_KEY 필요
python src/pipeline/04_build_link_dimension.py
python src/pipeline/05_build_speed_fact.py

python src/models/01_build_model_a.py
python src/models/02_build_model_bc.py --mode validate          # 도로 단위 GroupKFold 검증
python src/models/02_build_model_bc.py --mode final             # 최종 학습
python src/models/04_batch_predict_all_link_speed.py --start-date 20260707 --end-date 20260720

python src/transit/checks/validate_bc_tiers.py                  # B·C 계층별 성능(약 1시간)
```

이후 1)의 마을버스 분석을 실행한다. 노선 실제 경로를 다시 받으려면 `src/transit/00_fetch_bus_route_paths.py`를 실행한다(환경변수 `SEOUL_BUS_API_KEY` 필요).

### API 인증키

인증키는 코드와 저장소에 넣지 않고 환경변수로만 읽는다.

```powershell
$env:SEOUL_BUS_API_KEY = "<서울시 버스운행정보 API 키>"
$env:KMA_API_KEY = "<기상청 API허브 키>"
```

## 데이터 출처

국가교통정보센터(표준노드링크·통행속도), 국토지리정보원(DEM), 기상청(레이더 강수), 서울 열린데이터광장(버스 정류소·승하차·운행횟수·생활인구·자동차 등록·역사·시설), 한국교통연구원(View-T), 한국도로교통공단(사고다발지역), 서울 버스정보시스템(노선 경로·통행속도). 상세 목록은 [data/README.md](data/README.md).
