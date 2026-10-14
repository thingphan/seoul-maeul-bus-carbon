# 데이터

원자료와 분석 산출물은 크고 재생성 가능하므로 Git에 포함하지 않는다(예외: `processed/`의 속도 모델 조회표 두 개). 아래 표의 출처에서 내려받아 **같은 경로와 파일명**으로 두면 코드가 그대로 읽는다.

| 폴더 | 의미 |
| --- | --- |
| `raw/` | 내려받은 원본(수정하지 않음) |
| `interim/` | 파이프라인 중간 결과(코드로 재생성) |
| `processed/` | 모델·분석이 읽는 산출물(코드로 재생성) |

## 1. 원자료(`raw/`)

### 도로망·지형·기상(속도 모델용)

| 경로 | 데이터 | 출처 |
| --- | --- | --- |
| `raw/road/MOCT_LINK.*`, `MOCT_NODE.*`, `TURNINFO.dbf` | 전국 표준노드링크 | [ITS 국가교통정보센터](https://www.its.go.kr/nodelink/nodelinkStatus?service=inquiryNodelink) |
| `raw/boundary/seoul_admin_boundary.geojson` | 서울 자치구 경계(EPSG:4326) | [브이월드](https://www.vworld.kr/) 행정구역 경계에서 서울 25개 구 추출 |
| `raw/traffic/202607_5Min/*_5Min.csv` | 교통소통정보 5분 단위 통행속도(2026년 7월) | [ITS 국가교통정보센터 오픈데이터](https://www.its.go.kr/opendata/) |
| `raw/dem/*.img` | 수치표고모델(서울 도엽) | [국토정보플랫폼](https://map.ngii.go.kr/) |
| `raw/weather/grid/hsr_lon.txt`, `hsr_lat.txt` | HSR 레이더 격자 위·경도 | 기상청 API허브(`src/pipeline/03 --grid`) |
| `raw/weather/radar/hsr_pcph_acc60_{YYYYMMDDHHMM}.txt` | HSR 기반 1시간 누적 강수 | [기상청 API허브](https://apihub.kma.go.kr/apiList.do?seqApi=5)(`src/pipeline/03`, 키는 환경변수 `KMA_API_KEY`) |
| `raw/traffic/bus_lane/bus_lane.shp` | 서울시 버스전용차로 위치 | [서울 열린데이터광장 OA-22396](https://data.seoul.go.kr/dataList/OA-22396/F/1/datasetView.do) |
| `raw/traffic/school_zone/school_zone.shp` | 서울시 어린이 보호구역 | [서울 열린데이터광장 OA-2799](https://data.seoul.go.kr/dataList/OA-2799/F/1/datasetView.do) |
| `raw/traffic/speed_bump/speed_bump.shp` | 과속방지턱(험프) | [서울 열린데이터광장 OA-15541](https://data.seoul.go.kr/dataList/OA-15541/A/1/datasetView.do) |
| `raw/traffic/pedestrian_accident_2019_2024.csv` | 보행자 교통사고 다발지역 | [교통사고정보 개방시스템](https://opendata.koroad.or.kr/api/selectPedstriansDataSet.do) |
| `raw/traffic/traffic_volume.xlsx` | 링크별 교통량·AADT | [한국교통연구원 View-T](https://viewt.ktdb.go.kr/cong/map/second_map.do) |
| `raw/traffic/AverageSpeed(LINK).xlsx`, `CongestIndex(LINK).xlsx` | 시간대별 평균속도, 혼잡빈도·혼잡시간 | 〃 |
| `raw/traffic/250_LOCAL_RESD_202607/`, `250_LOCAL_RESD_ADMDONG_202607/` | 생활인구(250m 격자·행정동, 2026년 7월 일별) | [서울 열린데이터광장 생활인구](https://data.seoul.go.kr/dataVisual/seoul/seoulLivingPopulation.do) |
| `raw/traffic/match/match.shp` | 250m 격자 도형 | 〃(격자 경계 파일) |

### 마을버스 분석용

| 경로 | 데이터 | 출처 |
| --- | --- | --- |
| `raw/traffic/서울시버스노선별정류소정보(20260902).xlsx` | 노선별 정류장 순서·좌표 | [서울 열린데이터광장 OA-1095](https://data.seoul.go.kr/dataList/OA-1095/L/1/datasetView.do) |
| `raw/traffic/bus_route_path/route_paths.csv` | 노선 실제 경로 좌표 | 서울시 버스운행정보 API getRoutePath(`src/transit/00`, 키는 환경변수 `SEOUL_BUS_API_KEY`) |
| `raw/traffic/2026년_버스노선별_정류장별_시간대별_승하차_인원_정보(07월).csv` | 정류장별 시간대별 승하차 | [서울 열린데이터광장 OA-12913](https://data.seoul.go.kr/dataList/OA-12913/F/1/datasetView.do) |
| `raw/traffic/bus_trips/tpss_sta_route_hturn_*.csv` | 노선별 정류장별 운행횟수(2026년 7월) | [서울 열린데이터광장 OA-21220](https://data.seoul.go.kr/dataList/OA-21220/S/1/datasetView.do) |
| `raw/traffic/서울시 역사마스터 정보.csv` | 지하철역 좌표 | [서울 열린데이터광장 OA-21232](https://data.seoul.go.kr/dataList/OA-21232/S/1/datasetView.do) |
| `raw/traffic/vehicle_registration/seoul_fuel_registration_202607.csv` | 서울 연료별 자동차 등록(2026년 7월) | [서울 열린데이터광장 OA-15640](https://data.seoul.go.kr/dataList/OA-15640/F/1/datasetView.do) |
| `raw/traffic/서울시 자치구 읍면동별 연료별 자동차 등록현황(행정동)(26년7월).csv` | 행정동별 연료별 자동차 등록 | [서울 열린데이터광장 OA-21236](https://data.seoul.go.kr/dataList/OA-21236/F/1/datasetView.do) |
| `raw/traffic/통행속도 노선 통계.csv` | 노선유형별 시간대 버스 통행속도(2026.10.6~8) | [서울 버스정보시스템](https://bus.go.kr/) |

마을버스 연료별 대수(2026년 1월)와 재정지원 기준액은 [서울시 교통 누리집](https://news.seoul.go.kr/traffic/archives/300) 자료를 코드 상수로 넣었다(`src/transit/05`, `07`).

## 2. 중간·분석 산출물

| 경로 | 만드는 곳 | 내용 |
| --- | --- | --- |
| `interim/seoul_link.shp`, `seoul_link_attr.csv` | `pipeline/01` | 서울 링크 63,548개 |
| `interim/seoul_link_speed_5min_hourly.csv` | `pipeline/02` | 링크×날짜×시간 관측속도와 표본 수 |
| `processed/seoul_link_dimension.csv` | `pipeline/04` | 링크 차원 표(도로·경사·위상·시설·교통·인구) |
| `processed/seoul_link_speed_fact.csv` | `pipeline/05` | 관측속도 팩트(15,094,335행) |
| `processed/model_a_lookup.parquet` | `models/01` | Model A 조회표(Git 포함) |
| `processed/neighbor_speed_lookup.parquet` | `models/02` | 이웃속도 조회표(Git 포함) |
| `processed/all_link_speed_prediction_{tm}.csv` | `models/04` | 시점별 전 링크 속도(335개) |
| `processed/transit/` | `transit/01~11` | 마을버스 분석 결과([docs/TRANSIT_MAEUL_BUS.md](../docs/TRANSIT_MAEUL_BUS.md)) |

학습된 B·C 모델(`src/models/artifacts/model_nb.pkl`, `model_stat.pkl`)도 Git에 포함되어 있어, 원자료 없이도 저장된 모델로 예측 단계를 실행할 수 있다.
