# 통합 재계산과 보고서 수치 재현

[분석보고서](report/maeul_bus_integrated_report.md)의 탄소·유형·접근성 수치는 기본 분석(01~10)의 입력을 그대로 쓰되, 버스 에너지를 수정한 모형으로 다시 계산한 `data/processed/transit_full/`에서 나온다. 운행별 피크·차량 배치 수치는 [운행별 피크 예측과 차량 배치](MAEUL_UPGRADE_IMPLEMENTATION.md)의 `data/processed/transit_v2/`에서 나온다.

## 실행

기본 분석(01~10)과 운행 단계(`upgrade.run`)가 끝난 상태에서 실행한다.

```powershell
python -X utf8 -m src.transit.upgrade.integrated             # 탄소·유형·운영·접근성을 transit_full에 다시 계산
python -X utf8 -m src.transit.upgrade.integrated_figures     # 보고서 그림 1~4(docs/report/images/integrated)

python src/transit/checks/report_numbers.py                  # 보고서 수치 출력
python src/transit/checks/assumption_sensitivity.py          # 탑승인원·배출 배율·대체수단 기준 민감도
python src/transit/checks/ai_speed_stress.py                 # AI 추정 구간 속도 ±5km/h
python src/transit/checks/cluster_check.py                   # 군집분석과 네 유형 비교
```

`integrated_report`는 같은 수치로 5쪽 초안 PDF를 `output/pdf/`에 만든다(reportlab·PyMuPDF 필요). 제출본은 이 초안을 공모전 양식으로 옮겨 방법·정책 서술을 다듬은 것이다.

## integrated가 하는 일

1. 기본 분석 입력(`routes.csv`, `route_links.csv`, `route_demand.csv`, `link_speed_weekday_profile.parquet` 등 10개)을 `transit_full/`로 복사하고 원본의 SHA-256을 기록한다. 실행 끝에 원본이 바뀌지 않았는지 다시 확인한다.
2. 7~8월 공식 운행횟수와 `upgrade/energy`의 시간대별 버스 에너지(07~21시 평균)를 넣어 05(탄소·손익분기), 06의 유형 규칙, 07(시간대·운영), 10(노선 제거 접근성)을 다시 실행한다.
3. 전기 승용차에도 버스와 같은 충전 효율 0.9를 적용해 승용차 배출을 다시 계산한다.
4. 요약(`integrated_summary.json`)과 운행 피크 검증과의 연결(`integrated_findings.json`), 승용차 탑승인원 × 버스 배출 × 승용차 배출 18조건 민감도(`carbon_sensitivity.csv`)를 남긴다.

## 기본 분석 결과와 달라진 이유

- **버스 에너지 보정:** 카운티 일렉트릭의 303km는 마을버스 사양과 다른 사양의 정속 주행거리인데, 이전에는 복합 주행거리로 읽어 가감속 부가 전력을 역산했다. 지금은 마을버스 사양 정속 338km로 공기저항만 보정하고, 정류장 정지·출발·체류 에너지를 가정으로 따로 더한다.
- **충전 손실:** 버스와 전기 승용차 모두 전력망 기준(충전 효율 0.9)으로 맞췄다.
- **접근성:** 노선 제거로 정류장 연결을 새로 잃은 격자를 영향 0으로 처리하던 부분을 고쳐 영향 인구에 포함했다(`elderly_disconnected_new`). 평균 증가 시간은 접근시간이 유한한 격자로만 계산한다.

그 결과 버스 우위 노선은 218개에서 223개로, 전면 전기화 감축은 하루 15.54톤에서 14.33톤으로, 평균 3명 미만 시간대 소형화는 7.53톤에서 5.58톤으로 바뀌었다. 기본 경로(`transit/`)로 01~10만 실행한 결과는 정지·출발 에너지가 빠진 정속 근사이므로 보고서 수치와 다르다.

## 두 분석의 모집단

| 분석 | 모집단·기간 | 답하는 질문 |
| --- | --- | --- |
| 탄소·의존도·정책 유형·노선 제거 | 7월 집계, 247개 중 승객 누적 이상 17개를 뺀 230개 노선 | 노선이 승용차보다 효율적인가, 노선을 줄이면 고령자 접근성은 얼마나 나빠지는가 |
| 운행 피크·AI·차량 배치 | 교통카드 7/6~8/9, 247개 노선, 최종 검증 8/3~9 | 평균 기준 후보를 실제 운행 피크와 운영 조건으로 보면 어디까지 남는가 |

두 분석은 기간과 단위가 달라 감축량을 서로 비교하거나 더하지 않는다. 전기화와 소형화 시나리오도 대상이 겹치므로 합산하지 않는다.

## 해석상 유의점

- 탄소 비교는 노선별 실제 차량이 아니라 공통 7t 전기 마을버스와 서울 연료 구성을 적용한 시나리오다. 실제 대형 비중은 버스 배출 1.28·1.5배 민감도로 본다.
- 정지·출발·체류·냉방·회생은 가정이며 실차 기록으로 보정하지 않았다.
- 노선 제거 영향은 가장 가까운 남은 정류장까지의 보행시간 변화다. 목적지·환승·대기 부담은 포함하지 않는다. 도로망과 직선×1.3 결과는 계산 방식의 범위이며 신뢰구간이 아니다.
- '새로 10분 초과'는 제거 후 전체 접근시간이 10분을 넘는다는 뜻이며 추가 보행이 10분이라는 뜻이 아니다. 영향 인구는 정류장 주변 고령 생활인구로, 실제 승객 수가 아니고 노선끼리 합산할 수 없다.
- 도로 속도는 저장된 A·B·C 모델 산출물을 그대로 쓴다. 통합 단계에서 속도 모델을 다시 학습하지 않는다.

## 산출물

| 위치 | 내용 |
| --- | --- |
| `data/processed/transit_full/` | 재계산한 노선별 탄소·유형·운영·접근성 표와 요약(로컬) |
| `docs/report/metrics/integrated/` | 공개용 재계산표: `route_priority.csv`, `route_removal_access.csv`, `carbon_sensitivity.csv`, 요약 JSON |
| `docs/report/images/integrated/` | 보고서 그림 1~4(PNG·SVG) |
| `docs/report/작다고감편하지마라_분석보고서.pdf` | 제출본 |
