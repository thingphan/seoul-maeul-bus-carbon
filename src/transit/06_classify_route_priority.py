"""탄소 효율 × 교통약자 의존도로 마을버스 노선을 네 유형으로 나누고 유형별 정책을 붙인다.

입력 (data/processed/transit/)
  route_demand.csv(02), route_carbon_breakeven.csv(05)
  link_speed_weekday_profile.parquet, route_links.csv (소형 차량 시나리오 계산용)
  분류 규칙은 route_classes.py(09의 민감도 분석과 공용)

처리
  1. 교통약자 의존도: 세 구성요소의 노선 간 백분위 순위 평균
       catchment_elderly_share  정류소 300m 안 평일 낮 고령 비율
       catchment_elderly_day    같은 범위 고령 생활인구 수
       no_alt_boarding_share    지하철역 500m·일반버스 200m 밖 정류소의 승차 비중(대체수단 없음)
  2. 탄소 효율: 05의 승객 1명·1km당 CO2(bus_g_per_pkm). 거의 모든 노선이 휘발유 승용차보다
     낮으므로 노선 간 상대 비교로 나눈다(중앙값보다 크면 '손님에 비해 차가 큰' 노선).
  3. 네 유형
       1 증편·지원 우선      의존 높음 × 효율 높음  → 탄소와 형평성이 같은 방향
       2 소형 차량 전환      의존 높음 × 효율 낮음  → 노선은 지키고 차를 작게
       3 현행 유지           의존 낮음 × 효율 높음
       4 노선 조정 검토      의존 낮음 × 효율 낮음
  4. 소형 차량 시나리오(② 유형): 15인승급 소형 전기 승합차(질량 3.5t, 공기저항 면적 CdA 1.4m²,
     냉방 보조전력 3kW 가정)로 같은 노선·운행횟수를 돌릴 때 CO2를 다시 계산한다. 정지·출발 부가 전력은
     카운티 일렉트릭(인증 질량 6t)의 공인 복합 주행거리로 역산한 값(src/energy/ev_calibration.py)을
     질량 비(3.5/6)로 줄여 넣는다(가감속 에너지는 질량에 비례). 마을버스와 같은 기준으로 비교하기 위함이다. 차량 크기 효과만 보도록
     05의 전기 마을버스 값(bus_ev_g_per_km)과 비교한다(전기 전환 효과는 05에서 따로 낸다). 1회 운행 평균 최대
     재차인원이 8명(15인승의 약 절반) 이하인 노선만 전환 가능으로 본다. 감축량은 크기 효과만(전기 마을버스 대비,
     van_co2_cut_kg_per_day)과 지금 차량 구성 대비(van_co2_cut_vs_current_kg_per_day) 두 가지로 낸다.

출력
  data/processed/transit/route_priority.csv

사용: python src/transit/06_classify_route_priority.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from energy.ev_calibration import calibrate_county_electric  # noqa: E402
from energy.link_energy import VTCPEMParameters, calculate_link_energy  # noqa: E402
from route_classes import DEPENDENCE, assign_classes  # noqa: E402
from route_data import load_route_link_hours  # noqa: E402

TRANSIT = PROJECT_ROOT / "data" / "processed" / "transit"
GRID_KG_CO2_PER_KWH = 0.4173
SMALL_VAN = replace(VTCPEMParameters(), mass_kg=3500.0, drag_coefficient=0.35, frontal_area_m2=4.0, auxiliary_power_w=3000.0)
SMALL_VAN_MAX_PEAK_LOAD = 8.0


def small_van_parameters() -> VTCPEMParameters:
    """카운티 일렉트릭의 정지·출발 부가 전력을 질량 비로 옮겨 소형 승합차 보조전력에 더한다."""
    county, extra_w = calibrate_county_electric()
    van_extra_w = extra_w * SMALL_VAN.mass_kg / county.mass_kg
    return replace(SMALL_VAN, auxiliary_power_w=SMALL_VAN.auxiliary_power_w + van_extra_w)


def small_van_kwh_per_km(route_ids: set[str], van: VTCPEMParameters) -> pd.Series:
    frame = load_route_link_hours()
    frame = frame[frame["ROUTE_ID"].isin(route_ids)].copy()
    frame["wh"] = calculate_link_energy(frame[["LINK_ID", "LENGTH", "speed_kmh", "slope_pct"]], van)["energy_ev_wh"].to_numpy()
    per_hour = frame.groupby(["ROUTE_ID", "시간코드"]).agg(wh=("wh", "sum"), m=("LENGTH", "sum"))
    return (per_hour["wh"] / per_hour["m"]).groupby("ROUTE_ID").mean().rename("van_kwh_per_km")


def main() -> None:
    demand = pd.read_csv(TRANSIT / "route_demand.csv", dtype={"ROUTE_ID": str})
    carbon = pd.read_csv(TRANSIT / "route_carbon_breakeven.csv", dtype={"ROUTE_ID": str})
    carbon = carbon[~carbon["load_qa_flag"]]
    data = assign_classes(carbon.merge(demand[["ROUTE_ID", *DEPENDENCE]], on="ROUTE_ID", how="inner"))

    # 소형 차량 전환(② 유형) 효과
    data["avg_peak_load_per_trip"] = data["peak_onboard"] / data["trips_per_day"]
    van = small_van_kwh_per_km(set(data["ROUTE_ID"]), small_van_parameters())
    data = data.join(van, on="ROUTE_ID")
    data["van_g_per_km"] = data["van_kwh_per_km"] * GRID_KG_CO2_PER_KWH * 1000.0
    data["van_feasible"] = data["avg_peak_load_per_trip"] <= SMALL_VAN_MAX_PEAK_LOAD
    vkm = data["trips_per_day"] * data["length_km"]
    convert = (data["class_id"] == 2) & data["van_feasible"]
    data["van_co2_cut_kg_per_day"] = np.where(convert, (data["bus_ev_g_per_km"] - data["van_g_per_km"]) * vkm / 1000.0, 0.0)
    data["van_co2_cut_vs_current_kg_per_day"] = np.where(
        convert, (data["bus_g_per_km"] - data["van_g_per_km"]) * vkm / 1000.0, 0.0)
    data = data.sort_values(["class_id", "dependence_score"], ascending=[True, False])
    data.to_csv(TRANSIT / "route_priority.csv", index=False, encoding="utf-8-sig")

    print("네 유형 분류")
    summary = data.groupby("class").agg(
        노선수=("ROUTE_ID", "size"), 하루승차=("boardings_per_day", "sum"),
        평균승객=("avg_load", "median"), 승객1명km당_gCO2=("bus_g_per_pkm", "median"),
        고령비율=("catchment_elderly_share", "median"), 하루CO2감축_t=("co2_saved_kg_per_day", lambda s: s.sum() / 1000))
    print(summary.round(2).to_string())
    two = data[data["class_id"] == 2]
    print(f"\n소형 차량 전환 유형 {len(two)}개 중 전환 가능(1회 평균 최대 승객 {SMALL_VAN_MAX_PEAK_LOAD:.0f}명 이하) {int(two['van_feasible'].sum())}개")
    print(f"  1km당 CO2(같은 전기차끼리): 전기 마을버스 {two['bus_ev_g_per_km'].median():.0f}g → 소형 전기 승합차 {two['van_g_per_km'].median():.0f}g")
    print(f"  전환 시 감축: 하루 {two['van_co2_cut_vs_current_kg_per_day'].sum() / 1000:.2f}t(지금 차량 대비), "
          f"그중 크기 효과 {two['van_co2_cut_kg_per_day'].sum() / 1000:.2f}t(전기 마을버스 대비)")
    cols = ["노선명", "boardings_per_day", "avg_load", "bus_g_per_pkm", "catchment_elderly_share", "no_alt_boarding_share",
            "avg_peak_load_per_trip", "van_feasible"]
    print("\n소형 차량 전환 유형 상위(의존도 순) 10개:")
    print(two[cols].head(10).round(2).to_string(index=False))
    print("\n증편·지원 우선 유형 상위 10개:")
    print(data[data["class_id"] == 1][cols[:6]].head(10).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
