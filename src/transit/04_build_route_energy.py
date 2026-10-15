"""전기 중형 마을버스의 노선별 운행 에너지와 배터리 여유를 계산한다.

기준 차량: 카운티 마을버스 사양 128kWh, 73km/h 정속 338km.
303km는 다른 사양의 정속 값으로 복합 보정에 사용할 수 없다.
이 v1 링크 계산에는 가감속이 포함되지 않으며, v2의 명시적 정지·출발 시나리오와 구분한다.
에너지식은 src/energy/link_energy.py의 VT-CPEM 링크 근사식을 그대로 쓴다(정속, 가속도 0,
회생제동 미적용). 가감속·체류 등의 누락 때문에 배터리 용량의 보장 상한으로 해석하지 않는다.

입력
  data/processed/transit/route_links.csv, routes.csv, link_speed_weekday_profile.parquet (01, 03 산출물)
  1km당 에너지는 매칭 링크열(속도·경사 표본)에서 구하고, 1회 운행 에너지는 여기에 실제 경로
  길이(routes.csv의 length_km)를 곱해 얻는다.
  data/processed/seoul_link_dimension.csv
출력
  data/processed/transit/route_energy.csv  노선×시나리오: 1회 운행 kWh, kWh/km, 경사로 인한 추가 비중,
                                           충전 1회당 운행 횟수(가용 배터리 80% 가정)

사용: python src/transit/04_build_route_energy.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from energy.ev_calibration import COUNTY_BATTERY_KWH, calibrate_county_electric  # noqa: E402
from energy.link_energy import VTCPEMParameters, calculate_link_energy  # noqa: E402
from route_data import load_route_link_hours  # noqa: E402

TRANSIT = PROJECT_ROOT / "data" / "processed" / "transit"
USABLE_SHARE = 0.8

MASS_SCENARIOS_KG = {"6t": 6000.0, "7t": 7000.0, "8t": 8000.0}
AUX_SCENARIOS_W = {"냉방약(2kW)": 2000.0, "냉방중(5kW)": 5000.0, "냉방강(8kW)": 8000.0}
BASE_SCENARIO = ("7t", "냉방중(5kW)")


def main() -> None:
    county, extra_w = calibrate_county_electric()
    print(f"[1/3] 보정값 Cd = {county.drag_coefficient:.3f} (전면적 {county.frontal_area_m2}m²), "
          f"가감속 부가 전력 {extra_w / 1000:.2f}kW")

    frame = load_route_link_hours()
    official_km = pd.read_csv(TRANSIT / "routes.csv", dtype={"ROUTE_ID": str}).set_index("ROUTE_ID")["length_km"]

    print("[2/3] 노선×시간×시나리오 에너지 계산...")
    energy_input = frame[["LINK_ID", "LENGTH", "speed_kmh", "slope_pct"]]
    flat_input = energy_input.assign(slope_pct=0.0)

    rows = []
    for mass_name, mass in MASS_SCENARIOS_KG.items():
        for aux_name, aux in AUX_SCENARIOS_W.items():
            params = replace(VTCPEMParameters(), mass_kg=mass, drag_coefficient=county.drag_coefficient,
                             frontal_area_m2=county.frontal_area_m2, auxiliary_power_w=aux + extra_w)
            frame["wh"] = calculate_link_energy(energy_input, params)["energy_ev_wh"].to_numpy()
            frame["wh_flat"] = calculate_link_energy(flat_input, params)["energy_ev_wh"].to_numpy()
            per_hour = frame.groupby(["ROUTE_ID", "노선명", "kind", "시간코드"], as_index=False).agg(
                wh=("wh", "sum"), wh_flat=("wh_flat", "sum"), length_m=("LENGTH", "sum"))
            per_route = per_hour.groupby(["ROUTE_ID", "노선명", "kind"], as_index=False).agg(
                trip_kwh_mean=("wh", lambda s: s.mean() / 1000), trip_kwh_max=("wh", lambda s: s.max() / 1000),
                flat_kwh_mean=("wh_flat", lambda s: s.mean() / 1000), length_km=("length_m", lambda s: s.iloc[0] / 1000))
            # 링크열은 1km당 에너지 표본이고, 1회 운행 길이는 실제 경로선 길이(01)를 쓴다.
            scale = per_route["ROUTE_ID"].map(official_km) / per_route["length_km"]
            for col in ("trip_kwh_mean", "trip_kwh_max", "flat_kwh_mean"):
                per_route[col] = per_route[col] * scale
            per_route["length_km"] = per_route["ROUTE_ID"].map(official_km)
            per_route["kwh_per_km"] = per_route["trip_kwh_mean"] / per_route["length_km"]
            per_route["slope_extra_share"] = per_route["trip_kwh_mean"] / per_route["flat_kwh_mean"] - 1
            per_route["trips_per_charge"] = COUNTY_BATTERY_KWH * USABLE_SHARE / per_route["trip_kwh_max"]
            per_route["km_per_charge"] = per_route["trips_per_charge"] * per_route["length_km"]
            per_route["mass_scenario"], per_route["aux_scenario"] = mass_name, aux_name
            rows.append(per_route)
    energy = pd.concat(rows, ignore_index=True)
    energy["is_base"] = (energy["mass_scenario"] == BASE_SCENARIO[0]) & (energy["aux_scenario"] == BASE_SCENARIO[1])
    energy.to_csv(TRANSIT / "route_energy.csv", index=False, encoding="utf-8-sig")

    print("[3/3] 요약 (기준 시나리오: 7t, 냉방 5kW)")
    base = energy[energy["is_base"]]
    for kind, g in base.groupby("kind"):
        print(f"  {kind}: kWh/km 중앙 {g['kwh_per_km'].median():.2f} (범위 {g['kwh_per_km'].min():.2f}~{g['kwh_per_km'].max():.2f}), "
              f"경사로 인한 추가 중앙 {g['slope_extra_share'].median():.1%} (최대 {g['slope_extra_share'].max():.1%}), "
              f"충전 1회 주행 중앙 {g['km_per_charge'].median():.0f}km")
    spread = energy.groupby(["mass_scenario", "aux_scenario"])["kwh_per_km"].median().unstack()
    print("  시나리오별 마을버스·자율주행 전체 kWh/km 중앙값:\n" + spread.round(2).to_string())


if __name__ == "__main__":
    main()
