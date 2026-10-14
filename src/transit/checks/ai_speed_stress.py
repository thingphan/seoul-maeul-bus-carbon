"""AI가 추정한 구간(B·C 등급)의 속도를 한쪽으로 틀리게 잡아도 결론이 유지되는지 확인한다.

AI 구간 속도를 모두 −5km/h, +5km/h(B 등급 MAE 4.6km/h보다 큼) 바꾼 뒤, 05와 같은 방법으로 버스·승용차 배출을
다시 계산하고 06의 네 유형 규칙으로 다시 분류한다. 버스와 승용차 모두 같은 속도를 쓴다.

입력: data/processed/transit/route_priority.csv(06), link_speed_weekday_profile.parquet(03), route_links.csv(01)
사용: python src/transit/checks/ai_speed_stress.py
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "transit"))

from energy import bus_fuel, hev_fuel, ice_fuel  # noqa: E402
from energy.ev_calibration import calibrate_county_electric, calibrate_two_point  # noqa: E402
from energy.link_energy import VTCPEMParameters, calculate_link_energy  # noqa: E402
from route_classes import assign_classes  # noqa: E402
from route_data import load_route_link_hours  # noqa: E402

TRANSIT = PROJECT_ROOT / "data" / "processed" / "transit"
SHIFT_KMH = 5.0


def load_carbon_module():
    spec = importlib.util.spec_from_file_location("carbon", PROJECT_ROOT / "src" / "transit" / "05_build_route_carbon_breakeven.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    cb = load_carbon_module()
    priority = pd.read_csv(TRANSIT / "route_priority.csv", dtype={"ROUTE_ID": str})
    frame = load_route_link_hours(kind="마을버스")
    frame = frame[frame["ROUTE_ID"].isin(priority["ROUTE_ID"])]
    tier = pd.read_parquet(TRANSIT / "link_speed_weekday_profile.parquet", columns=["LINK_ID", "tier"]).drop_duplicates("LINK_ID")
    frame = frame.merge(tier, on="LINK_ID")
    ai = frame["tier"] != "A_lookup"

    county, extra = calibrate_county_electric()
    bus = replace(VTCPEMParameters(), mass_kg=7000.0, drag_coefficient=county.drag_coefficient,
                  frontal_area_m2=county.frontal_area_m2, auxiliary_power_w=5000.0 + extra)
    ev, ev_extra = calibrate_two_point(cb.IONIQ5, cb.IONIQ5_LABEL)
    ev = replace(ev, auxiliary_power_w=ev.auxiliary_power_w + ev_extra)
    shares = cb.car_fleet_shares()
    fleet = cb.BUS_FLEET
    base_table = priority.drop(columns=["dependence_score", "class_id", "class"]).set_index("ROUTE_ID")

    def classify(speed: pd.Series):
        inputs = frame[["LINK_ID", "LENGTH", "slope_pct"]].assign(speed_kmh=speed.clip(lower=3.0))
        inputs = inputs[["LINK_ID", "LENGTH", "speed_kmh", "slope_pct"]]
        f = frame[["ROUTE_ID", "시간코드", "LENGTH"]].copy()
        f["bus"] = calculate_link_energy(inputs, bus)["energy_ev_wh"].to_numpy()
        f["ev"] = calculate_link_energy(inputs, ev)["energy_ev_wh"].to_numpy()
        for fuel, preset in ice_fuel.CAR_PRESETS.items():
            f[fuel] = ice_fuel.calculate_link_fuel(inputs, preset)["co2_g"].to_numpy()
        f["하이브리드"] = hev_fuel.calculate_link_fuel(inputs)["co2_g"].to_numpy()
        cols = ["bus", "ev", *ice_fuel.CAR_PRESETS, "하이브리드"]
        hourly = f.groupby(["ROUTE_ID", "시간코드"])[cols + ["LENGTH"]].sum()
        per_km = hourly[cols].div(hourly["LENGTH"], axis=0).groupby("ROUTE_ID").mean()
        car_g = sum(shares[k] * (per_km["ev"] * cb.GRID_KG_CO2_PER_KWH * 1000 if k == "전기" else per_km[k] * 1000)
                    for k in shares)
        kwh = per_km["bus"]
        bus_g = (fleet["전기"] * kwh * cb.GRID_KG_CO2_PER_KWH * 1000 + fleet["경유"] * bus_fuel.diesel_g_per_km(kwh)
                 + fleet["CNG"] * bus_fuel.cng_g_per_km(kwh)) / sum(fleet.values())
        d = base_table.copy()
        d["bus_g_per_pkm"] = (bus_g / d["avg_load"]).reindex(d.index)
        d["ratio"] = d["bus_g_per_pkm"] / (car_g / cb.CAR_OCCUPANCY).reindex(d.index)
        return assign_classes(d.reset_index()).set_index("ROUTE_ID")

    results = {name: classify(frame["speed_kmh"].where(~ai, frame["speed_kmh"] + delta))
               for name, delta in (("기준", 0.0), (f"AI 구간 −{SHIFT_KMH:.0f}km/h", -SHIFT_KMH),
                                   (f"AI 구간 +{SHIFT_KMH:.0f}km/h", SHIFT_KMH))}
    base = results["기준"]
    n = len(base)
    print(f"AI 추정 구간의 노선 길이 비중: {frame.loc[ai, 'LENGTH'].sum() / frame['LENGTH'].sum():.1%}")
    for name, d in results.items():
        print(f"  {name}: 버스 우위 {int((d['ratio'] < 1).sum())}/{n}, 기준과 같은 유형 {int((d['class_id'] == base['class_id']).sum())}/{n}")
    all_better = pd.concat([d["ratio"] < 1 for d in results.values()], axis=1).all(axis=1)
    all_same = pd.concat([d["class_id"] == base["class_id"] for d in results.values()], axis=1).all(axis=1)
    print(f"세 경우 모두 버스 우위 {int(all_better.sum())}/{n}, 세 경우 모두 같은 유형 {int(all_same.sum())}/{n}")


if __name__ == "__main__":
    main()
