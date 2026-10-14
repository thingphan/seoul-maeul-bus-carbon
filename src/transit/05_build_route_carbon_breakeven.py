"""마을버스 노선별로 승객 1명·1km당 탄소(CO2)를 승용차와 비교한다.

질문: 이 노선의 전기 마을버스는 같은 길을 승용차로 가는 것보다 승객 1명당 탄소가 적은가?

입력
  data/processed/transit/route_links.csv, route_segments.csv, link_speed_weekday_profile.parquet, route_energy.csv (01, 03, 04)
  data/raw/traffic/서울시버스노선별정류소정보(20260902).xlsx   (정류소 순번)
  data/raw/traffic/2026년_버스노선별_정류장별_시간대별_승하차_인원_정보(07월).csv
  data/raw/traffic/bus_trips/tpss_sta_route_hturn_2026.07.*.csv   (서울 열린데이터광장 OA-21220,
      TOPIS 원천 노선별·정류장별 일별 버스 운행횟수, route_data.py가 캐시)

처리
  1. 구간별 차 안 승객 수: 정류소 순번대로 '승차 - 하차'를 누적한다. 교통카드 하차 태그
     누락으로 하차가 승차보다 적게 잡히므로 노선 전체 하차 합을 승차 합에 맞춰 비례 보정한다.
     같은 정류소가 기점·종점으로 두 번 나오면 승차는 첫 등장, 하차는 마지막 등장에 둔다.
     분기 노선(서대문02대·02소 등)은 승하차가 합쳐져 있어 route_kinds.py 규칙으로 정류장별로 나눈다.
  2. 승객·km = Σ(구간 승객 수 × 실제 경로선 기준 구간 길이). 7월 월합계를 31일로 나눠 하루값으로 쓴다.
  3. 하루 운행횟수: 노선·날짜별로 정류장 운행횟수의 중앙값을 취해 7/6~7/31(주말 포함, 승하차와
     같은 기준) 평균을 낸다.
     평균 승객 수 = 하루 승객·km / (하루 운행횟수 × 1회 운행 길이)
  4. 같은 노선 링크·같은 평일 시간대 속도로 1km당 CO2를 계산하고, 버스·승용차 모두 2026년 서울
     '현재 차량 구성'으로 가중평균한다.
       마을버스(2026.1 서울시 마을버스 현황: 전기 397·CNG 885·경유 344대, 크기는 중형으로 고정)
         전기: 04 기준 시나리오(카운티 일렉트릭 공인 주행거리 보정) kWh × 국가 전력배출계수
               0.4173kgCO2/kWh(2023년도, 2025.12 공표)
         경유·CNG: 같은 전기버스 kWh/km × 서울 실측 연료별 비율(src/energy/bus_fuel.py)
       승용차(2026.7 서울시 자치구별 연료별 등록현황, 승용 276만 대) — 연료별 중형급 대표 차종
         휘발유·경유·LPG: src/energy/ice_fuel.py CAR_PRESETS(쏘나타 2.0 / 쏘렌토 2.2 디젤 / 쏘나타 LPi,
           공인 도심·고속 연비 보정)
         하이브리드: src/energy/hev_fuel.py(Ahn & Rakha 2019 HEV 모델, 쏘나타 하이브리드 공인 연비 보정)
         전기·수소: VT-CPEM + 아이오닉5 롱레인지 2WD(공인 도심·고속 전비 보정, src/energy/ev_calibration.py)
         모든 차종을 공인 시험주기 값으로 보정해 정지·출발 소비를 같은 기준으로 반영한다.
     승용차는 1.2명 탑승으로 두고 승객 1명·1km당 CO2를 비교한다.
       손익분기 평균 승객 수 L* = (버스 gCO2/km) / (승용차 gCO2/km) × 1.2
       손익분기 운행횟수   N* = 하루 승객·km / (L* × 1회 운행 길이)
     참고 시나리오: 마을버스 전부 전기(전기 전환 효과), 승용차 전부 전기(장기).
  5. 평균 승차거리가 0.5km 미만인 노선은 정류소 순번(9월)과 승하차(7월) 시점 차이로 승객 수 누적이
     어긋난 것으로 보고 load_qa_flag로 표시한다.

출력
  data/processed/transit/route_carbon_breakeven.csv

사용: python src/transit/05_build_route_carbon_breakeven.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from dataclasses import replace  # noqa: E402

from energy import bus_fuel, hev_fuel, ice_fuel  # noqa: E402
from energy.ev_calibration import CITY_KMH, HWY_KMH, calibrate_two_point  # noqa: E402
from energy.link_energy import VTCPEMParameters, calculate_link_energy  # noqa: E402
from route_data import load_route_link_hours, load_route_stops, load_stop_ridership, route_trips_per_day  # noqa: E402
from route_kinds import branch_stop_weights, ridership_key  # noqa: E402

RAW_TRAFFIC = PROJECT_ROOT / "data" / "raw" / "traffic"
TRANSIT = PROJECT_ROOT / "data" / "processed" / "transit"

CAR_OCCUPANCY = 1.2
GRID_KG_CO2_PER_KWH = 0.4173
MIN_AVG_TRIP_KM = 0.5
REGISTRATION_CSV = RAW_TRAFFIC / "vehicle_registration" / "seoul_fuel_registration_202607.csv"
BUS_FLEET = {"전기": 397, "CNG": 885, "경유": 344}          # 2026.1 서울 마을버스 연료별 대수
# 아이오닉5 롱레인지 2WD(19인치): 공차중량 2,015kg + 운전자, 공인 도심 5.8·고속 4.6km/kWh(제조사 공개 제원)
IONIQ5 = replace(VTCPEMParameters(), mass_kg=2015.0 + 75.0, drag_coefficient=0.288,
                 frontal_area_m2=1.890 * 1.605 * 0.85)
IONIQ5_LABEL = [(CITY_KMH, 1000.0 / 5.8), (HWY_KMH, 1000.0 / 4.6)]


def car_fleet_shares() -> dict[str, float]:
    """서울 승용차 연료 비율: 휘발유(기타 연료 포함)·경유·LPG·하이브리드·전기(수소 포함)."""
    reg = pd.read_csv(REGISTRATION_CSV, encoding="cp949")
    cars = reg[reg["그룹"] == 0].groupby("연료명")["승용"].sum()
    total = cars.sum()
    groups = {
        "경유": cars.get("경유", 0),
        "LPG": cars.get("엘피지", 0),
        "하이브리드": cars[cars.index.str.startswith("하이브리드")].sum(),
        "전기": cars[cars.index.isin(["전기", "수소", "수소전기"])].sum(),
    }
    groups["휘발유"] = total - sum(groups.values())
    return {k: v / total for k, v in groups.items()}


def load_profiles(route_names: set[str]) -> pd.DataFrame:
    """노선×정류소 순번별 하루 승차·하차(하차는 노선 단위로 승차 합에 맞춰 보정)."""
    seq = load_route_stops()
    seq = seq[seq["노선명"].isin(route_names)].sort_values(["ROUTE_ID", "순번"])
    ride = load_stop_ridership()
    ride = pd.DataFrame({"ride_key": ride["ride_key"], "NODE_ID": ride["NODE_ID"],
                         "board": ride.filter(like="board_").sum(axis=1), "alight": ride.filter(like="alight_").sum(axis=1)})

    seq["first"] = ~seq.duplicated(["ROUTE_ID", "NODE_ID"], keep="first")
    seq["last"] = ~seq.duplicated(["ROUTE_ID", "NODE_ID"], keep="last")
    seq = seq.join(branch_stop_weights(seq).rename("share"), on=["ROUTE_ID", "NODE_ID"])
    seq["ride_key"] = seq["노선명"].map(ridership_key)
    seq = seq.merge(ride, on=["ride_key", "NODE_ID"], how="left").fillna({"board": 0.0, "alight": 0.0})
    seq["board"] *= seq["share"]
    seq["alight"] *= seq["share"]
    seq["board"] = np.where(seq["first"], seq["board"], 0.0)
    seq["alight"] = np.where(seq["last"], seq["alight"], 0.0)
    totals = seq.groupby("ROUTE_ID")[["board", "alight"]].transform("sum")
    seq["alight"] = seq["alight"] * (totals["board"] / totals["alight"].replace(0, np.nan)).fillna(0)
    seq["onboard_after"] = (seq["board"] - seq["alight"]).groupby(seq["ROUTE_ID"]).cumsum().clip(lower=0)
    return seq


def main() -> None:
    car = load_route_link_hours(kind="마을버스")
    bus = pd.read_csv(TRANSIT / "route_energy.csv", dtype={"ROUTE_ID": str})
    bus = bus[bus["is_base"] & (bus["kind"] == "마을버스")][["ROUTE_ID", "kwh_per_km", "length_km"]]

    print("[1/4] 구간별 승객 수와 승객·km...")
    seq = load_profiles(set(car["노선명"]))
    # 01의 route_segments.csv: stop_order n = '순번 정렬 기준 n번째 정류소 → n+1번째' 구간(실제 경로선 길이).
    seq["stop_order"] = seq.groupby("ROUTE_ID").cumcount() + 1
    segments = pd.read_csv(TRANSIT / "route_segments.csv", dtype={"ROUTE_ID": str})
    seg = seq.merge(segments, on=["ROUTE_ID", "stop_order"], how="left")
    seg["pkm"] = seg["onboard_after"] * seg["seg_km"].fillna(0)
    pax = seg.groupby(["ROUTE_ID", "노선명"], as_index=False).agg(
        boardings_per_day=("board", "sum"), pkm_per_day=("pkm", "sum"), peak_onboard=("onboard_after", "max"))
    pax["avg_trip_km"] = pax["pkm_per_day"] / pax["boardings_per_day"]

    print("[2/4] 노선별 하루 운행횟수(TOPIS)...")
    trips = route_trips_per_day(set(pax["ROUTE_ID"]))

    print("[3/4] 같은 길·같은 시간대 승용차(연료별) 1km 탄소...")
    link_input = car[["LINK_ID", "LENGTH", "speed_kmh", "slope_pct"]]
    ioniq5, extra_w = calibrate_two_point(IONIQ5, IONIQ5_LABEL)
    ioniq5 = replace(ioniq5, auxiliary_power_w=ioniq5.auxiliary_power_w + extra_w)
    car["ev_wh"] = calculate_link_energy(link_input, ioniq5)["energy_ev_wh"].to_numpy()
    for fuel, preset in ice_fuel.CAR_PRESETS.items():
        car[f"{fuel}_g"] = ice_fuel.calculate_link_fuel(link_input, preset)["co2_g"].to_numpy()
    car["하이브리드_g"] = hev_fuel.calculate_link_fuel(link_input)["co2_g"].to_numpy()
    value_cols = ["ev_wh", *[f"{f}_g" for f in ice_fuel.CAR_PRESETS], "하이브리드_g"]
    per_hour = car.groupby(["ROUTE_ID", "시간코드"])[value_cols + ["LENGTH"]].sum()
    car_per_km = pd.DataFrame({
        "ev_car_kwh_per_km": per_hour["ev_wh"] / per_hour["LENGTH"],
        **{f"car_{c[:-2]}_g_per_km": per_hour[c] / per_hour["LENGTH"] * 1000.0 for c in value_cols[1:]},
    }).groupby("ROUTE_ID").mean()

    print("[4/4] 승객 1명·1km당 비교...")
    out = (pax.merge(bus, on="ROUTE_ID", how="inner").join(car_per_km, on="ROUTE_ID").join(trips, on="ROUTE_ID")
           .rename(columns={"kwh_per_km": "bus_kwh_per_km"}))
    # 버스: 전기(노선별 모델값)와 경유·CNG(같은 전기버스 kWh × 서울 실측 비율)를 현재 대수로 가중
    fleet_total = sum(BUS_FLEET.values())
    out["bus_ev_g_per_km"] = out["bus_kwh_per_km"] * GRID_KG_CO2_PER_KWH * 1000.0
    out["bus_diesel_g_per_km"] = bus_fuel.diesel_g_per_km(out["bus_kwh_per_km"])
    out["bus_cng_g_per_km"] = bus_fuel.cng_g_per_km(out["bus_kwh_per_km"])
    out["bus_g_per_km"] = (BUS_FLEET["전기"] * out["bus_ev_g_per_km"] + BUS_FLEET["경유"] * out["bus_diesel_g_per_km"]
                           + BUS_FLEET["CNG"] * out["bus_cng_g_per_km"]) / fleet_total
    # 승용차: 연료별 대표 차종 모델값을 서울 등록 비율로 가중
    shares = car_fleet_shares()
    out["car_전기_g_per_km"] = out["ev_car_kwh_per_km"] * GRID_KG_CO2_PER_KWH * 1000.0
    out["car_g_per_km"] = sum(share * out[f"car_{fuel}_g_per_km"] for fuel, share in shares.items())

    out["avg_load"] = out["pkm_per_day"] / (out["trips_per_day"] * out["length_km"])
    out["breakeven_load"] = out["bus_g_per_km"] / out["car_g_per_km"] * CAR_OCCUPANCY
    out["breakeven_trips"] = out["pkm_per_day"] / (out["breakeven_load"] * out["length_km"])
    out["bus_g_per_pkm"] = out["bus_g_per_km"] / out["avg_load"]
    out["car_g_per_pkm"] = out["car_g_per_km"] / CAR_OCCUPANCY
    out["bus_vs_car"] = out["bus_g_per_pkm"] / out["car_g_per_pkm"]          # 1 미만이면 버스가 탄소 적음
    out["bus_better"] = out["bus_vs_car"] < 1
    out["co2_saved_kg_per_day"] = out["pkm_per_day"] * (out["car_g_per_pkm"] - out["bus_g_per_pkm"]) / 1000.0
    # 증편 1회가 탄소 손해를 보지 않으려면 승용차에서 넘어와야 하는 승객 수
    out["added_trip_kg"] = out["bus_g_per_km"] * out["length_km"] / 1000.0
    out["car_rider_kg"] = out["avg_trip_km"] * out["car_g_per_pkm"] / 1000.0
    out["added_trip_breakeven_riders"] = out["added_trip_kg"] / out["car_rider_kg"]
    # 참고 시나리오
    vkm = out["trips_per_day"] * out["length_km"]
    out["bus_co2_kg_per_day"] = out["bus_g_per_km"] * vkm / 1000.0
    out["bus_co2_kg_per_day_all_electric"] = out["bus_ev_g_per_km"] * vkm / 1000.0
    out["bus_vs_ev_car"] = (out["bus_kwh_per_km"] / out["avg_load"]) / (out["ev_car_kwh_per_km"] / CAR_OCCUPANCY)
    out["load_qa_flag"] = out["avg_trip_km"] < MIN_AVG_TRIP_KM
    out.to_csv(TRANSIT / "route_carbon_breakeven.csv", index=False, encoding="utf-8-sig")

    ok = out[~out["load_qa_flag"]]
    print(f"  분석 노선 {len(ok)}개 (승객 누적 이상으로 제외 {int(out['load_qa_flag'].sum())}개)")
    print("  승용차 구성(서울 2026.7): " + ", ".join(f"{k} {v:.1%}" for k, v in shares.items()))
    print("  1km당 CO2 중앙값(g): " + ", ".join(
        f"{c.removeprefix('car_').removesuffix('_g_per_km')} {ok[c].median():.0f}"
        for c in ok.columns if c.startswith("car_") and c.endswith("_g_per_km") and c != "car_g_per_km")
        + f" | 마을버스 전기 {ok['bus_ev_g_per_km'].median():.0f}, 경유 {ok['bus_diesel_g_per_km'].median():.0f},"
          f" CNG {ok['bus_cng_g_per_km'].median():.0f}, 현재 구성 {ok['bus_g_per_km'].median():.0f}")
    cols = ["boardings_per_day", "trips_per_day", "avg_load", "bus_g_per_km", "car_g_per_km",
            "breakeven_load", "bus_vs_car", "co2_saved_kg_per_day", "added_trip_breakeven_riders"]
    print(ok[cols].describe(percentiles=[.1, .25, .5, .75, .9]).round(2).to_string())
    print(f"  현재 승용차 구성보다 승객 1명당 탄소가 적은 노선: {int(ok['bus_better'].sum())} / {len(ok)}")
    saved = ok["co2_saved_kg_per_day"].sum() / 1000.0
    print(f"  같은 승객·km를 승용차로 갔을 때 대비 하루 CO2 차이: {saved:,.1f}t (연 {saved * 365 / 1000:,.1f}천t)")
    now, elec = ok["bus_co2_kg_per_day"].sum() / 1000.0, ok["bus_co2_kg_per_day_all_electric"].sum() / 1000.0
    print(f"  마을버스 배출: 현재 하루 {now:,.1f}t → 전부 전기 {elec:,.1f}t (전기 전환 감축 {now - elec:,.1f}t/일)")
    print(f"  참고(승용차가 모두 전기차, 버스도 전기) 버스가 나은 노선: {int((ok['bus_vs_ev_car'] < 1).sum())} / {len(ok)}")


if __name__ == "__main__":
    main()
