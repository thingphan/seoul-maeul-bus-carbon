"""시간대별 탑승, 한산한 시간대 소형 차량 효과, 적자 추정, 탄소 목적 증편 후보를 계산한다.

입력 (data/processed/transit/)
  route_carbon_breakeven.csv(05), route_priority.csv(06), route_demand.csv(02)
  승하차(OA-12913)·운행횟수(OA-21220)·정류소 순번 — route_data.py 공통 함수

처리
  1. 시간대별 평균 탑승: 노선의 하루 승객·km를 시간대별 승차 비중으로 나누고(승객·km가 승차와 같은
     시간 분포를 따른다고 근사), 그 시간대 운행횟수 × 1회 운행 길이로 나눈다. 운행횟수는 05와 같은
     정의(날짜마다 정류장 중앙값 → 기간 평균)를 시간대별로 적용한다.
  2. 한산한 시간대 소형 차량: 평균 탑승이 LOW_LOAD명 미만인 노선·시간대만 15인승급 소형 전기
     승합차(06의 van_g_per_km)로 돌릴 때 하루 CO2 감소량. 지금 차량 구성(bus_g_per_km) 대비 감축과,
     크기 효과만 본 전기 마을버스(bus_ev_g_per_km) 대비 감축을 함께 낸다. 1km당 값은 07~21시 평균 속도 기준.
  3. 적자 추정: 1회 운행 원가 = 1대당 하루 재정지원 기준액(2026년 510,457원) ÷ 1대당 하루 운행횟수
     (2026년 1월 서울 마을버스 월 831,798회 ÷ 31일 ÷ 운행 1,511대). 운송수입 = 하루 승차 × 승객
     1명당 실수입. 환승 정산으로 실수입이 노선마다 달라 600·900·1,200원 세 경우로 범위를 낸다.
  4. 탄소 목적 증편 후보: 지금도 손님이 많아 탄소 효율이 높은 유형(① 증편·지원 우선, ③ 현행 유지) 가운데
     노선 주변 승용차 보유가 중앙값 이상(증편 시 승용차 이용자를 끌어올 여지)이고 증편 1회 손익분기 인원이
     중앙값 이하(적은 전환으로도 본전)인 노선. ②·④ 유형은 손님이 적어 증편보다 차 크기·노선 조정이 먼저다.
     중앙값은 230개 노선 전체 기준이다.

출력 (data/processed/transit/)
  route_hourly_load.csv  노선×시간대(06~22시): 승차 비중, 운행횟수, 평균 탑승, 소형 차량 감소량
  route_operations.csv   노선 1행: 저녁 평균 탑승, 한산 운행 비중, 소형 차량 감소량, 적자 여부(세 경우), 증편 후보

사용: python src/transit/07_build_route_operations.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from route_data import load_route_stops, load_stop_ridership, route_trips_per_day
from route_kinds import branch_stop_weights, ridership_key

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRANSIT = PROJECT_ROOT / "data" / "processed" / "transit"
SERVICE_HOURS = list(range(6, 23))  # 06~22시
BANDS = {"출근(07~09시)": range(7, 10), "낮(10~16시)": range(10, 17),
         "퇴근(17~19시)": range(17, 20), "저녁(20~22시)": range(20, 23)}
LOW_LOAD = 3.0
COST_PER_VEHICLE_DAY = 510_457
TRIPS_PER_VEHICLE_DAY = 831_798 / 31 / 1_511
FARES = (600, 900, 1200)


def hourly_boardings(route_ids: set[str]) -> pd.DataFrame:
    """노선×시(0~23) 하루 평균 승차. 분기 노선은 정류장별 배분 비율을 곱한다."""
    stops = load_route_stops()
    stops = stops[stops["ROUTE_ID"].isin(route_ids)]
    member = stops[["ROUTE_ID", "노선명", "NODE_ID"]].drop_duplicates(["ROUTE_ID", "NODE_ID"])
    member = member.join(branch_stop_weights(stops).rename("w"), on=["ROUTE_ID", "NODE_ID"])
    member["ride_key"] = member["노선명"].map(ridership_key)
    board_cols = [f"board_{h}" for h in range(24)]
    merged = member.merge(load_stop_ridership()[["ride_key", "NODE_ID", *board_cols]], on=["ride_key", "NODE_ID"])
    out = merged[board_cols].mul(merged["w"], axis=0).groupby(merged["ROUTE_ID"]).sum()
    out.columns = range(24)
    return out


def main() -> None:
    carbon = pd.read_csv(TRANSIT / "route_carbon_breakeven.csv", dtype={"ROUTE_ID": str})
    carbon = carbon[~carbon["load_qa_flag"]]
    priority = pd.read_csv(TRANSIT / "route_priority.csv", dtype={"ROUTE_ID": str})[
        ["ROUTE_ID", "class_id", "class", "van_g_per_km"]]
    demand = pd.read_csv(TRANSIT / "route_demand.csv", dtype={"ROUTE_ID": str})[["ROUTE_ID", "route_cars_per_1000"]]
    data = carbon.merge(priority, on="ROUTE_ID").merge(demand, on="ROUTE_ID", how="left")
    ids = set(data["ROUTE_ID"])

    print("[1/4] 시간대별 평균 탑승...")
    board = hourly_boardings(ids)
    share = board.div(board.sum(axis=1), axis=0)[SERVICE_HOURS].stack().rename("board_share")
    trips = route_trips_per_day(ids, hourly=True)[SERVICE_HOURS].stack().rename("trips")
    hourly = (pd.concat([share.rename_axis(["ROUTE_ID", "hour"]), trips.rename_axis(["ROUTE_ID", "hour"])], axis=1, join="inner")
              .reset_index())
    hourly = hourly[hourly["trips"] > 0].merge(
        data[["ROUTE_ID", "pkm_per_day", "length_km", "bus_g_per_km", "bus_ev_g_per_km", "van_g_per_km"]], on="ROUTE_ID")
    hourly["pkm"] = hourly["pkm_per_day"] * hourly["board_share"]
    hourly["vkm"] = hourly["trips"] * hourly["length_km"]
    hourly["avg_load"] = hourly["pkm"] / hourly["vkm"]

    print("[2/4] 한산한 시간대 소형 차량 효과...")
    hourly["low_load"] = hourly["avg_load"] < LOW_LOAD
    hourly["van_cut_kg"] = np.where(hourly["low_load"],
                                    hourly["vkm"] * (hourly["bus_ev_g_per_km"] - hourly["van_g_per_km"]) / 1000.0, 0.0)
    hourly["van_cut_vs_current_kg"] = np.where(hourly["low_load"],
                                               hourly["vkm"] * (hourly["bus_g_per_km"] - hourly["van_g_per_km"]) / 1000.0, 0.0)
    hourly.drop(columns=["pkm_per_day", "length_km", "bus_g_per_km", "bus_ev_g_per_km", "van_g_per_km"]).to_csv(
        TRANSIT / "route_hourly_load.csv", index=False, encoding="utf-8-sig")
    hourly["low_trips"] = hourly["trips"].where(hourly["low_load"], 0.0)
    evening = hourly[hourly["hour"].isin(BANDS["저녁(20~22시)"])].groupby("ROUTE_ID")[["pkm", "vkm"]].sum()
    route_hours = hourly.groupby("ROUTE_ID").agg(trips=("trips", "sum"), low_trips=("low_trips", "sum"),
                                                  van_offpeak_cut_kg_per_day=("van_cut_kg", "sum"),
                                                  van_offpeak_cut_vs_current_kg_per_day=("van_cut_vs_current_kg", "sum"))
    route_hours["low_load_trip_share"] = route_hours["low_trips"] / route_hours["trips"]
    route_hours["evening_avg_load"] = evening["pkm"] / evening["vkm"]

    print("[3/4] 적자 추정...")
    data["cost_per_day"] = data["trips_per_day"] / TRIPS_PER_VEHICLE_DAY * COST_PER_VEHICLE_DAY
    for fare in FARES:
        data[f"deficit_at_{fare}"] = data["cost_per_day"] > data["boardings_per_day"] * fare

    print("[4/4] 탄소 목적 증편 후보...")
    data["carbon_service_candidate"] = (
        data["class_id"].isin([1, 3])
        & (data["route_cars_per_1000"] >= data["route_cars_per_1000"].median())
        & (data["added_trip_breakeven_riders"] <= data["added_trip_breakeven_riders"].median()))
    out = data.join(route_hours[["evening_avg_load", "low_load_trip_share", "van_offpeak_cut_kg_per_day",
                                 "van_offpeak_cut_vs_current_kg_per_day"]], on="ROUTE_ID")
    cols = ["ROUTE_ID", "노선명", "class", "boardings_per_day", "trips_per_day", "avg_load", "evening_avg_load",
            "low_load_trip_share", "van_offpeak_cut_kg_per_day", "van_offpeak_cut_vs_current_kg_per_day", "cost_per_day",
            *[f"deficit_at_{f}" for f in FARES],
            "route_cars_per_1000", "added_trip_breakeven_riders", "carbon_service_candidate"]
    out[cols].to_csv(TRANSIT / "route_operations.csv", index=False, encoding="utf-8-sig")

    print(f"\n  노선 {len(out)}개, 노선·시간대 {len(hourly):,}개 | 1회 운행 원가 약 {COST_PER_VEHICLE_DAY / TRIPS_PER_VEHICLE_DAY:,.0f}원")
    for band, hours in BANDS.items():
        b = hourly[hourly["hour"].isin(hours)]
        print(f"  {band}: 평균 탑승 {b['pkm'].sum() / b['vkm'].sum():.1f}명(노선·시간 중앙 {b['avg_load'].median():.1f}명), "
              f"승객 {LOW_LOAD:.0f}명 미만 운행 {b['low_trips'].sum() / b['trips'].sum():.0%}")
    total_ev = (out["trips_per_day"] * out["length_km"] * out["bus_ev_g_per_km"]).sum() / 1e6
    print(f"  승객 {LOW_LOAD:.0f}명 미만 시간대만 소형 전기차로: 하루 {out['van_offpeak_cut_vs_current_kg_per_day'].sum() / 1000:.2f}t 감소"
          f"(지금 차량 대비), 그중 크기 효과 {out['van_offpeak_cut_kg_per_day'].sum() / 1000:.2f}t"
          f"(전부 전기 마을버스일 때 하루 배출 {total_ev:.1f}t 대비), 해당 운행 비중 "
          f"{hourly['low_trips'].sum() / hourly['trips'].sum():.0%}")
    for fare in FARES:
        print(f"  승객 1명당 실수입 {fare:,}원: 적자 {int(out[f'deficit_at_{fare}'].sum())}/{len(out)}  유형별 적자 비율 "
              + str(out.groupby("class")[f"deficit_at_{fare}"].mean().round(2).to_dict()))
    cand = out[out["carbon_service_candidate"]].sort_values("added_trip_breakeven_riders")
    print(f"  탄소 목적 증편 후보 {len(cand)}개(①③ 유형 중 승용차 보유 중앙 이상 & 증편 손익분기 중앙 이하), 칸별 "
          + str(cand["class"].value_counts().sort_index().to_dict()))
    print(cand[["노선명", "class", "route_cars_per_1000", "added_trip_breakeven_riders", "avg_load"]].head(10)
          .round(1).to_string(index=False))


if __name__ == "__main__":
    main()
