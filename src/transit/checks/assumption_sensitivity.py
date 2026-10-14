"""주요 가정을 바꿔도 결론(버스 우위 노선 수, 네 유형 분류)이 유지되는지 확인한다.

1. 승용차 탑승인원: 1.2명(기준) → 1.35명(출근 평균) → 1.5명 → 1.82명(하루 평균)
2. 배출 배율: 버스 배출 1.28배(서울 마을버스 대형 비중 반영 추정)·1.5배, 승용차 배출 0.8배, 가장 불리한 조합
3. 대체수단 기준: 일반버스 정류장 150·200(기준)·300m(지하철 500m는 고정)
4. 대체수단 거리를 직선 대신 도로망 거리로(08 산출물): 유형이 바뀐 노선이 기준선 근처인지

입력: data/processed/transit/route_carbon_breakeven.csv(05), route_priority.csv(06), bus_stops.csv·routes.csv(01),
      route_class_network.csv(08)
사용: python src/transit/checks/assumption_sensitivity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "transit"))

from route_classes import assign_classes  # noqa: E402
from route_data import load_route_stops, load_stop_ridership  # noqa: E402
from route_kinds import branch_stop_weights, ridership_key  # noqa: E402

TRANSIT = PROJECT_ROOT / "data" / "processed" / "transit"
NEAR_PCT = 0.10  # 기준선(중앙값)에서 백분위 10%p 이내면 '기준선 근처'


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(TRANSIT / name, dtype={"ROUTE_ID": str, "NODE_ID": str})


def occupancy(carbon: pd.DataFrame) -> None:
    print("== 1. 승용차 탑승인원")
    for occ in (1.2, 1.35, 1.5, 1.82):
        car_pkm = carbon["car_g_per_km"] / occ
        better = carbon["bus_g_per_pkm"] < car_pkm
        breakeven = carbon["bus_g_per_km"] / carbon["car_g_per_km"] * occ
        added = carbon["bus_g_per_km"] * carbon["length_km"] / (carbon["avg_trip_km"] * car_pkm)
        print(f"  {occ}명: 버스 우위 {int(better.sum())}/{len(carbon)}, 손익분기 재차인원 중앙 {breakeven.median():.2f}명, "
              f"증편 손익분기 중앙 {added.median():.1f}명")


def emission_scaling(carbon: pd.DataFrame) -> None:
    print("== 2. 배출 배율(버스 배출 × a, 승용차 배출 × b)")
    ratio = carbon["bus_g_per_pkm"] / carbon["car_g_per_pkm"]
    for bus_x, car_x, label in ((1.0, 1.0, "기준"), (1.28, 1.0, "버스 1.28배(대형 비중 반영)"), (1.5, 1.0, "버스 1.5배"),
                                (1.0, 0.8, "승용차 0.8배"), (1.5, 0.8, "가장 불리한 조합(버스 1.5배·승용차 0.8배)")):
        print(f"  {label}: 버스 우위 {int((ratio * bus_x / car_x < 1).sum())}/{len(carbon)}")


def trunk_radius(priority: pd.DataFrame) -> None:
    print("== 3. 일반버스 정류장 접근 거리")
    stops = read("bus_stops.csv")
    trunk = stops["served_kinds"].str.contains("일반").to_numpy()
    tree = cKDTree(stops.loc[trunk, ["x", "y"]].to_numpy())
    dist_trunk = np.where(trunk, 0.0, tree.query(stops[["x", "y"]].to_numpy())[0])
    route_ids = set(read("routes.csv")["ROUTE_ID"])
    rs = load_route_stops()
    rs = rs[rs["ROUTE_ID"].isin(route_ids)]
    member = rs[["ROUTE_ID", "노선명", "NODE_ID"]].drop_duplicates(["ROUTE_ID", "NODE_ID"])
    member = member.join(branch_stop_weights(rs).rename("w"), on=["ROUTE_ID", "NODE_ID"])
    member["ride_key"] = member["노선명"].map(ridership_key)
    ride = load_stop_ridership()
    ride["board"] = ride.filter(like="board_").sum(axis=1)
    m = member[member["ROUTE_ID"].isin(priority["ROUTE_ID"])].merge(ride[["ride_key", "NODE_ID", "board"]],
                                                                      on=["ride_key", "NODE_ID"])
    village = stops["served_kinds"].str.contains("마을버스")
    base = priority.drop(columns=["dependence_score", "class_id", "class"])
    for radius in (150, 200, 300):
        stops["no_alt"] = (stops["dist_to_station_m"] > 500) & (dist_trunk > radius)
        mm = m.merge(stops[["NODE_ID", "no_alt"]], on="NODE_ID", how="left")
        mm["b"] = mm["board"] * mm["w"]
        share = (mm["b"] * mm["no_alt"].fillna(False)).groupby(mm["ROUTE_ID"]).sum() / mm.groupby("ROUTE_ID")["b"].sum()
        redone = assign_classes(base.assign(no_alt_boarding_share=base["ROUTE_ID"].map(share)))
        same = int((redone["class_id"].to_numpy() == priority["class_id"].to_numpy()).sum())
        note = ""
        if radius == 200:
            gap = np.nanmax(np.abs(share.reindex(priority["ROUTE_ID"]).to_numpy() - priority["no_alt_boarding_share"].to_numpy()))
            note = f" (02 결과와 최대 차이 {gap:.1e})"
        print(f"  {radius}m: 대체수단 없는 마을버스 정류장 {stops.loc[village, 'no_alt'].mean():.1%}, "
              f"기준과 같은 유형 {same}/{len(priority)}{note}")


def network_distance(priority: pd.DataFrame) -> None:
    print("== 4. 대체수단 거리를 도로망 거리로")
    net = read("route_class_network.csv")
    m = priority[["ROUTE_ID", "dependence_score", "bus_g_per_pkm"]].merge(net, on="ROUTE_ID")
    near = ((m["dependence_score"].rank(pct=True) - 0.5).abs() < NEAR_PCT) | \
           ((m["bus_g_per_pkm"].rank(pct=True) - 0.5).abs() < NEAR_PCT)
    for col, label in (("class_id_no_alt_net", "도로망 거리(기준 거리 그대로)"),
                       ("class_id_no_alt_net_scaled", "도로망 거리(기준 거리 × 1.3)")):
        changed = m["class_id"] != m[col]
        print(f"  {label}: 같은 유형 {int((~changed).sum())}/{len(m)}, 바뀐 {int(changed.sum())}개 중 "
              f"기준선 근처 {near[changed].mean():.0%}(전체 노선은 {near.mean():.0%})")


def main() -> None:
    carbon = read("route_carbon_breakeven.csv")
    carbon = carbon[~carbon["load_qa_flag"]]
    priority = read("route_priority.csv")
    occupancy(carbon)
    emission_scaling(carbon)
    trunk_radius(priority)
    network_distance(priority)


if __name__ == "__main__":
    main()
