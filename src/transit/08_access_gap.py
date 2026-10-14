"""정류장까지 걷는 부담(노인 걸음, 언덕 반영)과 도로망 거리로 본 '대체수단 없는 정류장'을 계산한다.

입력
  data/processed/transit/grid_population_daytime.csv(02), bus_stops.csv(01), routes.csv(01), route_priority.csv(06)
  data/interim/seoul_link.shp, data/processed/seoul_link_dimension.csv (road_network.py)
  data/raw/traffic/서울시 역사마스터 정보.csv, 정류소·승하차(route_data.py)

처리
  걷기 도로망: 표준노드링크를 양방향 보행 그래프로 쓴다. 인도·계단·좁은 골목 일부가 빠져 있어 실제보다
  길게 잡힐 수 있으므로, 직선거리 기반 간이 계산과 함께 범위로 본다.
  1. 노인 보행 시간(250m 격자 중심 → 가장 가까운 버스 정류장, 모든 버스):
       Tobler 보행속도 v = 6·exp(−3.5·|s + 0.05|) km/h (s: 걷는 방향 경사, 내리막은 음수), 노인은 0.75배
       (평지 약 1.05m/s).
       도로망: 링크마다 걷는 방향 경사로 시간을 내고, 정류장 전체를 출발점으로 한 번에 최단 시간을 구한다.
               격자 중심·정류장에서 가장 가까운 노드까지는 평지 속도로 걷는다고 둔다.
       간이:   직선거리 × 우회계수 1.3, 경사는 격자 300m 안 도로 링크 평균 |경사|를 오르막으로 둔다.
       각각 평지 가정(경사 0)과 비교해 언덕이 늘리는 몫을 고령 인구 가중으로 낸다.
  2. 대체수단 없는 정류장(마을버스 정류장): 직선 기준(01: 지하철역 500m 밖 & 일반버스 정류장 200m 안 없음)을
     도로망 거리로 다시 판정한다. 기준 거리를 그대로 둔 경우와 ×1.3(직선→도로 환산)한 경우를 함께 낸다.
  3. 도로망 기준으로 노선의 '대체수단 없는 승차 비중'을 다시 구해 네 유형(route_classes.py)을 다시 나누고,
     유형이 바뀌는 노선 수를 센다(민감도). 직선 기준으로 다시 계산한 값이 02와 같은지 먼저 확인한다.

출력 (data/processed/transit/)
  grid_walk_access.csv     격자 1행: 고령 인구, 정류장까지 노인 보행 시간(도로망·간이 × 언덕·평지)
  stop_network_access.csv  정류장 1행: 지하철역·일반버스 정류장까지 도로망 거리, 대체수단 없음 판정 3종
  route_class_network.csv  노선 1행: 대체수단 없는 승차 비중(직선·도로망), 네 유형(직선·도로망)

사용: python src/transit/08_access_gap.py
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from road_network import edge_graph, elderly_mps, elderly_walk_graph, link_grade, load_network, nearest_source_cost
from route_classes import assign_classes
from route_data import load_dong_names, load_route_stops, load_stop_ridership
from route_kinds import branch_stop_weights, ridership_key

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRANSIT = PROJECT_ROOT / "data" / "processed" / "transit"
STATION_CSV = PROJECT_ROOT / "data" / "raw" / "traffic" / "서울시 역사마스터 정보.csv"

DETOUR_FACTOR = 1.3
SLOPE_RADIUS_M = 300.0
STATION_WALK_M, TRUNK_WALK_M = 500.0, 200.0
STATION_SNAP_MAX_M = 300.0    # 서울 도로망 노드에서 이보다 먼 역(경기 구간 등)은 제외
TRUNK_SEARCH_M = 1500.0
WALK_THRESHOLDS_MIN = (5, 10)
GAP_STATION_M = 800.0


def main() -> None:
    print("[1/4] 도로망·격자·정류장...")
    links, nodes, node_xy = load_network(extra_cols=("slope_pct",))
    fi, ti = nodes.get_indexer(links["F_NODE"]), nodes.get_indexer(links["T_NODE"])
    length = links["LENGTH"].to_numpy()
    grade = link_grade(links)
    node_tree = cKDTree(node_xy)

    cells = pd.read_csv(TRANSIT / "grid_population_daytime.csv", dtype={"dong": str})
    cell_geo = gpd.GeoSeries(gpd.points_from_xy(cells["cx"], cells["cy"]), crs=5179).to_crs(links.crs)
    cell_xy = np.c_[cell_geo.x, cell_geo.y]
    stops = pd.read_csv(TRANSIT / "bus_stops.csv", dtype={"NODE_ID": str})
    stop_xy = stops[["x", "y"]].to_numpy()
    stop_snap, stop_node = node_tree.query(stop_xy)
    cell_snap, cell_node = node_tree.query(cell_xy)
    stations = pd.read_csv(STATION_CSV, encoding="cp949")
    station_geo = gpd.GeoSeries(gpd.points_from_xy(stations["경도"], stations["위도"]), crs=4326).to_crs(links.crs)
    station_xy = np.c_[station_geo.x, station_geo.y]

    print("[2/4] 노인 보행 시간(격자 → 가장 가까운 버스 정류장)...")
    flat = float(elderly_mps(0.0))
    for label, g in (("net", grade), ("net_flat", np.zeros_like(grade))):
        graph = elderly_walk_graph(fi, ti, length, g, len(nodes))
        to_stop = nearest_source_cost(graph, stop_node, stop_snap / flat)
        cells[f"walk_min_{label}"] = (to_stop[cell_node] + cell_snap / flat) / 60.0
    cells["dist_to_stop_m"] = cKDTree(stop_xy).query(cell_xy)[0]
    mid = links.geometry.centroid
    near_links = cKDTree(np.c_[mid.x, mid.y]).query_ball_point(cell_xy, SLOPE_RADIUS_M)
    abs_grade = np.abs(grade)
    cells["local_grade"] = [abs_grade[h].mean() if h else 0.0 for h in near_links]
    cells["walk_min_straight"] = cells["dist_to_stop_m"] * DETOUR_FACTOR / elderly_mps(cells["local_grade"].to_numpy()) / 60.0
    cells["walk_min_straight_flat"] = cells["dist_to_stop_m"] * DETOUR_FACTOR / flat / 60.0
    cells["dist_to_station_m"] = cKDTree(station_xy).query(cell_xy)[0]
    cells.drop(columns=["cx", "cy"]).to_csv(TRANSIT / "grid_walk_access.csv", index=False, encoding="utf-8-sig")

    print("[3/4] 도로망 거리로 본 대체수단...")
    walk = edge_graph(np.r_[fi, ti], np.r_[ti, fi], np.r_[length, length], len(nodes))
    station_snap, station_node = node_tree.query(station_xy)
    inside = station_snap <= STATION_SNAP_MAX_M
    to_station = nearest_source_cost(walk, station_node[inside], station_snap[inside])
    trunk = stops["served_kinds"].str.contains("일반").to_numpy()
    to_trunk = nearest_source_cost(walk, stop_node[trunk], stop_snap[trunk], limit=TRUNK_SEARCH_M)
    stops["net_station_m"] = to_station[stop_node] + stop_snap
    stops["net_trunk_m"] = np.where(trunk, 0.0, to_trunk[stop_node] + stop_snap)
    stops["no_alt_straight"] = (stops["dist_to_station_m"] > STATION_WALK_M) & ~stops["trunk_stop_within_200m"]
    stops["no_alt_net"] = (stops["net_station_m"] > STATION_WALK_M) & ~(stops["net_trunk_m"] <= TRUNK_WALK_M)
    stops["no_alt_net_scaled"] = ((stops["net_station_m"] > STATION_WALK_M * DETOUR_FACTOR)
                                  & ~(stops["net_trunk_m"] <= TRUNK_WALK_M * DETOUR_FACTOR))
    flag_cols = ["no_alt_straight", "no_alt_net", "no_alt_net_scaled"]
    stops[["NODE_ID", "served_kinds", "dist_to_station_m", "net_station_m", "net_trunk_m", *flag_cols]].to_csv(
        TRANSIT / "stop_network_access.csv", index=False, encoding="utf-8-sig")

    print("[4/4] 도로망 기준으로 네 유형 다시 나누기(민감도)...")
    priority = pd.read_csv(TRANSIT / "route_priority.csv", dtype={"ROUTE_ID": str})
    route_ids = set(pd.read_csv(TRANSIT / "routes.csv", dtype={"ROUTE_ID": str})["ROUTE_ID"])
    route_stops = load_route_stops()
    route_stops = route_stops[route_stops["ROUTE_ID"].isin(route_ids)]  # 분기 비율은 02와 같은 노선 집합으로
    member = route_stops[["ROUTE_ID", "노선명", "NODE_ID"]].drop_duplicates(["ROUTE_ID", "NODE_ID"])
    member = member.join(branch_stop_weights(route_stops).rename("w"), on=["ROUTE_ID", "NODE_ID"])
    member["ride_key"] = member["노선명"].map(ridership_key)
    ride = load_stop_ridership()
    ride["board"] = ride.filter(like="board_").sum(axis=1)
    m = (member[member["ROUTE_ID"].isin(priority["ROUTE_ID"])]
         .merge(ride[["ride_key", "NODE_ID", "board"]], on=["ride_key", "NODE_ID"])
         .merge(stops[["NODE_ID", *flag_cols]], on="NODE_ID", how="left"))
    m["b"] = m["board"] * m["w"]
    total = m.groupby("ROUTE_ID")["b"].sum()
    shares = {f: (m["b"] * m[f].fillna(False)).groupby(m["ROUTE_ID"]).sum() / total.where(total > 0) for f in flag_cols}
    check = priority.set_index("ROUTE_ID")["no_alt_boarding_share"]
    gap = (shares["no_alt_straight"].reindex(check.index) - check).abs().max()
    if not gap < 1e-9:
        raise SystemExit(f"직선 기준 대체수단 없는 승차 비중이 02와 다릅니다(최대 차이 {gap:.3g}).")
    base = priority.drop(columns=["dependence_score", "class_id", "class"])
    out = priority[["ROUTE_ID", "노선명", "no_alt_boarding_share", "class_id", "class"]].copy()
    for f in ("no_alt_net", "no_alt_net_scaled"):
        redone = assign_classes(base.assign(no_alt_boarding_share=base["ROUTE_ID"].map(shares[f])))
        out[f"{f}_boarding_share"] = redone["no_alt_boarding_share"].to_numpy()
        out[f"class_id_{f}"] = redone["class_id"].to_numpy()
    out.to_csv(TRANSIT / "route_class_network.csv", index=False, encoding="utf-8-sig")

    print("\n  정류장까지 노인 보행 시간(고령 인구 가중)")
    elderly = cells["elderly"].fillna(0)
    for th in WALK_THRESHOLDS_MIN:
        parts = [f"{name} {elderly[cells[col] > th].sum() / elderly.sum():.1%}"
                 for name, col in (("도로망·언덕", "walk_min_net"), ("도로망·평지", "walk_min_net_flat"),
                                   ("간이·언덕", "walk_min_straight"), ("간이·평지", "walk_min_straight_flat"))]
        print(f"    {th}분 초과: " + ", ".join(parts))
    blind = cells[(cells["walk_min_net"] > 10) & (cells["dist_to_station_m"] > GAP_STATION_M)]
    top = blind.groupby("dong")["elderly"].sum().nlargest(5).round(0)
    names = load_dong_names()
    top.index = [names.get(code, code) for code in top.index]
    print(f"  사각지대(도로망·언덕 10분 초과 & 지하철역 {GAP_STATION_M:.0f}m 밖): 격자 {len(blind)}개, "
          f"고령 {blind['elderly'].sum():,.0f}명({blind['elderly'].sum() / elderly.sum():.1%}), "
          f"고령이 많은 행정동 {top.to_dict()}")
    village = stops[stops["served_kinds"].str.contains("마을버스")]
    ratio = (village["net_station_m"] / village["dist_to_station_m"]).replace(np.inf, np.nan)
    print(f"\n  마을버스 정류장 {len(village)}개 → 지하철역 도로망/직선 거리 비 중앙 {ratio.median():.2f} (90분위 {ratio.quantile(0.9):.2f})")
    print(f"  대체수단 없는 정류장: 직선 {village['no_alt_straight'].mean():.1%} → 도로망(기준 그대로) "
          f"{village['no_alt_net'].mean():.1%} / 도로망(기준 ×{DETOUR_FACTOR}) {village['no_alt_net_scaled'].mean():.1%}")
    for f in ("no_alt_net", "no_alt_net_scaled"):
        changed = int((out["class_id"] != out[f"class_id_{f}"]).sum())
        print(f"  {f}: 노선 대체수단 없는 승차 비중 중앙 {out['no_alt_boarding_share'].median():.3f} → "
              f"{out[f'{f}_boarding_share'].median():.3f}, 유형이 바뀐 노선 {changed}/{len(out)}")
    print(pd.crosstab(out["class"], out["class_id_no_alt_net"], rownames=["직선 기준"], colnames=["도로망(기준 그대로)"]).to_string())


if __name__ == "__main__":
    main()
