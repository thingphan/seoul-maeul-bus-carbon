"""마을버스 노선 하나를 없앴을 때 어르신이 버스 정류장까지 더 걸어야 하는 시간을 노선별로 계산한다.

입력
  data/processed/transit/grid_population_daytime.csv(02), bus_stops.csv(01), route_priority.csv(06),
  grid_walk_access.csv(08, 격자 주변 평균 경사), 정류소·승하차(route_data.py), 도로망(road_network.py)

처리
  1. 정류장마다 서는 노선(일반·마을·자율주행 전부)을 모은다. 노선 R을 없애면 R만 서는 정류장이 사라진다.
     그 정류장에서 R을 타던 하루 승차 인원도 함께 센다(분기 노선은 배분 비율 적용).
  2. 보행 시간은 09와 같은 노인 걸음(Tobler 보행속도 × 0.75)으로 구한다.
       도로망: 링크마다 걷는 방향 경사로 시간을 내고, 남은 정류장 전체를 출발점으로 최단 시간을 구한다.
       간이:   가장 가까운 남은 정류장까지 직선거리 × 1.3, 격자 주변 평균 경사를 오르막으로 둔다.
     두 방식의 차이를 범위로 본다(도로망에는 계단·좁은 골목 일부가 없다).
  3. 격자별 증가 시간 × 격자 고령 인구(평일 낮 생활인구 기준)로 노선별 피해를 요약한다.
       elderly_affected        보행 시간이 1분 이상 늘어나는 고령 인구
       added_min_mean          그 고령 인구의 평균 증가 시간(분)
       elderly_over_10min_new  원래 10분 이내였는데 10분을 넘게 되는 고령 인구
     다른 마을버스 정류장까지 걸어가도 목적지가 다를 수 있으므로 실제 불편보다 작게 잡힌 값이다.

출력
  data/processed/transit/route_removal_access.csv  노선 1행
  data/processed/transit/route_removal_cells.csv   노선×격자: 보행 시간이 1분 이상 늘어나는 격자의 증가 시간(분)

사용: python src/transit/10_route_removal_access.py
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from road_network import elderly_mps, elderly_walk_graph, link_grade, load_network, nearest_source_cost
from route_data import load_route_stops, load_stop_ridership
from route_kinds import branch_stop_weights, ridership_key

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRANSIT = PROJECT_ROOT / "data" / "processed" / "transit"
DETOUR_FACTOR = 1.3
AFFECTED_MIN = 1.0
LONG_WALK_MIN = 10.0


def summarize(base: np.ndarray, after: np.ndarray, elderly: np.ndarray, prefix: str) -> dict:
    # 도로망과 이어지지 않은 격자(무한대)는 비교에서 뺀다.
    ok = np.isfinite(base) & np.isfinite(after)
    with np.errstate(invalid="ignore"):
        added = np.where(ok, after - base, 0.0)
    hit = added >= AFFECTED_MIN
    affected = float(elderly[hit].sum())
    return {
        f"{prefix}_elderly_affected": affected,
        f"{prefix}_added_min_mean": float(np.average(added[hit], weights=elderly[hit])) if affected > 0 else 0.0,
        f"{prefix}_elderly_over_10min_new": float(elderly[ok & (base <= LONG_WALK_MIN) & (after > LONG_WALK_MIN)].sum()),
    }


def main() -> None:
    print("[1/4] 도로망·격자·정류장...")
    links, nodes, node_xy = load_network(extra_cols=("slope_pct",))
    fi, ti = nodes.get_indexer(links["F_NODE"]), nodes.get_indexer(links["T_NODE"])
    graph = elderly_walk_graph(fi, ti, links["LENGTH"].to_numpy(), link_grade(links), len(nodes))
    node_tree = cKDTree(node_xy)
    flat = float(elderly_mps(0.0))

    cells = pd.read_csv(TRANSIT / "grid_population_daytime.csv").merge(
        pd.read_csv(TRANSIT / "grid_walk_access.csv", usecols=["CELL_ID", "local_grade"]), on="CELL_ID", how="left")
    cell_geo = gpd.GeoSeries(gpd.points_from_xy(cells["cx"], cells["cy"]), crs=5179).to_crs(links.crs)
    cell_xy = np.c_[cell_geo.x, cell_geo.y]
    cell_snap, cell_node = node_tree.query(cell_xy)
    elderly = cells["elderly"].fillna(0).to_numpy()
    uphill_mps = elderly_mps(cells["local_grade"].fillna(0).to_numpy())

    stops = pd.read_csv(TRANSIT / "bus_stops.csv", dtype={"NODE_ID": str})
    stop_xy = stops[["x", "y"]].to_numpy()
    stop_snap, stop_node = node_tree.query(stop_xy)

    print("[2/4] 노선별로 그 노선만 서는 정류장과 그 정류장 승차...")
    route_stops = load_route_stops()
    served: dict[str, set[str]] = defaultdict(set)
    for route_id, node_id in zip(route_stops["ROUTE_ID"], route_stops["NODE_ID"]):
        served[node_id].add(route_id)
    exclusive_of = stops["NODE_ID"].map(lambda n: next(iter(served[n])) if len(served[n]) == 1 else None)

    priority = pd.read_csv(TRANSIT / "route_priority.csv", dtype={"ROUTE_ID": str})
    route_ids = set(pd.read_csv(TRANSIT / "routes.csv", dtype={"ROUTE_ID": str})["ROUTE_ID"])
    rs = route_stops[route_stops["ROUTE_ID"].isin(route_ids)]
    member = rs[["ROUTE_ID", "노선명", "NODE_ID"]].drop_duplicates(["ROUTE_ID", "NODE_ID"])
    member = member.join(branch_stop_weights(rs).rename("w"), on=["ROUTE_ID", "NODE_ID"])
    member["ride_key"] = member["노선명"].map(ridership_key)
    ride = load_stop_ridership()
    ride["board"] = ride.filter(like="board_").sum(axis=1)
    member = member.merge(ride[["ride_key", "NODE_ID", "board"]], on=["ride_key", "NODE_ID"], how="left")
    member["board"] = member["board"].fillna(0) * member["w"]

    print("[3/4] 기준 보행 시간(모든 정류장 있음)...")
    base_net = (nearest_source_cost(graph, stop_node, stop_snap / flat)[cell_node] + cell_snap / flat) / 60.0
    base_dist = cKDTree(stop_xy).query(cell_xy)[0]
    base_simple = base_dist * DETOUR_FACTOR / uphill_mps / 60.0

    print("[4/4] 노선을 하나씩 없애고 다시 계산...")
    rows, cell_rows = [], []
    for route_id, name, class_id, class_name in priority[["ROUTE_ID", "노선명", "class_id", "class"]].itertuples(index=False, name=None):
        gone = (exclusive_of == route_id).to_numpy()
        gone_nodes = set(stops.loc[gone, "NODE_ID"])
        boardings_lost = member.loc[(member["ROUTE_ID"] == route_id) & member["NODE_ID"].isin(gone_nodes), "board"].sum()
        row = {"ROUTE_ID": route_id, "노선명": name, "class_id": class_id, "class": class_name,
               "n_exclusive_stops": int(gone.sum()), "boardings_at_exclusive_stops": float(boardings_lost)}
        if gone.any():
            keep = ~gone
            after_net = (nearest_source_cost(graph, stop_node[keep], stop_snap[keep] / flat)[cell_node] + cell_snap / flat) / 60.0
            after_simple = cKDTree(stop_xy[keep]).query(cell_xy)[0] * DETOUR_FACTOR / uphill_mps / 60.0
            row.update(summarize(base_net, after_net, elderly, "net"))
            row.update(summarize(base_simple, after_simple, elderly, "simple"))
            with np.errstate(invalid="ignore"):
                add_net, add_simple = after_net - base_net, after_simple - base_simple
            changed = np.isfinite(add_net) & ((add_net >= AFFECTED_MIN) | (add_simple >= AFFECTED_MIN))
            cell_rows.append(pd.DataFrame({"ROUTE_ID": route_id, "CELL_ID": cells.loc[changed, "CELL_ID"].to_numpy(),
                                           "elderly": elderly[changed], "added_min_net": add_net[changed],
                                           "added_min_simple": add_simple[changed],
                                           "after_min_net": after_net[changed], "after_min_simple": after_simple[changed]}))
        rows.append(row)
    out = pd.DataFrame(rows).fillna(0.0)
    out.to_csv(TRANSIT / "route_removal_access.csv", index=False, encoding="utf-8-sig")
    pd.concat(cell_rows, ignore_index=True).to_csv(TRANSIT / "route_removal_cells.csv", index=False, encoding="utf-8-sig")

    print(f"\n  노선 {len(out)}개 중 그 노선만 서는 정류장이 있는 노선 {int((out['n_exclusive_stops'] > 0).sum())}개")
    cols = ["n_exclusive_stops", "boardings_at_exclusive_stops", "net_elderly_affected", "net_added_min_mean",
            "simple_elderly_affected", "simple_added_min_mean", "net_elderly_over_10min_new"]
    print(out.groupby("class")[cols].median().round(1).to_string())
    two = out[out["class_id"] == 2].sort_values("net_elderly_affected", ascending=False)
    print("\n  ② 유형 노선 상위 8개(도로망 기준 영향 고령 인구 순):")
    print(two[["노선명", *cols]].head(8).round(1).to_string(index=False))


if __name__ == "__main__":
    main()
