"""마을버스·자율주행버스 노선을 표준노드링크 위에 링크 단위로 붙인다(맵매칭).

입력
  data/raw/traffic/bus_route_path/route_paths.csv              (00 산출물: 노선별 실제 경로 좌표)
  data/raw/traffic/서울시버스노선별정류소정보(20260902).xlsx  (노선별 경유 정류소 순번·좌표)
  data/raw/traffic/서울시 역사마스터 정보.csv                  (지하철역 좌표)
  data/interim/seoul_link.shp, data/processed/seoul_link_dimension.csv

처리
  1. 노선 유형 구분(route_kinds.py): 승하차 데이터의 공식 분류 '서울마을버스'면 마을버스(강남06-1 같은
     파생 노선, 서대문02대·02소 같은 분기 노선 포함), 이름에 'A+숫자'가 있으면 자율주행버스, 나머지는 일반.
  2. 노선 길이와 정류소 사이 거리는 실제 경로선에서 직접 잰다(정류소를 경로선에 앞으로만 투영).
  3. 실제 경로 맵매칭: 경로선을 RESAMPLE_M 간격 점으로 나누고, 각 점을 반경 MATCH_RADIUS_M 안에서
     진행 방향이 맞는 가장 가까운 링크에 붙인다. 표준노드링크의 골목길 일부는 한 방향 링크로만
     그려져 있어 진행 방향과 반대인 링크도 후보로 두되(역방향 통과, 경사 부호가 뒤집힘) 거리에
     REVERSE_EXTRA_M을 더해 정방향을 우선한다. 직전 링크가 후보에 있으면 유지해 평행 링크 사이를
     오가는 것을 막는다. 붙인 링크 사이가 끊기면, 실제 경로상 간격에 비해 짧은 최단경로만 메운다.
     매칭 링크열은 노선이 지나는 도로의 속도·경사 표본(1km당 에너지 산정용)으로 쓰고,
     링크마다 '몇 번째 정류소 구간'인지(from_stop_order)를 정류소 투영 위치로 단다.
  4. 실제 경로가 없는 노선은 연속 정류소 사이를 최단경로로 잇는 방식으로 대신한다.
  5. 정류소별 최근접 지하철역 거리와, 일반버스 정류소가 200m 안에 있는지를 계산한다.

출력 (data/processed/transit/)
  bus_stops.csv    정류소 1행: 좌표, 최근접 링크 거리, 최근접 지하철역·거리, 일반버스 접근
  route_links.csv     노선×링크 순서(역방향 통과 여부, 정류소 구간 번호 포함)
  route_segments.csv  노선×정류소 구간: 실제 경로선 기준 구간 길이[km]
  routes.csv          노선 1행: 노선 길이(실제 경로), 매칭 링크 길이, 경로 대비 매칭 비율, QA 플래그

사용: python src/transit/01_build_bus_route_links.py
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from shapely.geometry import LineString

from road_network import load_network
from route_data import load_route_stops
from route_kinds import classify_route

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_TRAFFIC = PROJECT_ROOT / "data" / "raw" / "traffic"
ROUTE_PATH_CSV = RAW_TRAFFIC / "bus_route_path" / "route_paths.csv"
STATION_CSV = RAW_TRAFFIC / "서울시 역사마스터 정보.csv"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "transit"

REVERSE_PENALTY = 1.5          # 최단경로에서 역방향 통과 비용 배수
TRUNK_STOP_RADIUS_M = 200.0    # 일반버스 정류소 접근 반경
RESAMPLE_M = 20.0              # 경로선 점 간격
MATCH_RADIUS_M = 25.0          # 링크 후보 반경
REVERSE_EXTRA_M = 8.0          # 역방향 후보 거리 불이익
ALIGN_COS = 0.7                # 진행 방향 일치 기준(약 45도)
STAY_TOLERANCE_M = 5.0         # 직전 링크 유지 허용 거리차
GAP_LIMIT_M = 800.0            # 끊긴 링크 사이를 메우는 최단경로 비용 상한
FILL_FACTOR, FILL_SLACK_M = 1.5, 30.0  # 메우는 경로 길이 ≤ 1.5 × 실제 경로상 간격 + 30m
MATCH_QA_RANGE = (0.8, 1.1)    # 매칭 링크 길이 / 실제 경로 길이 허용 범위(표본 충분성)
DETOUR_QA_LIMIT = 2.0          # (정류소 방식) 우회비 상한


def build_graph(links: gpd.GeoDataFrame, nodes: pd.Index) -> tuple[csr_matrix, dict[tuple[int, int], tuple[int, bool]]]:
    """정방향 링크 + 패널티를 준 역방향 통과로 희소 그래프를 만든다."""
    fi = nodes.get_indexer(links["F_NODE"])
    ti = nodes.get_indexer(links["T_NODE"])
    forward = pd.DataFrame({"u": fi, "v": ti, "w": links["LENGTH"].to_numpy() + 1e-6, "idx": links.index, "rev": False})
    backward = pd.DataFrame({"u": ti, "v": fi, "w": forward["w"] * REVERSE_PENALTY, "idx": links.index, "rev": True})
    edges = pd.concat([forward, backward]).sort_values("w").drop_duplicates(["u", "v"])
    graph = csr_matrix((edges["w"], (edges["u"], edges["v"])), shape=(len(nodes), len(nodes)))
    lookup = {(u, v): (i, r) for u, v, i, r in zip(edges["u"], edges["v"], edges["idx"], edges["rev"])}
    return graph, lookup


def shortest_links(graph, lookup, u: int, v: int, limit: float = np.inf) -> list[tuple[int, bool]] | None:
    dist, pred = dijkstra(graph, directed=True, indices=u, return_predecessors=True, limit=limit)
    if not np.isfinite(dist[v]):
        return None
    segment, x = [], v
    while x != u:
        p = pred[x]
        segment.append(lookup[(p, x)])
        x = p
    return segment[::-1]


def unit_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """점 배열 a→b 방향 단위벡터(길이 0이면 0벡터)."""
    vec = shapely.get_coordinates(b) - shapely.get_coordinates(a)
    norm = np.sqrt((vec * vec).sum(axis=1, keepdims=True))
    return np.divide(vec, norm, out=np.zeros_like(vec), where=norm > 0)


def resample(line: LineString) -> tuple[np.ndarray, np.ndarray]:
    """경로선을 RESAMPLE_M 간격 점으로 나누고, 각 점의 진행 방향(앞뒤 RESAMPLE_M/2 구간)을 구한다."""
    distances = np.arange(0.0, line.length, RESAMPLE_M)
    points = shapely.line_interpolate_point(line, distances)
    headings = unit_vectors(shapely.line_interpolate_point(line, np.maximum(distances - RESAMPLE_M / 2, 0.0)),
                            shapely.line_interpolate_point(line, np.minimum(distances + RESAMPLE_M / 2, line.length)))
    return points, headings


def match_path(line: LineString, geoms: np.ndarray, tree: shapely.STRtree) -> list[tuple[int, bool, float, float]]:
    """경로선 → (링크 인덱스, 역방향 여부, 경로상 처음·마지막 매칭 위치[m]) 열. 연속 중복은 하나로 합친다.

    후보 링크의 거리·방향은 모든 (점, 후보) 쌍에 대해 한 번에 계산하고, 직전 링크 유지 규칙처럼
    앞 점의 결과에 따라 달라지는 선택만 점 순서대로 처리한다.
    """
    points, headings = resample(line)
    point_idx, link_idx = tree.query(points, predicate="dwithin", distance=MATCH_RADIUS_M)
    order = np.argsort(point_idx, kind="stable")  # 점 순서로 묶되 점 안의 후보 순서는 그대로 둔다
    point_idx, link_idx = point_idx[order], link_idx[order]
    cand, at = geoms[link_idx], points[point_idx]
    # 후보 링크 위 가장 가까운 지점의 앞뒤 5m로 링크 방향을 잡는다.
    along = shapely.line_locate_point(cand, at)
    link_dir = unit_vectors(shapely.line_interpolate_point(cand, np.maximum(along - 5.0, 0.0)),
                            shapely.line_interpolate_point(cand, np.minimum(along + 5.0, shapely.length(cand))))
    cos = (headings[point_idx] * link_dir).sum(axis=1)
    forward, backward = cos >= ALIGN_COS, cos <= -ALIGN_COS
    keep = forward | backward
    point_idx, link_idx, rev = point_idx[keep], link_idx[keep], backward[keep]
    score = shapely.distance(cand[keep], at[keep]) + np.where(rev, REVERSE_EXTRA_M, 0.0)
    bounds = np.searchsorted(point_idx, np.arange(len(points) + 1))

    matched: list[list] = []
    for k in range(len(points)):
        lo, hi = bounds[k], bounds[k + 1]
        if lo == hi:
            continue
        j = lo + int(np.argmin(score[lo:hi]))  # 동점이면 먼저 나온 후보
        best, best_score = (int(link_idx[j]), bool(rev[j])), score[j]
        if matched:
            same = (link_idx[lo:hi] == matched[-1][0]) & (rev[lo:hi] == matched[-1][1])
            if same.any() and score[lo:hi][same][-1] <= best_score + STAY_TOLERANCE_M:
                best = (matched[-1][0], matched[-1][1])
        position = k * RESAMPLE_M
        if matched and (matched[-1][0], matched[-1][1]) == best:
            matched[-1][3] = position
        else:
            matched.append([best[0], best[1], position, position])
    return [tuple(m) for m in matched]


def connect(matched, fi: np.ndarray, ti: np.ndarray, lengths: np.ndarray, graph, lookup) -> tuple[list[tuple[int, bool]], int]:
    """연속 매칭 링크의 끝-시작 노드가 다르면 최단경로로 메운다.

    메우는 경로가 실제 경로상 두 매칭 지점 간격보다 지나치게 길면(반대 차로·막다른 길로 돌아가는
    군더더기) 메우지 않고 틈으로 센다. 노선 길이는 실제 경로선에서 따로 재므로 링크열은 노선이
    지나는 도로의 속도·경사 표본으로만 쓴다.
    """
    chain: list[tuple[int, bool]] = []
    gaps = 0
    prev_last = 0.0
    for idx, rev, first_pos, last_pos in matched:
        if chain:
            p_idx, p_rev = chain[-1]
            end = fi[p_idx] if p_rev else ti[p_idx]
            start = ti[idx] if rev else fi[idx]
            if end != start:
                filler = shortest_links(graph, lookup, end, start, limit=GAP_LIMIT_M)
                path_gap = max(first_pos - prev_last, 0.0)
                if filler is not None and sum(lengths[i] for i, _ in filler) <= FILL_FACTOR * path_gap + FILL_SLACK_M:
                    chain.extend(filler)
                else:
                    gaps += 1
        chain.append((idx, rev))
        prev_last = last_pos
    return chain, gaps


def stop_positions(line: LineString, stop_xy: np.ndarray) -> np.ndarray:
    """정류소를 순번을 지키며 경로선 위치[m]에 배치한다.

    왕복 노선은 같은 길을 두 번 지나므로 정류소마다 가장 가까운 점을 따로 고르면 순서가 꼬인다.
    '정류소 순번 = 경로상 위치 순서'를 지키면서 정류소-경로 거리 합이 최소가 되는 배치를
    동적 계획법으로 찾는다.
    """
    distances = np.arange(0.0, line.length + RESAMPLE_M / 2, RESAMPLE_M / 2)
    samples = shapely.get_coordinates(shapely.line_interpolate_point(line, distances))
    dist = np.hypot(samples[None, :, 0] - stop_xy[:, 0:1], samples[None, :, 1] - stop_xy[:, 1:2])
    cost = dist[0].copy()
    back = np.zeros(dist.shape, dtype=np.int64)
    index = np.arange(len(cost))
    for k in range(1, len(stop_xy)):
        # back[k][j] = j 이하에서 cost가 가장 작은 위치(동점이면 뒤쪽). 누적 최소가 새로 갱신되는 위치를
        # 앞에서부터 이어 붙이면 된다.
        best_prev = np.minimum.accumulate(cost)
        is_new = cost <= np.r_[np.inf, best_prev[:-1]]
        back[k] = np.maximum.accumulate(np.where(is_new, index, 0))
        cost = dist[k] + best_prev
    j = int(np.argmin(cost))
    positions = [j]
    for k in range(len(stop_xy) - 1, 0, -1):
        j = int(back[k][j])
        positions.append(j)
    return distances[np.array(positions[::-1])]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[1/5] 도로망 로드...")
    links, nodes, node_xy = load_network()
    graph, lookup = build_graph(links, nodes)
    node_tree = cKDTree(node_xy)
    geoms = links.geometry.to_numpy()
    link_tree = shapely.STRtree(geoms)
    fi = nodes.get_indexer(links["F_NODE"])
    ti = nodes.get_indexer(links["T_NODE"])

    print("[2/5] 정류소·지하철역·실제 경로 로드...")
    route_stops = load_route_stops()
    route_stops["kind"] = classify_route(route_stops["노선명"])
    stop_points = gpd.GeoDataFrame(
        route_stops, geometry=gpd.points_from_xy(route_stops["X좌표"], route_stops["Y좌표"]), crs=4326
    ).to_crs(links.crs)
    route_stops["x"] = stop_points.geometry.x
    route_stops["y"] = stop_points.geometry.y
    route_stops["snap_node_idx"] = node_tree.query(route_stops[["x", "y"]].to_numpy())[1]

    paths = pd.read_csv(ROUTE_PATH_CSV, dtype={"ROUTE_ID": str}) if ROUTE_PATH_CSV.exists() else pd.DataFrame()
    route_lines: dict[str, LineString] = {}
    if not paths.empty:
        paths = paths.sort_values(["ROUTE_ID", "no"])
        path_xy = gpd.GeoSeries(gpd.points_from_xy(paths["gpsX"], paths["gpsY"]), crs=4326).to_crs(links.crs)
        paths["x"], paths["y"] = path_xy.x.to_numpy(), path_xy.y.to_numpy()
        for route_id, g in paths.groupby("ROUTE_ID"):
            xy = g[["x", "y"]].to_numpy()
            keep = np.r_[True, np.any(np.diff(xy, axis=0) != 0, axis=1)]
            if keep.sum() >= 2:
                route_lines[route_id] = LineString(xy[keep])

    stations = pd.read_csv(STATION_CSV, encoding="cp949")
    station_points = gpd.GeoDataFrame(
        stations, geometry=gpd.points_from_xy(stations["경도"], stations["위도"]), crs=4326
    ).to_crs(links.crs)
    station_tree = cKDTree(np.c_[station_points.geometry.x, station_points.geometry.y])

    print("[3/5] 정류소 테이블...")
    served = route_stops.groupby("NODE_ID")["kind"].agg(lambda s: sorted(set(s)))
    stops = route_stops.drop_duplicates("NODE_ID")[["NODE_ID", "ARS_ID", "정류소명", "X좌표", "Y좌표", "x", "y"]].copy()
    stops["served_kinds"] = stops["NODE_ID"].map(lambda n: "|".join(served[n]))
    stop_geo = gpd.GeoDataFrame(stops, geometry=gpd.points_from_xy(stops["x"], stops["y"]), crs=links.crs)
    nearest_link = gpd.sjoin_nearest(stop_geo[["NODE_ID", "geometry"]], links[["LINK_ID", "geometry"]], distance_col="dist_to_link_m")
    stops["dist_to_link_m"] = stops["NODE_ID"].map(nearest_link.drop_duplicates("NODE_ID").set_index("NODE_ID")["dist_to_link_m"])
    station_dist, station_idx = station_tree.query(stops[["x", "y"]].to_numpy())
    stops["nearest_station"] = stations["역사명"].to_numpy()[station_idx]
    stops["nearest_station_line"] = stations["호선"].to_numpy()[station_idx]
    stops["dist_to_station_m"] = station_dist
    trunk = stops[stops["served_kinds"].str.contains("일반")]
    trunk_tree = cKDTree(trunk[["x", "y"]].to_numpy())
    stops["trunk_stop_within_200m"] = np.array(
        [len(hits) > 0 for hits in trunk_tree.query_ball_point(stops[["x", "y"]].to_numpy(), TRUNK_STOP_RADIUS_M)]
    )
    stops.to_csv(OUT_DIR / "bus_stops.csv", index=False, encoding="utf-8-sig")

    print("[4/5] 마을버스·자율주행 노선 매칭...")
    target = route_stops[route_stops["kind"] != "일반"].sort_values(["ROUTE_ID", "순번"])
    lengths = links["LENGTH"].to_numpy()
    link_ids = links["LINK_ID"].to_numpy()
    link_rows: list[dict] = []
    route_rows: list[dict] = []
    segment_rows: list[dict] = []
    for (route_id, name, kind), group in target.groupby(["ROUTE_ID", "노선명", "kind"], sort=False):
        stop_xy = group[["x", "y"]].to_numpy()
        line = route_lines.get(route_id)
        gaps = 0
        if line is not None:
            method = "실제경로"
            matched = match_path(line, geoms, link_tree)
            chain, gaps = connect(matched, fi, ti, lengths, graph, lookup)
            positions = stop_positions(line, stop_xy)
            segment_rows.extend({"ROUTE_ID": route_id, "stop_order": k + 1, "seg_km": km}
                                for k, km in enumerate(np.diff(positions) / 1000.0))
            # 링크의 정류소 구간: 매칭된 링크는 경로상 위치로, 메운 링크는 직전 링크 구간을 따른다.
            pos_of: dict[tuple[int, bool], float] = {}
            for idx, rev, first_pos, _ in matched:
                pos_of.setdefault((idx, rev), first_pos)
            orders, last = [], 1
            for idx, rev in chain:
                if (idx, rev) in pos_of:
                    last = int(max(np.searchsorted(positions, pos_of[(idx, rev)], side="right"), 1))
                orders.append(min(last, max(len(positions) - 1, 1)))
            stop_order = np.array(orders)
        else:
            method = "정류소연결"
            chain, stop_order_list = [], []
            seq = group["snap_node_idx"].to_numpy()
            for k, (u, v) in enumerate(zip(seq[:-1], seq[1:]), start=1):
                segment = shortest_links(graph, lookup, u, v) if u != v else []
                if segment is None:
                    gaps += 1
                    continue
                chain.extend(segment)
                stop_order_list.extend([k] * len(segment))
                segment_rows.append({"ROUTE_ID": route_id, "stop_order": k,
                                     "seg_km": sum(lengths[i] for i, _ in segment) / 1000.0})
            stop_order = np.array(stop_order_list, dtype=int)

        route_links = [{
            "ROUTE_ID": route_id, "노선명": name, "kind": kind, "link_order": i + 1,
            "from_stop_order": int(stop_order[i]), "LINK_ID": link_ids[idx],
            "LENGTH": lengths[idx], "reversed": bool(rev),
        } for i, (idx, rev) in enumerate(chain)]
        link_rows.extend(route_links)
        matched_m = sum(r["LENGTH"] for r in route_links)
        reversed_m = sum(r["LENGTH"] for r in route_links if r["reversed"])
        straight = float(np.hypot(np.diff(stop_xy[:, 0]), np.diff(stop_xy[:, 1])).sum())
        if line is not None:
            length_km = line.length / 1000
            match_ratio = matched_m / 1000 / length_km
            qa = bool(not (MATCH_QA_RANGE[0] <= match_ratio <= MATCH_QA_RANGE[1]))
        else:
            length_km = matched_m / 1000
            match_ratio = np.nan
            qa = bool(gaps > 0 or straight <= 0 or matched_m / straight > DETOUR_QA_LIMIT)
        route_rows.append({
            "ROUTE_ID": route_id, "노선명": name, "kind": kind, "method": method, "n_stops": len(group),
            "length_km": length_km, "matched_link_km": matched_m / 1000, "match_ratio": match_ratio,
            "straight_km": straight / 1000, "reversed_share": reversed_m / matched_m if matched_m else np.nan,
            "unfilled_gaps": gaps, "qa_flag": qa,
        })

    route_links_df = pd.DataFrame(link_rows)
    routes_df = pd.DataFrame(route_rows)
    route_links_df.to_csv(OUT_DIR / "route_links.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(segment_rows).to_csv(OUT_DIR / "route_segments.csv", index=False, encoding="utf-8-sig")
    routes_df.to_csv(OUT_DIR / "routes.csv", index=False, encoding="utf-8-sig")

    print("[5/5] 요약")
    for kind, g in routes_df.groupby("kind"):
        print(f"  {kind}: 노선 {len(g)}개(실제경로 {int((g['method'] == '실제경로').sum())}), 길이 중앙 {g['length_km'].median():.2f}km, "
              f"매칭 링크/실제 경로 비 중앙 {g['match_ratio'].median():.3f} ({MATCH_QA_RANGE[0]}~{MATCH_QA_RANGE[1]} 안 {g['match_ratio'].between(*MATCH_QA_RANGE).mean():.1%}), "
              f"역방향 비중 평균 {g['reversed_share'].mean():.1%}, QA 대상 {int(g['qa_flag'].sum())}개")
    village = stops[stops["served_kinds"] == "마을버스"]
    print(f"  마을버스 전용 정류소 {len(village)}개: 지하철역 500m 밖 {(village['dist_to_station_m'] > 500).mean():.1%}, "
          f"일반버스 200m 내 없음 {(~village['trunk_stop_within_200m']).mean():.1%}")
    print(f"  저장: {OUT_DIR}")


if __name__ == "__main__":
    main()
